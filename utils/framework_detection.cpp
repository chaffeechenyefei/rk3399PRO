#include "framework_detection.hpp"

using namespace ucloud;


/*******************************************************************************
AnyDetection + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
AnyDetectionV4ByteTrack::AnyDetectionV4ByteTrack(){
    // m_detector = std::make_shared<YoloDetectionV4>();
    m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
}

RET_CODE AnyDetectionV4ByteTrack::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> AnyDetectionV4ByteTrack::run";
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    BYTETRACKPARM track_param = {threshold, threshold+0.1f};
    LOGI << "threshold: " << threshold << ", high det: " << threshold +0.1f;
    RET_CODE ret = m_detector->run(tvimage, bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
#ifdef TIMING    
    m_Tk.start();
#endif
    if(m_trackor){
        m_trackor->update(tvimage, bboxes, track_param);
        m_trackor->clear();
    }
#ifdef TIMING    
    m_Tk.end("tracking");
#endif
    LOGI << "<- AnyDetectionV4ByteTrack::run";
    return RET_CODE::SUCCESS;
}


RET_CODE AnyDetectionV4ByteTrack::run(TvaiImage &tvimage, VecObjBBox &bboxes,string &filename, float threshold, float nms_threshold){
    LOGI << "-> AnyDetectionV4ByteTrack::run";
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    BYTETRACKPARM track_param = {threshold, threshold+0.1f};
    RET_CODE ret = m_detector->run(tvimage, bboxes, filename, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
#ifdef TIMING    
    m_Tk.start();
#endif
    if(m_trackor){
        m_trackor->update(tvimage, bboxes, track_param);
        m_trackor->clear();
    }
#ifdef TIMING    
    m_Tk.end("tracking");
#endif
    LOGI << "<- AnyDetectionV4ByteTrack::run";
    return RET_CODE::SUCCESS;
}



RET_CODE AnyDetectionV4ByteTrack::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> AnyDetectionV4ByteTrack::init";
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::init(const std::string &modelpath){
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    return m_detector->get_class_type(valid_clss);
}

RET_CODE AnyDetectionV4ByteTrack::set_detector(AlgoAPI* ptr){
    m_detector.reset(ptr);
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::set_trackor(TRACKMETHOD trackmethod){
    switch (trackmethod)
    {
    case TRACKMETHOD::BYTETRACK_ORIGIN :
        m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
        break;
    case TRACKMETHOD::BYTETRACK_NO_REID :
        m_trackor = std::make_shared<ByteTrackNoReIDPool>(m_fps,m_nn_buf);
        break;        
    default:
        printf("unsupported tracking method, ByteTrackOriginPool will be used\n");
        m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
        break;
    }
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
    return m_detector->set_output_cls_order(output_clss);
}

float AnyDetectionV4ByteTrack::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float AnyDetectionV4ByteTrack::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}


/*******************************************************************************
PipelineNaive
chaffee.chen@2022-10-09
*******************************************************************************/
RET_CODE PipelineNaive::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    std::set<CLS_TYPE> clss;
    for(auto&& handle: m_handles){
        std::vector<CLS_TYPE> vec_tmp;
        handle->get_class_type(vec_tmp);
        clss.insert(vec_tmp.begin(),vec_tmp.end());
    }
    for(auto&& cls: clss){
        valid_clss.push_back(cls);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE PipelineNaive::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::SUCCESS;
    VecObjBBox bboxes_filtered;
    for(int i=0; i < m_handles.size(); i++){
        //run
        if(i==unfixed_thresholds_index)
            ret = m_handles[i]->run(tvimage,bboxes_filtered,threshold, nms_threshold);
        else
            ret = m_handles[i]->run(tvimage,bboxes_filtered,m_thresholds[i], m_nms_thresholds[i]);
        if(ret!=RET_CODE::SUCCESS){
            printf("ERR[%s][%d]::PipelineNaive::run [%d]th handle return [%d]\n",__FILE__, __LINE__, i, ret);
            return ret;
        }
        //filter
        VecObjBBox tmp;
        if(m_filter_funcs[i]!=nullptr){
            ret = m_filter_funcs[i](bboxes_filtered, tmp);
            if(ret!=RET_CODE::SUCCESS){
                printf("ERR[%s][%d]::PipelineNaive::filter [%d]th handle return [%d]\n",__FILE__, __LINE__, i, ret);
                return ret;
            }
        }
        bboxes_filtered.swap(tmp);
    }
    bboxes = bboxes_filtered;
    return ret;
}