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

RET_CODE AnyDetectionV4ByteTrack::init(std::map<InitParam, WeightData> &weightConfig){
    LOGI << "-> AnyDetectionV4ByteTrack::init";
    return m_detector->init(weightConfig);
}

RET_CODE AnyDetectionV4ByteTrack::init(const std::string &modelpath){
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::init(WeightData weightConfig){
    return m_detector->init(weightConfig);
}

RET_CODE AnyDetectionV4ByteTrack::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    return m_detector->get_class_type(valid_clss);
}

RET_CODE AnyDetectionV4ByteTrack::set_detector(AlgoAPI* ptr){
    m_detector.reset(ptr);
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::set_anchor(std::vector<float> &anchors){
    LOGI << "-> AnyDetectionV4ByteTrack::set_anchor";
    return m_detector->set_anchor(anchors);
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





/*******************************************************************************
 * AnyModelWithBBox
 * chaffee.chen@2022-11-03
*******************************************************************************/
AnyModelWithBBox::AnyModelWithBBox(){
    LOGI<<"-> AnyModelWithBBox";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
}

AnyModelWithBBox::~AnyModelWithBBox(){
    LOGI<<"-> ~AnyModelWithBBox";
}

ucloud::RET_CODE AnyModelWithBBox::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
    LOGI<<"-> AnyModelWithBBox::init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;

    if (weightConfig.find(ucloud::InitParam::BASE_MODEL)== weightConfig.end()){
        printf("**[%s][%d] base model not found in weightConfig\n", __FILE__, __LINE__);
        return ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    bool useDRM = false;
    ret = m_net->base_init(weightConfig[ucloud::InitParam::BASE_MODEL],useDRM);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        return ret;
    }
    m_InpNum = m_net->get_input_shape().size();
    m_OutNum = m_net->get_output_shape().size();
    // if(m_InpNum != m_net->get_input_shape().size()){
    //     printf("** dims err m_InpuNum[%d] != m_net->get_input_shape().size()[%d]\n", m_InpNum, m_net->get_input_shape().size());
    //     return RET_CODE::FAILED;
    // }
    // if(m_OutNum != m_net->get_output_shape().size()){
    //     printf("** dims err m_OutNum[%d] != m_net->get_output_shape().size()[%d]\n", m_OutNum, m_net->get_output_shape().size());
    //     return RET_CODE::FAILED;
    // }

    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    m_param_img2tensor.keep_aspect_ratio = true;
    m_param_img2tensor.pad_both_side = true;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
    m_param_img2tensor.model_input_shape = m_InpSp;
    LOGI << "<- AnyModelWithBBox::init";
    return ret;
}


ucloud::RET_CODE AnyModelWithBBox::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"-> AnyModelWithBBox::init";
    std::map<InitParam, WeightData> weightConfig;
    for(auto &&modelp: modelpath){
        int szBuf = 0;
        unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
        weightConfig[modelp.first] = WeightData{tmpBuf,szBuf};
    }
    RET_CODE ret = init(weightConfig);
    for(auto &&wC: weightConfig){
        free(wC.second.pData);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- AnyModelWithBBox::init";
    return ret;
}

ucloud::RET_CODE AnyModelWithBBox::postprocess(std::vector<float*> &output_datas ,BBox &bbox, float aX, float aY){
    LOGI << "-> AnyModelWithBBox::postprocess";
    RET_CODE ret = RET_CODE::SUCCESS;
    LOGI << "<- AnyModelWithBBox::postprocess";
    return ret;
}


/*******************************************************************************
 * run 对bboxes中的每个区域进行分类, 并将结果更新到bboxes中(objtype, objectness, confidence)
*******************************************************************************/
ucloud::RET_CODE AnyModelWithBBox::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"-> AnyModelWithBBox::run";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    switch (tvimage.format)
    {
    case ucloud::TVAI_IMAGE_FORMAT_BGR:
    case ucloud::TVAI_IMAGE_FORMAT_RGB:
    case ucloud::TVAI_IMAGE_FORMAT_NV12:
    case ucloud::TVAI_IMAGE_FORMAT_NV21:
        ret = ucloud::RET_CODE::SUCCESS;
        break;
    
    default:
        ret = ucloud::RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if (ret!=ucloud::RET_CODE::SUCCESS) return ret;
    
    if(bboxes.empty()) return RET_CODE::SUCCESS;//没有目标直接返回

    for(auto &&box: bboxes){
        std::vector<unsigned char*> input_datas;
        std::vector<float*> output_datas;
        TvaiRect roi = box.rect;
        vector<float> aX,aY;
        
    #ifdef USEDRM  
        roi =  get_valid_rect(roi, tvimage.width, tvimage.height);
        ret = m_cv_preprocess_net->preprocess_drm(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
    #else
        ret = preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
    #endif
        if(ret!=ucloud::RET_CODE::SUCCESS){
            printf("**[%s][%d] Classification preprocess return [%d]\n", __FILE__, __LINE__, ret);
            return ret;
        } 

        ret  = m_net->general_infer_uint8_nhwc_to_float(input_datas,output_datas);
        if(ret!=RET_CODE::SUCCESS) {
            for(auto &&t: input_datas) free(t);
            return ret;
        }

        ret = postprocess(output_datas, box, aX[0], aY[0]);

        if(ret!=RET_CODE::SUCCESS) {
            for(auto &&t: input_datas) free(t);
            for(auto &&t: output_datas) free(t);
            return ret;
        }

        for(auto &&t: output_datas){
            free(t);
        }
        for(auto &&t: input_datas){
            free(t);
        }
    }
    return ret;
}


// /*******************************************************************************
//  * AnyModelWithTvaiImage RGB input only
//  * chaffee.chen@2022-11-03
// *******************************************************************************/
// AnyModelWithTvaiImage::AnyModelWithTvaiImage(){
//     LOGI<<"-> AnyModelWithTvaiImage";
//     m_net = std::make_shared<BaseModel>();
//     m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
// }

// AnyModelWithTvaiImage::~AnyModelWithTvaiImage(){
//     LOGI<<"-> ~AnyModelWithTvaiImage";
// }

// ucloud::RET_CODE AnyModelWithTvaiImage::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
//     LOGI<<"-> AnyModelWithTvaiImage::init";
//     ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;

//     if (weightConfig.find(ucloud::InitParam::BASE_MODEL)== weightConfig.end()){
//         printf("**[%s][%d] base model not found in weightConfig\n", __FILE__, __LINE__);
//         return ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
//     }
//     bool useDRM = false;
//     ret = m_net->base_init(weightConfig[ucloud::InitParam::BASE_MODEL],useDRM);
//     if (ret!=ucloud::RET_CODE::SUCCESS){
//         return ret;
//     }
//     m_InpNum = m_net->get_input_shape().size();
//     m_OutNum = m_net->get_output_shape().size();
//     // if(m_InpNum != m_net->get_input_shape().size()){
//     //     printf("** dims err m_InpuNum[%d] != m_net->get_input_shape().size()[%d]\n", m_InpNum, m_net->get_input_shape().size());
//     //     return RET_CODE::FAILED;
//     // }
//     // if(m_OutNum != m_net->get_output_shape().size()){
//     //     printf("** dims err m_OutNum[%d] != m_net->get_output_shape().size()[%d]\n", m_OutNum, m_net->get_output_shape().size());
//     //     return RET_CODE::FAILED;
//     // }

//     m_InpSp = m_net->get_input_shape()[0];
//     m_OutEleDims = m_net->get_output_dims();
//     m_OutEleNums = m_net->get_output_elem_num();
//     m_param_img2tensor.keep_aspect_ratio = true;
//     m_param_img2tensor.pad_both_side = true;
//     m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
//     m_param_img2tensor.model_input_shape = m_InpSp;
//     LOGI << "<- AnyModelWithTvaiImage::init";
//     return ret;
// }


// ucloud::RET_CODE AnyModelWithTvaiImage::init(std::map<ucloud::InitParam,std::string> &modelpath){
//     LOGI<<"-> AnyModelWithTvaiImage::init";
//     std::map<InitParam, WeightData> weightConfig;
//     for(auto &&modelp: modelpath){
//         int szBuf = 0;
//         unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
//         weightConfig[modelp.first] = WeightData{tmpBuf,szBuf};
//     }
//     RET_CODE ret = init(weightConfig);
//     for(auto &&wC: weightConfig){
//         free(wC.second.pData);
//     }
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     LOGI << "<- AnyModelWithTvaiImage::init";
//     return ret;
// }

// ucloud::RET_CODE AnyModelWithTvaiImage::postprocess(std::vector<float*> &output_datas ,BBox &bbox, float aX, float aY){
//     LOGI << "-> AnyModelWithTvaiImage::postprocess";
//     RET_CODE ret = RET_CODE::SUCCESS;
//     LOGI << "<- AnyModelWithTvaiImage::postprocess";
//     return ret;
// }