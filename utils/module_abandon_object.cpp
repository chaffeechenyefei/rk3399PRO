#include "module_abandon_object.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace ucloud;
using namespace cv;

static float sigmoid(float input){
    return 1.0/(1.0+expf(-input));
}

static float unsigmoid(float input){
    return -1.0*logf((1.0/input) - 1.0);
}

RET_CODE ABANDON_OBJECT_DETECTION::updateBG(const Mat cur){
    // Mat cur(tvimage.height,tvimage.width,CV_8UC3,tvimage.pData);
    Mat first = m_batchBg.front();
    m_bg -=  first*m_Bg_weight[0];
    m_batchBg.pop();
    m_bg += cur*m_Bg_weight[m_Bg_weight.size()-1];
    m_batchBg.push(cur);
    return RET_CODE::SUCCESS;
}

// RET_CODE Segment::calc_diff(const Mat cur,Mat &diff){
//     diff = cur - m_bg;
//     return RET_CODE::SUCCESS;
// }

RET_CODE ABANDON_OBJECT_DETECTION::create_trackor(int uuid_cam){
    if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
        LOGI << "-> Segment::create_trackor non-trival";
        std::shared_ptr<BoxTraceSet> m_trackor_t(new BoxTraceSet());
        m_Trackors.insert(std::pair<int,std::shared_ptr<BoxTraceSet>>(uuid_cam,m_trackor_t));
    } else LOGI << "-> Segment::create_trackor trival";
    return RET_CODE::SUCCESS;
}


RET_CODE ABANDON_OBJECT_DETECTION::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> AnyModelWithTvaiImage::run";
    RET_CODE ret = RET_CODE::SUCCESS;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    cout<<"theshold is "<< threshold<<" nms_threshold"<< nms_threshold<<endl;

    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        ret = RET_CODE::SUCCESS;
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    std::vector<unsigned char*> input_datas;
    std::vector<float> aX, aY;
    std::vector<float*> output_datas;
    // 考虑到图像与模板是否一致的问题，将tracker在run函数中init
    create_trackor(tvimage.uuid_cam);
    //tvimage 中传入的是新来图像与木板的asbdiff,这样做的原因
    // 1 如果使用batchInImage,需要在内部重新取址计算mat,然后计算mat，
    // 2 如果使用tvimage，结合成员变量m_bg, 那么需要计算得到absdiff，传入preprocess的时候需要修改tvimage的pData
    // 3 如果采用convProxy的方式
    Mat matDiff(tvimage.height,tvimage.width,CV_8UC3,tvimage.pData);
    float diff_sum = (float)cv::sum(matDiff)[0];
    bool same = diff_sum==0;//判断图像与模板是否有差异，如果没有差异，并且对应的trackor中存在轨迹
    std::vector<BoxTrace> his = m_Trackors[tvimage.uuid_cam]->mboxTrs;
    //当图像与模板不同时，进行分割推理
    //与pre frame一致并且 his中存在数据，

    if (!same){
        #ifdef TIMING    
            m_Tk.start();
        #endif
        #ifdef USEDRM
            ret = m_cv_preprocess_net->preprocess_drm(tvimage, m_param_img2tensor, input_datas, aX, aY);
        #else
            ret = m_cv_preprocess_net->preprocess_opencv(tvimage, m_param_img2tensor, input_datas, aX, aY);
        #endif
        #ifdef TIMING    
            m_Tk.end("preprocess");
        #endif
            if(ret!=RET_CODE::SUCCESS) return ret;


        #ifdef TIMING    
            m_Tk.start();
        #endif
            ret = m_net->general_infer_uint8_nhwc_to_float(input_datas, output_datas);
        #ifdef TIMING    
            m_Tk.end("general_infer_uint8_nhwc_to_float");
        #endif    
            if(ret!=RET_CODE::SUCCESS) {
                for(auto &&t: input_datas) free(t);
                return ret;
            }

        #ifdef TIMING    
            m_Tk.start();
        #endif
            ret = postprocess(output_datas, TvaiRect{0,0,tvimage.width, tvimage.height} , threshold, nms_threshold, bboxes, aX[0], aY[0]);
        #ifdef TIMING    
            m_Tk.end("postprocess");
        #endif
    }

    // 不管图像与模板是否存在差异，都要更新tracker，如果same==True，使用之前track中的最后一个trackPoint坐标进行更新
    // 如果 same==False，则使用bboxes进行更新
    ret = trackprocess(tvimage,bboxes,same);

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

    LOGI << "<- AnyModelWithTvaiImage::run";
    return RET_CODE::SUCCESS;
}


RET_CODE ABANDON_OBJECT_DETECTION::run(TvaiImage& tvimage, TvaiRect roi , VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> AnyModelWithTvaiImage::run with roi";
    RET_CODE ret = RET_CODE::SUCCESS;
    threshold = clip_threshold(threshold);

    nms_threshold = clip_threshold(nms_threshold);
    roi = get_valid_rect(roi, tvimage.width, tvimage.height);

    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        ret = RET_CODE::SUCCESS;
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    std::vector<unsigned char*> input_datas;
    std::vector<float> aX, aY;
    std::vector<float*> output_datas;
    create_trackor(tvimage.uuid_cam);
    Mat matDiff(tvimage.height,tvimage.width,CV_8UC3,tvimage.pData);
    float diff_sum = (float)cv::sum(matDiff)[0];
    bool same = diff_sum==0;
    std::vector<BoxTrace> his = m_Trackors[tvimage.uuid_cam]->mboxTrs;
    if (!same){

        #ifdef TIMING    
            m_Tk.start();
        #endif
        #ifdef USEDRM
            ret = m_cv_preprocess_net->preprocess_drm(tvimage, roi, m_param_img2tensor ,input_datas, aX, aY);
        #else
            ret = m_cv_preprocess_net->preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
        #endif
        #ifdef TIMING    
            m_Tk.end("preprocess");
        #endif
            if(ret!=RET_CODE::SUCCESS) return ret;

        #ifdef TIMING    
            m_Tk.start();
        #endif
            ret = m_net->general_infer_uint8_nhwc_to_float(input_datas, output_datas);
        #ifdef TIMING    
            m_Tk.end("general_infer_uint8_nhwc_to_float");
        #endif    
            if(ret!=RET_CODE::SUCCESS) {
                for(auto &&t: input_datas) free(t);
                return ret;
            }

        #ifdef TIMING    
            m_Tk.start();
        #endif
            ret = postprocess(output_datas, roi, threshold, nms_threshold, bboxes, aX[0], aY[0]);
        #ifdef TIMING    
            m_Tk.end("postprocess");
        #endif  
    }  
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        for(auto &&t: output_datas) free(t);
        return ret;
    }
    ret = trackprocess(tvimage,bboxes,same);

    for(auto &&t: output_datas){
        free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }

    LOGI << "<- AnyModelWithTvaiImage::run with roi";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE ABANDON_OBJECT_DETECTION::postprocess(std::vector<float*> &output_datas, ucloud::TvaiRect roi, float threshold, float nms_threshold ,ucloud::VecObjBBox &bboxes,float aX, float aY){
    int output_w = m_OutEleDims[0][0];
    int output_h = m_OutEleDims[0][1];
    // for (int i=0;i<output_h;i++){
    //     for (int j=0;j<output_w;j++){
    //         float value = *(output_datas[0]+i*output_w+j);
    //         if (value>=0.2){
    //             cout<<value<<" ";
    //         }
    //     }
    //     cout<<endl;
    // }
    Mat result(output_h,output_w,CV_32FC1,output_datas[0]);
    // Mat result(output_h,output_w,CV_8UC1,output_datas[0]);
    // imwrite("./data/abandon01_result/result.jpg",result);
    cout<< "Mat result value"<<format(result,Formatter::FMT_NUMPY)<<endl;
    // float u_threshold = unsigmoid(threshold);
    // cout<<"u_thresh "<< u_threshold<<endl;
    Mat mask = result>threshold;
    cout<< "Mat result value"<<format(mask,Formatter::FMT_NUMPY)<<endl;
    Mat dilate_mask;
    std::vector<std::vector<Point> > vec_cv_contours;
    // Mat kernel = getStructuringElement(0,Size(5,5),Point(1,1));
    dilate(mask,mask,Mat::ones(5,5,CV_8UC1),Point(1,1)); 
    findContours(dilate_mask,vec_cv_contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        if(rect.width < 2 || rect.height < 2) continue;//in 224x224 scale
        BBox bbox;
        bbox.objtype = CLS_TYPE::TARGET;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aX); bbox.rect.width = ((1.0*rect.width) / aX);
        bbox.rect.y = ((1.0*rect.y) / aY); bbox.rect.height = ((1.0*rect.height) / aY);
        bboxes.push_back(bbox);
    }
    LOGI << "<- Segment::postprocess";
    return RET_CODE::SUCCESS;
}


RET_CODE ABANDON_OBJECT_DETECTION::trackprocess(TvaiImage &tvimage, VecObjBBox &ins,bool same){
    int min_box_num = 4;//认定为轨迹至少需要几个box
  
    std::vector<BoxPoint> bpts;
    int cur_time = 1 + m_Trackors[tvimage.uuid_cam]->m_time;
    /// same 表示当前帧跟背景的差异，如果基本没有差异，将之前的track中的持续稳定追踪上的box track进行更新   
    /// 而m_trackor 中的 boxpoint 则不进行更新
    /// 涉及到trakcer中取m_track
    if (same){
        std::vector<BoxTrace> his = m_Trackors[tvimage.uuid_cam]->mboxTrs;
        std::vector<BoxPoint> hbox = m_Trackors[tvimage.uuid_cam]->mboxPts;
        BoxPoint cc_box;
        for (auto item:his){
            cc_box = item.get_last_trace();
            if (cc_box.empty()){
                continue;
            }else{
                BBox in;
                in.rect.x = cc_box.x;
                in.rect.y = cc_box.y;
                in.rect.width = cc_box.w;
                in.rect.height = cc_box.h;
                BoxPoint tmp = BoxPoint(in,cur_time);
                bpts.push_back(tmp);
            }
        }  
        for (auto &&item:hbox){
            item.updateT(cur_time);
            bpts.push_back(item);
        }
    }
    else{
        for(auto in: ins){
            BoxPoint tmp = BoxPoint(in, cur_time);
            bpts.push_back( tmp );
        }
    }
    m_Trackors[tvimage.uuid_cam]->push_back( bpts );
    std::vector<BoxPoint> marked_Spts,marked_Mpts, unmarked_pts;
    // m_Trackors[tvimage.uuid_cam]->output_last_point_of_trace(marked_pts, unmarked_pts, min_box_num);
    m_Trackors[tvimage.uuid_cam]->output_trace(marked_Spts, marked_Mpts, unmarked_pts, min_box_num);
    ins.clear();
    for(auto bxpt: marked_Spts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::ABADNON_STATIC;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        pt.track_id = bxpt.m_trace_id;
        ins.push_back(pt);
    }
    for(auto bxpt: marked_Mpts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::ABADNON_MOVE;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        pt.track_id = bxpt.m_trace_id;
        ins.push_back(pt);
    }

    for(auto bxpt: unmarked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::OTHERS;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        ins.push_back(pt);
    }
    
    return RET_CODE::SUCCESS;
}

RET_CODE ABANDON_OBJECT_DETECTION::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    valid_clss = m_clss;
    return RET_CODE::SUCCESS;
}

