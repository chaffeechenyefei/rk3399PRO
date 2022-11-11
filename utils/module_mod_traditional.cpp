#include "module_mod_traditional.hpp"
#include <opencv2/video/background_segm.hpp>

using namespace ucloud;
using namespace cv;

RET_CODE BACKGROUND_SEGMENTATION::init(){
    int h = 416;
    int w = 736;
    m_param_img2tensor.keep_aspect_ratio = false;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
    m_param_img2tensor.model_input_shape = DATA_SHAPE{1,3,h,w};
    return RET_CODE::SUCCESS;
}

RET_CODE BACKGROUND_SEGMENTATION::init(std::map<ucloud::InitParam,std::string> &modelpath){
    return BACKGROUND_SEGMENTATION::init();
}

RET_CODE BACKGROUND_SEGMENTATION::init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig){
    return BACKGROUND_SEGMENTATION::init();
}

RET_CODE BACKGROUND_SEGMENTATION::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    valid_clss = m_clss;
    return RET_CODE::SUCCESS;
}

RET_CODE BACKGROUND_SEGMENTATION::create_model(int uuid_cam){
    if(m_Models.find(uuid_cam)==m_Models.end()){
        LOGI << "-> BACKGROUND_SEGMENTATION::create_model non-trival";
#ifdef OPENCV3 //opencv3.4.6
        //一定要用cv::Ptr<BackgroundSubtractor>, 不能用BackgroundSubtractor*, 否则会自动释放出现coredump
        cv::Ptr<BackgroundSubtractor> ptM = cv::createBackgroundSubtractorMOG2(50,32,true);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorGMG(50,0.95);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorGSOC(cv::bgsegm::LSBPCameraMotionCompensation::LSBP_CAMERA_MOTION_COMPENSATION_LK);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorLSBP(0,50,16);
        m_Models.insert(std::pair<int,cv::Ptr<BackgroundSubtractor>>(uuid_cam,ptM));
#else
        BackgroundSubtractorMOG2 *ptM = new BackgroundSubtractorMOG2(100,16,true);
        std::share_ptr<BackgroundSubtractor> model_t(ptM);
        m_Models.insert(std::pair<int,std::shared_ptr<BackgroundSubtractor>>(uuid_cam,model_t));
#endif
    } else LOGI << "-> BACKGROUND_SEGMENTATION::create_model trival";
    return RET_CODE::SUCCESS;
}

RET_CODE BACKGROUND_SEGMENTATION::create_trackor(int uuid_cam){
    if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
        LOGI << "-> BACKGROUND_SEGMENTATION::create_trackor non-trival";
        std::shared_ptr<BoxTraceSet> m_trackor_t(new BoxTraceSet());
        m_Trackors.insert(std::pair<int,std::shared_ptr<BoxTraceSet>>(uuid_cam,m_trackor_t));
    } else LOGI << "-> BACKGROUND_SEGMENTATION::create_trackor trival";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE BACKGROUND_SEGMENTATION::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"-> BACKGROUND_SEGMENTATION::run";
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

    std::vector<float> aX,aY;
    std::vector<unsigned char*> input_datas;
        
#ifdef USEDRM  
    ret = m_cv_preprocess_net->preprocess_drm(tvimage, m_param_img2tensor, input_datas, aX, aY);
#else
    ret = preprocess_opencv(tvimage, m_param_img2tensor, input_datas, aX, aY);
#endif
    if(ret!=RET_CODE::SUCCESS){
        printf("[%s][%d] m_cv_preprocess_net error %d\n", __FILE__, __LINE__, ret);
        for(auto &&t: input_datas){
            if(t!=nullptr) free(t);
            t = nullptr;
        }
    }
    cv::Mat cropped_img(
        m_param_img2tensor.model_input_shape.h, 
        m_param_img2tensor.model_input_shape.w, 
        CV_8UC3, 
        input_datas[0]);
    VecObjBBox bboxes_before_filter;
    postprocess(cropped_img, bboxes_before_filter, aX[0], aY[0], threshold, tvimage.uuid_cam);
    postfilter(tvimage, bboxes_before_filter, bboxes, threshold);
    trackprocess(tvimage, bboxes);

    for(auto &&t: input_datas){
        if(t!=nullptr) free(t);
        t = nullptr;
    }

    return ret;
}


RET_CODE BACKGROUND_SEGMENTATION::postprocess(cv::Mat cvInp, VecObjBBox &bboxes, 
    float aX, float aY, float threshold, int uuid_cam)
{
    LOGI << "-> BACKGROUND_SEGMENTATION::postprocess";
    create_model(uuid_cam);
    Mat fgmask;//0:bg 255:fg
    float lr = 0.3;
    Mat gray_img, bal_img;
    cvtColor(cvInp, gray_img, CV_RGB2GRAY);
    blur(gray_img, gray_img, Size(3,3));
    gray_img.convertTo(bal_img, CV_32FC1);
    cv::pow(bal_img/255,0.7,bal_img);
    bal_img = bal_img*255;
    bal_img.convertTo(gray_img,CV_8UC1);
    // imwrite("tmp.jpg", gray_img);
    // equalizeHist(gray_img, gray_img);
#ifdef OPENCV3
    // LOGI << "-> BackgroundSegment::apply";
    m_Models[uuid_cam]->apply(gray_img, fgmask, lr);
    // LOGI << "<- BackgroundSegment::apply";
#else
    (*(m_Models[uuid_cam]))(gray_img, fgmask, lr);
#endif
    erode(fgmask, fgmask, Mat::ones(3,3,CV_8UC1));
    dilate(fgmask, fgmask, Mat::ones(5,5,CV_8UC1));
    
    std::vector<std::vector<Point>> vec_cv_contours;
    findContours(fgmask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // printf("vec_cv_contours %d\n", vec_cv_contours.size());
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aX); bbox.rect.width = ((1.0*rect.width) / aX);
        bbox.rect.y = ((1.0*rect.y) / aY); bbox.rect.height = ((1.0*rect.height) / aY);
        bboxes.push_back(bbox);
    }
    LOGI << "<- BACKGROUND_SEGMENTATION::postprocess";
    return RET_CODE::SUCCESS;
}


RET_CODE BACKGROUND_SEGMENTATION::postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold){
    //过滤掉全屏的误报
    LOGI << "bboxes before filter = " << ins.size();
    for(auto iter=ins.begin(); iter!=ins.end(); iter++){
        float box_size = (float(iter->rect.width*iter->rect.height)) / (tvimage.width*tvimage.height);
        if(box_size > 0.5) continue;
        if(iter->confidence > threshold ){
            outs.push_back(*iter);
        }
    }
    LOGI << "bboxes after filter = " << outs.size();
    return RET_CODE::SUCCESS;
}


RET_CODE BACKGROUND_SEGMENTATION::trackprocess(TvaiImage &tvimage, VecObjBBox &ins){
    int min_box_num = 4;//认定为轨迹至少需要几个box
    create_trackor(tvimage.uuid_cam);
    std::vector<BoxPoint> bpts;
    int cur_time = 1 + m_Trackors[tvimage.uuid_cam]->m_time;
    for(auto in: ins){
        BoxPoint tmp = BoxPoint(in, cur_time);
        bpts.push_back( tmp );
    }
    m_Trackors[tvimage.uuid_cam]->push_back( bpts );
    std::vector<BoxPoint> marked_pts, unmarked_pts;
    // m_Trackors[tvimage.uuid_cam]->output_last_point_of_trace(marked_pts, unmarked_pts, min_box_num);
    m_Trackors[tvimage.uuid_cam]->output_trace(marked_pts, unmarked_pts, min_box_num);
    ins.clear();
    for(auto bxpt: marked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        pt.track_id = bxpt.m_trace_id;
        ins.push_back(pt);
    }
    for(auto bxpt: unmarked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        ins.push_back(pt);
    }
    return RET_CODE::SUCCESS;
}