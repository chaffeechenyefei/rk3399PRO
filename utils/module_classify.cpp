#include "module_classify.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace ucloud;

Classification::Classification(){
    LOGI<<"-> Classification";
    m_net = std::make_shared<BaseModel>();
    m_drm = std::make_shared<ImageUtil>();
}

Classification::~Classification(){
    LOGI<<"-> ~Classification";
}


ucloud::RET_CODE Classification::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"-> Classification::init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;

    if (modelpath.find(ucloud::InitParam::BASE_MODEL)== modelpath.end()){
        printf("**[%s][%d] base model not found in modelpath\n", __FILE__, __LINE__);
        return ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    bool useDRM = false;
    ret = m_net->base_init(modelpath[ucloud::InitParam::BASE_MODEL],useDRM);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        return ret;
    }
    if(m_InpNum != m_net->get_input_shape().size()){
        printf("** dims err m_InpuNum[%d] != m_net->get_input_shape().size()[%d]\n", m_InpNum, m_net->get_input_shape().size());
        return RET_CODE::FAILED;
    }
    if(m_OutNum != m_net->get_output_shape().size()){
        printf("** dims err m_OutNum[%d] != m_net->get_output_shape().size()[%d]\n", m_OutNum, m_net->get_output_shape().size());
        return RET_CODE::FAILED;
    }

    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    m_param_img2tensor.keep_aspect_ratio = false;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
    m_param_img2tensor.model_input_shape = m_InpSp;
    LOGI << "<- Classification::init";
    return ret;
}

/***whole image preprocess with opencv**/
ucloud::RET_CODE Classification::preprocess_opencv(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY ){
    LOGI << "-> Classification::preprocess_opencv";
    bool use_subpixel = false;
    std::vector<cv::Mat> dst;
    std::vector<cv::Rect> roi = {cv::Rect(0,0,tvimage.width,tvimage.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, roi, 
        dst, m_param_img2tensor, aX, aY, use_subpixel);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        // cv::imwrite("preprocess_img.png", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        input_datas.push_back(data);
    }
    LOGI << "<- Classification::preprocess_opencv";
    return ret;
}

/***image with preprocess with roi+drm**/
ucloud::RET_CODE Classification::preprocess_drm(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi, std::vector<unsigned char*> &input_datas, 
    std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> Classification::preprocess_drm with roi";
    bool valid_input_format = true;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        break;
    default:
        valid_input_format = false;
        break;
    }
    if(!valid_input_format) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;

    unsigned char* data = (unsigned char*)std::malloc(3*m_InpSp.w*m_InpSp.w);
    RET_CODE uret = m_drm->init(tvimage);
    if(uret!=RET_CODE::SUCCESS) return uret;
    // int ret = m_drm->resize(tvimage,m_InpSp, data);
    int ret = m_drm->resize(tvimage, roi, m_param_img2tensor, data);
    input_datas.push_back(data);
    aX.push_back( (float(m_InpSp.w))/roi.width );
    aY.push_back( (float(m_InpSp.h))/roi.height );

#ifdef VISUAL
    cv::Mat cvimage_show( cv::Size(m_InpSp.w, m_InpSp.h), CV_8UC3, data);
    cv::cvtColor(cvimage_show, cvimage_show, cv::COLOR_RGB2BGR);
    cv::imwrite("preprocess_drm.jpg", cvimage_show);
#endif

    LOGI << "<- Classification::preprocess_drm with roi";
    return RET_CODE::SUCCESS;
}

/***whole image preprocess with drm**/
ucloud::RET_CODE Classification::preprocess_drm(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, 
    std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> Classification::preprocess_drm";
    bool valid_input_format = true;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        break;
    default:
        valid_input_format = false;
        break;
    }
    if(!valid_input_format) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;

    unsigned char* data = (unsigned char*)std::malloc(3*m_InpSp.w*m_InpSp.w);
    RET_CODE uret = m_drm->init(tvimage);
    if(uret!=RET_CODE::SUCCESS) return uret;
    // int ret = m_drm->resize(tvimage,m_InpSp, data);
    int ret = m_drm->resize(tvimage, m_param_img2tensor, data);
    input_datas.push_back(data);
    aX.push_back( (float(m_InpSp.w))/tvimage.width );
    aY.push_back( (float(m_InpSp.h))/tvimage.height );

#ifdef VISUAL
    cv::Mat cvimage_show( cv::Size(m_InpSp.w, m_InpSp.h), CV_8UC3, data);
    cv::cvtColor(cvimage_show, cvimage_show, cv::COLOR_RGB2BGR);
    cv::imwrite("preprocess_drm.jpg", cvimage_show);
#endif

    LOGI << "<- Classification::preprocess_drm";
    return RET_CODE::SUCCESS;
}

/***image preprocess with roi+opencv**/
ucloud::RET_CODE Classification::preprocess_opencv(ucloud::TvaiImage& tvimage, TvaiRect roi, std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY  ){
    LOGI << "-> Classification::preprocess_opencv with roi";
    bool use_subpixel = false;
    std::vector<cv::Mat> dst;
    std::vector<cv::Rect> _roi_ = {cv::Rect(roi.x,roi.y,roi.width,roi.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, _roi_, 
        dst, m_param_img2tensor, aX, aY, use_subpixel);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        // cv::imwrite("preprocess_img.png", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        input_datas.push_back(data);
    }
    LOGI << "<- Classification::preprocess_opencv with roi";
    return ret;
}

ucloud::RET_CODE Classification::postprocess(std::vector<float*> &output_datas, float threshold,BBox &bbox){
    LOGI << "-> Classification::postprocess";
    RET_CODE ret = RET_CODE::SUCCESS;
    if(output_datas.empty()) return RET_CODE::FAILED;
    if(m_net->get_output_elem_num()[0]!=m_clss.size()){
        printf("**[%s][%d] m_clss[%d]!=output_dim0[%d]\n",__FILE__, __LINE__, m_clss.size(), m_net->get_output_elem_num()[0] );
        return RET_CODE::FAILED;
    }
    float *ptr = output_datas[0];
    LOGI << "output_datas[0][0-2]" << ptr[0] << ", " << ptr[1] << ", " << ptr[2];
    float max_score = -1;
    CLS_TYPE max_score_type = CLS_TYPE::OTHERS;
    for(int i = 0; i < m_clss.size(); i++ ){
        if(m_clss[i]==CLS_TYPE::OTHERS) continue;
        if(ptr[i] > max_score){
            max_score = ptr[i];
            max_score_type = m_clss[i];
        } 
    }
    LOGI << "max_score: " << max_score << ", cls_type: " << max_score_type;
    if(max_score > threshold){
        bbox.objtype = max_score_type;
        bbox.confidence = max_score;
    }
    return ret;
    LOGI << "<- Classification::postprocess";
}


ucloud::RET_CODE Classification::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"-> Classification::run";
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
    
    std::vector<int> output_nums = m_net->get_output_elem_num();
    int num_cls = m_clss.size();
    if(output_nums[0]!=num_cls){
        printf("**[%s][%d] output_nums[0][%d] != num_cls[%d]\n",__FILE__,__LINE__,output_nums[0], num_cls);
        return RET_CODE::FAILED;
    }
    
    if(bboxes.empty()) return RET_CODE::SUCCESS;//没有目标直接返回

    for(auto &&box: bboxes){
        std::vector<unsigned char*> input_datas;
        std::vector<float*> output_datas;
        TvaiRect roi = box.rect;
        vector<float> aX,aY;
        
    #ifdef USEDRM  
        roi = get_valid_rect(roi, tvimage.width, tvimage.height);
        ret = preprocess_drm(tvimage, roi, input_datas, aX, aY);
    #else
        ret = preprocess_opencv(tvimage, roi, input_datas, aX, aY);
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

        ret = postprocess(output_datas, threshold, box);

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
    

ucloud::RET_CODE Classification::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return ucloud::RET_CODE::ERR_MODEL_NOT_INIT;
    std::set<CLS_TYPE> unique_vec;
    unique_vec.insert(m_clss.begin(), m_clss.end());
    // unique_vec = unique_vec - std::set<CLS_TYPE>{CLS_TYPE::OTHERS};
    valid_clss.insert(valid_clss.end(), unique_vec.begin(), unique_vec.end());
    return ucloud::RET_CODE::SUCCESS;
}

ucloud::RET_CODE Classification::set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss){
    m_clss = output_clss;
    return RET_CODE::SUCCESS;
}




