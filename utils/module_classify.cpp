#include "module_classify.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

Classification::Classification(){
    LOGI<<"->Classify";
    m_net = std::make_shared<BaseModel>();
    m_drm = std::make_shared<ImageUtil>();
}

Classification::~Classification(){
    LOGI<<"->~Classify";
}


ucloud::RET_CODE Classification::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->Classify init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    printf("-->step into classification init");
    if (modelpath.find(ucloud::InitParam::SUB_MODEL)== modelpath.end()){
        LOGI << "base model not found in modelpath";
        return ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    bool useDRM = false;
    ret = m_net->base_init(modelpath[ucloud::InitParam::SUB_MODEL],useDRM);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        return ret;
    }
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());

    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    m_param_img2tensor.keep_aspect_ratio = false;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;
    m_param_img2tensor.model_input_shape = m_InpSp;
    LOGI << "<- Classfication::init";
    return ret;
}



ucloud::RET_CODE Classification::preprocess_drm(ucloud::TvaiImage &tvimg,std::vector<unsigned char*> &input_datas){

  LOGI << "-> Classification::preprocess_drm";
    bool valid_input_format = true;
    std::vector<float> ax,ay;
    switch (tvimg.format)
    {
    case ucloud::TVAI_IMAGE_FORMAT_RGB:
    case ucloud::TVAI_IMAGE_FORMAT_BGR:
    case ucloud::TVAI_IMAGE_FORMAT_NV12:
    case ucloud::TVAI_IMAGE_FORMAT_NV21:
        break;
    default:
        valid_input_format = false;
        break;
    }
    if(!valid_input_format) return ucloud::RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;

    unsigned char* data = (unsigned char*)std::malloc(3*m_InpSp.w*m_InpSp.w);
    ucloud::RET_CODE uret = m_drm->init(tvimg);
    if(uret!=ucloud::RET_CODE::SUCCESS) return uret;
    // int ret = m_drm->resize(tvimage,m_InpSp, data);
    int ret = m_drm->resize(tvimg, m_param_img2tensor, data);
    input_datas.push_back(data);
    ax.push_back( (float(m_InpSp.w))/tvimg.width );
    ay.push_back( (float(m_InpSp.h))/tvimg.height );

    // cv::Mat cvimage_show( cv::Size(m_InpSp.w, m_InpSp.h), CV_8UC3, data);
    // cv::cvtColor(cvimage_show, cvimage_show, cv::COLOR_RGB2BGR);
    // cv::imwrite("preprocess_drm.jpg", cvimage_show);

    LOGI << "<- Classification::preprocess_drm";
    return ucloud::RET_CODE::SUCCESS;


}

ucloud::RET_CODE Classification::preprocess_opencv(ucloud::TvaiImage &tvimg,std::vector<unsigned char*> &input_datas,std::vector<cv::Rect> &rois){
    LOGI<<"->Classification preprocess opencv";
    bool use_subpixel = false;
    std::vector<cv::Mat> dst;
    std::vector<float> ax,ay;
    if (rois.empty()){
        rois = {cv::Rect(0,0,tvimg.width,tvimg.height)}; 
    }
    // std::vector<cv::Rect> roi ={cv::Rect(0,0,tvimg.width,tvimg.height)};
    // printf("tvimg shape  width: %d ,height: %d \n",tvimg.width,tvimg.height);
    
    ucloud::RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimg,rois,dst,m_param_img2tensor,
                                                                ax,ay);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        return ret;
    }
    for (auto &&ele:dst){
        unsigned char *data = (unsigned char*)std::malloc(ele.total()*3);
        std::memcpy(data,ele.data,ele.total()*3);
        input_datas.push_back(data);
    }
    printf("-->>classifcation::preprocess_opencv finish!\n");
    LOGI << "<- Classification::preprocess_opencv";
    return ret;
}



ucloud::RET_CODE Classification::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"->Classfication run";
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
    std::vector<unsigned char*> input_datas;
    std::vector<unsigned char*> single_data;
    std::vector<float*> output_datas;
    std::vector<cv::Rect> rois;
    std::vector<int> output_nums = m_net->get_output_elem_num();
    // printf("--->> classify get output_nums %d finished!\n",output_nums.size());
    int roi_nums = input_datas.size();
    int num_cls = m_clss.size();
    // printf("--->> classify num cls is : %d, select_idx is: %d",num_cls,m_select);
    
    printf("step into phone classify ! \n");
    if (!bboxes.empty()){
        printf("bboxes size is %d\n",bboxes.size());
        for (auto box:bboxes){
            rois.push_back(cv::Rect(box.rect.x,box.rect.y,box.rect.width,box.rect.height));
        }
    }

    // printf("--->> classify process image !\n");
    ret = preprocess_opencv(tvimage,input_datas,rois);
    // printf("-->>classify input num %d\n",input_datas.size());
    // printf("--->> classify proccess image finished!\n");
    if(ret!=ucloud::RET_CODE::SUCCESS) return ret;
    for (int i=0;i<input_datas.size();i++){
        single_data.push_back(input_datas[i]);
        ret  = m_net->general_infer_uint8_nhwc_to_float(single_data,output_datas);
        printf("--->> classify single infer finished!\n");
        if (ret!=ucloud::RET_CODE::SUCCESS) {
             break;
             printf("-->>classify %d target failed!\n",i);
            //return ret;
            }    
        // printf("--->> classify output_datas nums: %d !\n",output_datas.size()); 
        for (int j=0;j<output_nums.size();j++){
            // printf("--->> classify get result!\n");
            float* buf = output_datas[j];
            float m_score = buf[m_select];
            // printf("ROI %d,buf select score:%f\n",i,m_score);
            // printf("m_select: %d,threshold: %f,buf select score:%f\n",m_select,threshold,m_score);
            printf("--->> classify %d target score is %f class type is phone",i,m_score);
            if (m_score>threshold){
                    printf("-->>classify step into threshold select!");
                    // bboxes[j]. = m_score;
                    bboxes[i].objtype= m_clss[m_select];
                    
            }
        }
        single_data.clear();
        output_datas.clear();
    } 
    if(ret!=ucloud::RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        for(auto &&t: output_datas) free(t);
        for(auto &&t: single_data) free(t);
        return ret;
    }
    for(auto &&t: input_datas) free(t);
    for(auto &&t: output_datas) free(t);
    for(auto &&t: single_data) free(t);

    /// 错误使用了genral_infer-unit8_nhwc_to_float的接口，该接口是针对
    /// 单个模型多个输入的一个输出的情况；因此针对分类模型多个输入多个输出，
    /// 只能选择for循环
    /************
    ret = m_net->general_infer_uint8_nhwc_to_float(input_datas,output_datas);
    printf("--->> classify infer finished!\n");
    if (ret!=ucloud::RET_CODE::SUCCESS) return ret;
    std::vector<int> output_nums = m_net->get_output_elem_num();
    printf("--->> classify get output_nums %d finished!\n",output_nums.size());
    printf("--->> classify output_datas nums: %d !\n",output_datas.size()); 
    // ucloud::VecObjBBox sbbox;
    int roi_nums = input_datas.size();
    int num_cls = m_clss.size();
    printf("--->> classify num cls is : %d, select_idx is: %d",num_cls,m_select);
    for (int i=0;i<output_nums.size();i++){
        printf("--->> classify get result!\n");
        float* buf = output_datas[i];
        for (int j=0;j<roi_nums;j++){
            int s_idx = j*num_cls + m_select;
            // printf("i %d buf:0 is %f,1:%f,2:%f\n",i,buf[0],buf[1],buf[2]);
            float m_score = buf[s_idx];
            printf("ROI %d,buf select score:%f\n",j,m_score);
            // printf("m_select: %d,threshold: %f,buf select score:%f\n",m_select,threshold,m_score);
            if (m_score>threshold){
                printf("-->>classify step into threshold select!");
                // bboxes[j]. = m_score;
                bboxes[j].objtype= m_clss[m_select];
            }
        }
    }
    **********/
    return ret;
}
    


ucloud::RET_CODE Classification::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return ucloud::RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&m: m_clss){
        bool FLAG_exsit_class = false;
        for( auto &&n:valid_clss){
            if( m==n){
                FLAG_exsit_class = true;
                break;
            }
        }
        if(!FLAG_exsit_class) valid_clss.push_back(m);
    }
    return ucloud::RET_CODE::SUCCESS;
}


static inline int get_unique_cls_num(std::vector<ucloud::CLS_TYPE>& output_clss, std::map<ucloud::CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::vector<ucloud::CLS_TYPE> unique_cls;
    for(auto i=output_clss.begin(); i !=output_clss.end(); i++){
        bool conflict = false;
        for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
            if( *i == *iter ){
                conflict = true;
                break;
            }
        }
        if(!conflict) unique_cls.push_back(*i);
    }
    for(int i=0; i < unique_cls.size(); i++ ){
        unique_cls_order.insert(std::pair<ucloud::CLS_TYPE,int>(unique_cls[i],i));
    }
    return unique_cls.size();
}

ucloud::RET_CODE Classification::set_output_cls_order(std::vector<ucloud::CLS_TYPE>& output_clss,int select_idx){
    printf("--->> Classification set output cls oder output_clss size:%d, select_idx :%d",output_clss,select_idx);
    m_nc = output_clss.size();
    m_select = select_idx;
    // printf("m_nc is %d\n",m_nc);
    m_clss = output_clss;
    get_unique_cls_num(output_clss, m_unique_clss_map);
    return ucloud::RET_CODE::SUCCESS;
}




