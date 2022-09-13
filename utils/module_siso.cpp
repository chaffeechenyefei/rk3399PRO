#include "module_siso.hpp"
#include <opencv2/opencv.hpp>
using namespace ucloud;
using namespace std;
using namespace cv;

NaiveModel::NaiveModel(){
    LOGI << "-> NaiveModel";
    m_net = std::make_shared<BaseModel>();
}

NaiveModel::~NaiveModel(){
    LOGI << "-> NaiveModel";
}

RET_CODE NaiveModel::init(std::map<ucloud::InitParam, std::string> &modelpath){
    LOGI << "-> NaiveModel::init";
    RET_CODE ret = RET_CODE::SUCCESS;
    if( modelpath.find(InitParam::BASE_MODEL) == modelpath.end() ){
        LOGI << "base model not found in modelpath";
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    // m_net->release();
    bool useDRM = false;
    ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], useDRM);
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO的体现, 都只取index0的数据
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());
    m_InpSp = m_net->get_input_shape()[0];
    m_OutSp = m_net->get_output_shape()[0];
    m_OutEleNum = m_net->get_output_elem_num()[0];
    //初始化内存池, 供内存复用的使用
    int mem_pool_nodes = 1;
    size_t mem_pool_size = reinterpret_cast<size_t>(m_OutEleNum*sizeof(float));
    m_OtpMemPool.init( mem_pool_size ,mem_pool_nodes);
    //图像前处理参数
    m_param_img2tensor.keep_aspect_ratio = true;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;
    m_param_img2tensor.model_input_shape = m_InpSp;

    LOGI << "<- NaiveModel::init";
    return ret;
}

RET_CODE NaiveModel::run(TvaiImage& tvimage, VecObjBBox &bboxes){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> NaiveModel::run";
    RET_CODE ret = RET_CODE::SUCCESS;

    if(tvimage.format != TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    std::vector<unsigned char*> input_datas;
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
    ret = preprocess(tvimage, input_datas);
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
    ret = postprocess(output_datas);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif    
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

    LOGI << "<- NaiveModel::run";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE NaiveModel::run_mem(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes){
    LOGI << "-> NaiveModel::run_mem";
    RET_CODE ret = RET_CODE::SUCCESS;

    if(tvimage.format != TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    std::vector<unsigned char*> input_datas;
    std::vector<float*> output_datas;//不用释放
    std::vector<MemNode*> used_MemNodes;//需要释放,还给mem pool

    for(int i = 0; i < m_OtpNum; i++){
        MemNode* tmp = m_OtpMemPool.malloc();
        output_datas.push_back( (float*)(tmp->ptr) );
        used_MemNodes.push_back(tmp);
    }
#ifdef TIMING    
    m_Tk.start();
#endif
    ret = preprocess(tvimage, input_datas);
#ifdef TIMING    
    m_Tk.end("preprocess");
#endif    
    
#ifdef TIMING    
    m_Tk.start();
#endif    
    ret = m_net->general_infer_uint8_nhwc_to_float_mem(input_datas, output_datas);
#ifdef TIMING    
    m_Tk.end("general_infer_uint8_nhwc_to_float_mem");
#endif    
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif  
    ret = postprocess(output_datas);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif      
    if(ret!=RET_CODE::SUCCESS) return ret;

    for(auto &&t: used_MemNodes){
        m_OtpMemPool.free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }
    LOGI << "<- NaiveModel::run_mem";
    return RET_CODE::SUCCESS;
}

RET_CODE NaiveModel::run_drm(TvaiImage& tvimage, VecObjBBox &bboxes){
    // return run_mem(tvimage, bboxes);
    LOGI << "-> NaiveModel::run_drm";
    RET_CODE ret = RET_CODE::SUCCESS;

    if(tvimage.format != TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    // std::vector<unsigned char*> input_datas;
    std::vector<float*> output_datas;

    Mat im(tvimage.height,tvimage.width,CV_8UC3, tvimage.pData); 

#ifdef TIMING    
    m_Tk.start();
#endif
    ret = m_net->general_infer_uint8_nhwc_to_float(im, output_datas);
#ifdef TIMING    
    m_Tk.end("general_infer_uint8_nhwc_to_float");
#endif    
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif
    ret = postprocess(output_datas);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif    
    if(ret!=RET_CODE::SUCCESS) return ret;

    for(auto &&t: output_datas){
        free(t);
    }
    LOGI << "<- NaiveModel::run_drm";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE NaiveModel::preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas){
    LOGI << "-> NaiveModel::preprocess";
    // Mat im(tvimage.height,tvimage.width,CV_8UC3, tvimage.pData);
    // Mat resized_im;
    // cv::resize(im, resized_im, Size(m_InpSp.w,m_InpSp.h));
    // unsigned char *tmp = (unsigned char *)malloc(resized_im.cols*resized_im.rows*3);
    // memcpy(tmp, resized_im.data, resized_im.cols*resized_im.rows*3);
    // input_datas.push_back(tmp);
    std::vector<cv::Mat> dst;
    std::vector<float> aX,aY;
    std::vector<cv::Rect> roi = {cv::Rect(0,0,tvimage.width,tvimage.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, roi, 
        dst, m_param_img2tensor, aX, aY);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        // cv::imwrite("preprocess_img.png", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        input_datas.push_back(data);
    }
    LOGI << "<- NaiveModel::preprocess";
    return ret;
}


ucloud::RET_CODE NaiveModel::postprocess(std::vector<float*> &output_datas){
    LOGI << "-> NaiveModel::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;
    
    float* tmp = output_datas[0];
    printf("===OUTPUT===\n");
    printf("[ ");
    for( int i = 0 ; i < m_OutEleNum; i++ ){
        printf("%f, ", tmp[i]);
    }
    printf("]\n");
    printf("============\n");
    LOGI << "<- NaiveModel::postprocess";
    return RET_CODE::SUCCESS;
}

