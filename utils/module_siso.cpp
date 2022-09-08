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
    ret = m_net->base_init(modelpath[InitParam::BASE_MODEL]);
    if(ret!=RET_CODE::SUCCESS) return ret;

    m_InpSp = m_net->get_input_shape()[0];
    m_OutSp = m_net->get_output_shape()[0];
    m_OutEleNum = m_net->get_output_elem_num()[0];

    LOGI << "<- NaiveModel::init";
    return ret;
}

RET_CODE NaiveModel::run(TvaiImage& tvimage, VecObjBBox &bboxes){
    LOGI << "-> NaiveModel::run";
    RET_CODE ret = RET_CODE::SUCCESS;

    if(tvimage.format != TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    std::vector<unsigned char*> input_datas;
    std::vector<float*> output_datas;

    ret = preprocess(tvimage, input_datas);
    
    ret = m_net->general_infer(input_datas, output_datas);
    if(ret!=RET_CODE::SUCCESS) return ret;

    ret = postprocess(output_datas);
    if(ret!=RET_CODE::SUCCESS) return ret;

    for(auto &&t: output_datas){
        free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }

    LOGI << "<- NaiveModel::run";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE NaiveModel::preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas){
    LOGI << "-> NaiveModel::preprocess";
    Mat im(tvimage.height,tvimage.width,CV_8UC3, tvimage.pData);
    Mat resized_im;
    cv::resize(im, resized_im, Size(m_InpSp.w,m_InpSp.h));
    unsigned char *tmp = (unsigned char *)malloc(resized_im.cols*resized_im.rows*3);
    memcpy(tmp, resized_im.data, resized_im.cols*resized_im.rows*3);
    input_datas.push_back(tmp);
    LOGI << "<- NaiveModel::preprocess";
    return RET_CODE::SUCCESS;
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