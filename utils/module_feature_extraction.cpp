#include "module_feature_extraction.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>

using namespace std;
using namespace ucloud;

FeatureExtractor::FeatureExtractor(){
    LOGI<<"-> FeatureExtractor";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
}

FeatureExtractor::~FeatureExtractor(){
    LOGI<<"-> ~FeatureExtractor";
}

ucloud::RET_CODE FeatureExtractor::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
    LOGI<<"-> FeatureExtractor::init";
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
    m_param_img2tensor.keep_aspect_ratio = true;
    m_param_img2tensor.pad_both_side = true;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
    m_param_img2tensor.model_input_shape = m_InpSp;
    LOGI << "<- FeatureExtractor::init";
    return ret;
}


ucloud::RET_CODE FeatureExtractor::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"-> FeatureExtractor::init";
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
    LOGI << "<- FeatureExtractor::init";
    return ret;
}

ucloud::RET_CODE FeatureExtractor::postprocess(std::vector<float*> &output_datas ,BBox &bbox){
    LOGI << "-> FeatureExtractor::postprocess";
    RET_CODE ret = RET_CODE::SUCCESS;
    if(output_datas.empty()) return RET_CODE::FAILED;
    float *ptr = output_datas[0];
    int dims = m_net->get_output_elem_num()[0];
    unit_norm(ptr,dims);
    bbox.feat.featureLen = (unsigned int)(dims*sizeof(float));
    bbox.feat.pFeature = reinterpret_cast<uint8_t*>(ptr);
    LOGI << "<- FeatureExtractor::postprocess";
    return ret;
}

void FeatureExtractor::unit_norm(float* ptr, int dims){
    float sum = 0;
    for(int i = 0; i < dims; i++){
        sum += ptr[i]*ptr[i];
    }
    sum = std::sqrt(sum + 1e-3);
    for(int i = 0; i < dims; i++){
        ptr[i] /= sum;
    }
    return;
}

/*******************************************************************************
 * run 对bboxes中的每个区域进行分类, 并将结果更新到bboxes中(objtype, objectness, confidence)
*******************************************************************************/
ucloud::RET_CODE FeatureExtractor::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"-> FeatureExtractor::run";
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

        ret = postprocess(output_datas, box);

        if(ret!=RET_CODE::SUCCESS) {
            for(auto &&t: input_datas) free(t);
            // for(auto &&t: output_datas) free(t);
            return ret;
        }

        // for(auto &&t: output_datas){
        //     free(t);
        // }
        for(auto &&t: input_datas){
            free(t);
        }
    }
    return ret;
}
