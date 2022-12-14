#include "imp_licplate_detection.hpp"

using namespace ucloud;
using namespace std;


ucloud::RET_CODE IMP_LICPLATE_DETECTOR::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
    LOGI<<"->IMP_LICPLATE_DETECTOR init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    if ( weightConfig.find(ucloud::InitParam::BASE_MODEL)==weightConfig.end() || \
            weightConfig.find(ucloud::InitParam::SUB_MODEL)==weightConfig.end() ){
        printf("**[%s][%d]IMP_LICPLATE_DETECTOR fail to search detector and classify model\n", __FILE__, __LINE__);
        ret = ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
        return ret;
    }

    std::map<ucloud::InitParam,ucloud::WeightData> dnetConfig = {
        {ucloud::InitParam::BASE_MODEL, weightConfig[ucloud::InitParam::BASE_MODEL] },
    };

    ret = m_detectHandle->init(dnetConfig);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::IMP_LICPLATE_DETECTOR m_detectHandle init return [%d]\n", ret);
        return ret;
    }

    std::map<ucloud::InitParam,ucloud::WeightData> cnetConfig = {
        {ucloud::InitParam::BASE_MODEL, weightConfig[ucloud::InitParam::SUB_MODEL] },
    };

    ret = m_clsHandle->init(cnetConfig);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** IMP_LICPLATE_DETECTOR classfication init failed\n");
        return ret;
    }

    return ret;
}


ucloud::RET_CODE IMP_LICPLATE_DETECTOR::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->IMP_LICPLATE_DETECTOR init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    std::map<InitParam, WeightData> weightConfig;
    for(auto &&modelp: modelpath){
        int szBuf = 0;
        unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
        weightConfig[modelp.first] = WeightData{tmpBuf,szBuf};
    }
    ret = init(weightConfig);
    for(auto &&wC: weightConfig){
        free(wC.second.pData);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    return ret;
}


ucloud::RET_CODE IMP_LICPLATE_DETECTOR::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI << "-> IMP_LICPLATE_DETECTOR::run";
    ucloud::VecObjBBox detBboxes;
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ret = m_detectHandle->run(tvimage,detBboxes,threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_LICPLATE_DETECTOR detect person failed!\n", __FILE__, __LINE__);
        return ret;
    }
    LOGI << detBboxes.size() << " licplates detected by yolo";

    ret = m_clsHandle->run(tvimage,detBboxes);

    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_LICPLATE_DETECTOR classify images failed!\n", __FILE__, __LINE__);
        return ret;
    }
    bboxes = detBboxes;
    LOGI << "<- IMP_LICPLATE_DETECTOR::run";
    return ret;
}


ucloud::RET_CODE IMP_LICPLATE_DETECTOR::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return m_detectHandle->get_class_type(valid_clss);
}


