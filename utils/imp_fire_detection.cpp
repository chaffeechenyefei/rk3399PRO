#include "imp_fire_detection.hpp"

using namespace ucloud;
using namespace std;


ucloud::RET_CODE IMP_FIRE_DETECTOR::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
    LOGI<<"->IMP_FIRE_DETECTOR init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    WeightData dnetPath,cnetPath;
    if (weightConfig.find(ucloud::InitParam::BASE_MODEL)==weightConfig.end()){
        printf("**[%s][%d]IMP_FIRE_DETECTOR fail to search detector and classify model\n", __FILE__, __LINE__);
        ret = ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
        return ret;
    }
    dnetPath = weightConfig[ucloud::InitParam::BASE_MODEL];
    cnetPath = weightConfig[ucloud::InitParam::SUB_MODEL];

    vector<CLS_TYPE> yolov5s_target = {CLS_TYPE::OTHERS};
    m_detectHandle->set_output_cls_order(yolov5s_target);
    ret = m_detectHandle->init(dnetPath);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::IMP_FIRE_DETECTOR m_detectHandle init return [%d]\n", ret);
        return ret;
    }

    ret = m_clsHandle->init(cnetPath);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** IMP_FIRE_DETECTOR classfication init failed\n");
        return ret;
    }
    vector<CLS_TYPE> classify_target = {CLS_TYPE::OTHERS, CLS_TYPE::FIRE};
    m_clsHandle->set_output_cls_order(classify_target);
    // vector<CLS_TYPE> output_dim_cls_order = {CLS_TYPE::OTHERS, m_cls, CLS_TYPE::OTHERS};
    // m_clsHandle->set_output_cls_order(output_dim_cls_order);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** IMP_FIRE_DETECTOR failed\n");
        return ret;
    }
    
    return ret;
}


ucloud::RET_CODE IMP_FIRE_DETECTOR::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->IMP_FIRE_DETECTOR init";
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


ucloud::RET_CODE IMP_FIRE_DETECTOR::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI << "-> IMP_FIRE_DETECTOR::run";
    ucloud::VecObjBBox detBboxes, candBboxes;
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ret = m_detectHandle->run(tvimage,detBboxes,threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_FIRE_DETECTOR detect person failed!\n", __FILE__, __LINE__);
        return ret;
    }
    LOGI << detBboxes.size() << " fire detected by yolo";

    if(m_trust_threshold > threshold ){//高可信触发
        for(auto &&box: detBboxes){
            if(box.confidence > m_trust_threshold){
                bboxes.push_back(box);
            } else {
                candBboxes.push_back(box);
            }
        }
    } else {
        candBboxes = detBboxes;
    }
    LOGI << bboxes.size() << " fire trusted";
    LOGI << candBboxes.size() << " fire to be classified";

    if(candBboxes.empty()) return RET_CODE::SUCCESS;

    ret = m_clsHandle->run(tvimage,candBboxes,m_fire_classify_threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_FIRE_DETECTOR classify images failed!\n", __FILE__, __LINE__);
        return ret;
    }
    for(auto &&box: candBboxes){
        if(box.objtype == CLS_TYPE::FIRE)
            bboxes.push_back(box);
    }
    LOGI << "<- IMP_FIRE_DETECTOR::run";
    return ret;
}


ucloud::RET_CODE IMP_FIRE_DETECTOR::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return m_clsHandle->get_class_type(valid_clss);
}


