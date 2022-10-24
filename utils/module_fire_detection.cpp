#include "module_fire_detection.hpp"

using namespace ucloud;
using namespace std;


ucloud::RET_CODE FireDetector::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->FireDetector init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    std::string dnetPath,cnetPath;
    if (modelpath.find(ucloud::InitParam::BASE_MODEL)==modelpath.end()){
        printf("**[%s][%d]FireDetector fail to search detector and classify model\n", __FILE__, __LINE__);
        ret = ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
        return ret;
    }
    dnetPath = modelpath[ucloud::InitParam::BASE_MODEL];
    cnetPath = modelpath[ucloud::InitParam::SUB_MODEL];


    vector<CLS_TYPE> yolov5s_target = {CLS_TYPE::OTHERS};
    m_detectHandle->set_output_cls_order(yolov5s_target);
    ret = m_detectHandle->init(dnetPath);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::FireDetector m_detectHandle init return [%d]\n", ret);
        return ret;
    }

    ret = m_clsHandle->init(cnetPath);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** FireDetector classfication init failed\n");
        return ret;
    }
    vector<CLS_TYPE> classify_target = {CLS_TYPE::OTHERS, CLS_TYPE::FIRE};
    m_clsHandle->set_output_cls_order(classify_target);
    // vector<CLS_TYPE> output_dim_cls_order = {CLS_TYPE::OTHERS, m_cls, CLS_TYPE::OTHERS};
    // m_clsHandle->set_output_cls_order(output_dim_cls_order);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** FireDetector failed\n");
        return ret;
    }
    
    return ret;
}


ucloud::RET_CODE FireDetector::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI << "-> FireDetector::run";
    ucloud::VecObjBBox detBboxes, candBboxes;
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ret = m_detectHandle->run(tvimage,detBboxes,threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] FireDetector detect person failed!\n", __FILE__, __LINE__);
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
        printf("**[%s][%d] FireDetector classify images failed!\n", __FILE__, __LINE__);
        return ret;
    }
    for(auto &&box: candBboxes){
        if(box.objtype == CLS_TYPE::FIRE)
            bboxes.push_back(box);
    }
    LOGI << "<- FireDetector::run";
    return ret;
}


ucloud::RET_CODE FireDetector::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return m_clsHandle->get_class_type(valid_clss);
}


