#include "imp_phone.hpp"

using namespace ucloud;
using namespace std;


void IMP_PHONE_DETECTOR::transform_box_to_ped_box(VecObjBBox &in_boxes){
    for( auto &&in_box: in_boxes){
        TvaiRect body_rect = in_box.rect;
        float hw_ratio = ((float)(1.0*body_rect.height))/ body_rect.width;
        if(hw_ratio >= 2){
            body_rect.height *= 0.5;
        }else if(hw_ratio >= 1.5){
            body_rect.height *= 0.8;
        }
        in_box.rect = body_rect;
    }
    return;
}


ucloud::RET_CODE IMP_PHONE_DETECTOR::init(std::map<InitParam, ucloud::WeightData> &weightConfig){
    LOGI<<"->IMP_PHONE_DETECTOR init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    WeightData dnetPath,cnetPath;
    if (weightConfig.find(ucloud::InitParam::BASE_MODEL)==weightConfig.end()||
    weightConfig.find(ucloud::InitParam::SUB_MODEL)==weightConfig.end()){
        LOGI<<"IMP_PHONE_DETECTOR fail to search detector and classify model";
        ret = ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
        return ret;
    }
    dnetPath = weightConfig[ucloud::InitParam::BASE_MODEL];
    cnetPath = weightConfig[ucloud::InitParam::SUB_MODEL];

    ret = m_ped_detectHandle->init(dnetPath);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::PhoningDetection m_ped_detectHandle init return [%d]\n", ret);
        return ret;
    }

    ret = m_clsHandle->init(cnetPath);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** IMP_PHONE_DETECTOR phone classfication init failed\n");
        return ret;
    }
    // vector<CLS_TYPE> output_dim_cls_order = {CLS_TYPE::OTHERS, m_cls, CLS_TYPE::OTHERS};
    // m_clsHandle->set_output_cls_order(output_dim_cls_order);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        printf("** IMP_PHONE_DETECTOR person detector failed\n");
        return ret;
    }
    
    return ret;
}

ucloud::RET_CODE IMP_PHONE_DETECTOR::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->IMP_PHONE_DETECTOR::init";
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
    LOGI << "<- IMP_PHONE_DETECTOR::init";
    return ret;
}


ucloud::RET_CODE IMP_PHONE_DETECTOR::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI << "-> IMP_PHONE_DETECTOR::run";
    ucloud::VecObjBBox detBboxes;
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ret = m_ped_detectHandle->run(tvimage,detBboxes,m_ped_threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_PHONE_DETECTOR detect person failed!\n", __FILE__, __LINE__);
        return ret;
    }
    for(auto &&box: detBboxes){
        if(box.objtype == ucloud::CLS_TYPE::PEDESTRIAN){
            bboxes.push_back(box);
        }
    }
    LOGI << bboxes.size() << " ped detected";
    if(bboxes.empty()) return RET_CODE::SUCCESS;

    transform_box_to_ped_box(bboxes);

    ret = m_clsHandle->run(tvimage,bboxes,threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_PHONE_DETECTOR phone classify images failed!\n", __FILE__, __LINE__);
        return ret;
    } 
    LOGI << "<- IMP_PHONE_DETECTOR::run";
    return ret;
}


ucloud::RET_CODE IMP_PHONE_DETECTOR::set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss){
    // vector<CLS_TYPE> output_dim_cls_order = {CLS_TYPE::OTHERS, m_cls, CLS_TYPE::OTHERS};
    return m_clsHandle->set_output_cls_order(output_clss);
}

ucloud::RET_CODE IMP_PHONE_DETECTOR::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return m_clsHandle->get_class_type(valid_clss);
}


