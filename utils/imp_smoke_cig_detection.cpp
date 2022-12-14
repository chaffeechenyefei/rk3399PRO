#include "imp_smoke_cig_detection.hpp"

using namespace ucloud;

static bool box_sort_confidence_max(const BBox& a, const BBox& b){
    return a.confidence > b.confidence;
}

/*******************************************************************************
 * IMP_SMOKE_CIG_DETECTION
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
RET_CODE IMP_SMOKE_CIG_DETECTION::init(std::map<InitParam, ucloud::WeightData> &weightConfig){
    LOGI << "-> IMP_SMOKE_CIG_DETECTION::init";
    WeightData face_detect_modelpath ,cig_detect_modelpath;
    if(weightConfig.find(InitParam::BASE_MODEL)==weightConfig.end() || \
        weightConfig.find(InitParam::SUB_MODEL)==weightConfig.end()) {
            std::cout << weightConfig.size() << endl;
            for(auto param: weightConfig){
                printf( "[%d]:[%s], ", param.first, param.second);
            }
            printf("ERR:: SmokingDetectionV2->init() still missing models\n");
            return RET_CODE::ERR_INIT_PARAM_FAILED;
        }
    RET_CODE ret = RET_CODE::FAILED;
    face_detect_modelpath = weightConfig[InitParam::BASE_MODEL];
    cig_detect_modelpath = weightConfig[InitParam::SUB_MODEL];

    //face detection
    ret = m_face_detectHandle->init(face_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    //cig detection
    ret = m_cig_detectHandle->init(cig_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    vector<CLS_TYPE> cls_types = {m_cls};
    m_cig_detectHandle->set_output_cls_order(cls_types);
    
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_SMOKE_CIG_DETECTION::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> IMP_SMOKE_CIG_DETECTION::init";
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
    // if(ret!=RET_CODE::SUCCESS) return ret;
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_SMOKE_CIG_DETECTION::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> IMP_SMOKE_CIG_DETECTION::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    RET_CODE ret = RET_CODE::FAILED;
    float expand_scale = 1.5;
    VecObjBBox face_bboxes;
    ret = m_face_detectHandle->run(tvimage, face_bboxes, m_face_threshold, 0.6);
    for(auto &&face_bbox: face_bboxes){
        VecObjBBox target_bboxes;
        ucloud::TvaiRect scaled_face_rect = globalscaleTvaiRect(face_bbox.rect, expand_scale, tvimage.width, tvimage.height);
        ret = m_cig_detectHandle->run(tvimage, scaled_face_rect, target_bboxes, threshold, nms_threshold);
        if(!target_bboxes.empty()){
            std::sort(target_bboxes.begin(), target_bboxes.end(), box_sort_confidence_max );
            face_bbox.confidence = target_bboxes[0].confidence;
            face_bbox.objectness = target_bboxes[0].objectness;
            face_bbox.rect = scaled_face_rect;
            face_bbox.objtype = target_bboxes[0].objtype;
            bboxes.push_back(face_bbox);
        } else {//???????????????????????????
            bboxes.push_back(face_bbox);
        }

        //
        // for(auto &&target_bbox: target_bboxes)
        // {
        //     bboxes.push_back(target_bbox);
        // }
    }
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_SMOKE_CIG_DETECTION::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}
