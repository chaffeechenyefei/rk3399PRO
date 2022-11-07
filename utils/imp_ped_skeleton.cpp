#include "imp_ped_skeleton.hpp"

using namespace ucloud;

#define PI 3.1415
/*******************************************************************************
inner function
*******************************************************************************/
static uPoint get_head_center(BBox &box);
static uPoint get_upper_body_center(BBox &box);
static float calc_angle(uPoint &a, uPoint &b, uPoint &c);
static float calc_angle2(uPoint &a, uPoint &b, uPoint &c);

uPoint get_head_center(BBox &box){
    //HEAD
    uPoint head_center{0,0};
    for(int i=0; i < 5; i++){
        head_center.x += box.Pts.pts[i].x;
        head_center.y += box.Pts.pts[i].y;
    }
    head_center.x /= 5;
    head_center.y /= 5;
    return head_center;
}

uPoint get_upper_body_center(BBox &box){
    uPoint left_shoulder = box.Pts.pts[5];
    uPoint right_shoulder = box.Pts.pts[6];
    uPoint left_hip = box.Pts.pts[11];
    uPoint right_hip = box.Pts.pts[12];
    //UPPER_BODY_CENTER
    uPoint upper_body_center{0,0};
    upper_body_center.x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x)/4;
    upper_body_center.y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y)/4;
    return upper_body_center;
}

//angle between ac and bc. [0-180]
float calc_angle(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;
    /**
     * https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
     * The orientation of this angle matches that of the coordinate system. 
     * In a left-handed coordinate system, i.e. x pointing right and y down as is common for computer graphics, 
     * this will mean you get a positive sign for clockwise angles. 
     * If the orientation of the coordinate system is mathematical with y up, 
     * you get counter-clockwise angles as is the convention in mathematics. 
     * Changing the order of the inputs will change the sign, so if you are unhappy with the signs just swap the inputs.
     * 
     * https://en.cppreference.com/w/cpp/numeric/math/atan2
     * If no errors occur, the arc tangent of y/x (arctan(y/x)) in the range [-π , +π] radians, is returned.
     */
    float dot = ac.x*bc.x + ac.y*bc.y;
    float det = ac.x*bc.y - ac.y*bc.x;
    float angle = std::atan2(det,dot)*180/PI;
    return std::abs(angle);
}


float calc_angle2(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;

    float m = ac.x*bc.x + ac.y*bc.y;
    float n = std::sqrt( ac.x * ac.x + ac.y * ac.y )* std::sqrt( bc.x * bc.x + bc.y * bc.y ) + 1e-3;
    float cos_abc = std::acos(m/n)*180/PI;
    //[-180,180]
    if( cos_abc > 180) cos_abc = cos_abc - 360;
    if( cos_abc < -180) cos_abc = cos_abc + 360;
    //[0,180]
    // std::cout << "angle = " << cos_abc << std::endl;
    return std::abs(cos_abc);
}

/**Reference
 * "keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
*/

/*******************************************************************************
 * IMP_PED_FALLING_DETECTION
 * 行人摔倒检测
 * chaffee.chen@ucloud.cn 2022-11-07
*******************************************************************************/

RET_CODE IMP_PED_FALLING_DETECTION::init(std::map<InitParam, ucloud::WeightData> &weightConfig){
    LOGI << "-> IMP_PED_FALLING_DETECTION::init";
    WeightData ped_detect_modelpath ,sk_detect_modelpath;
    if(weightConfig.find(InitParam::BASE_MODEL)==weightConfig.end() || \
        weightConfig.find(InitParam::SUB_MODEL)==weightConfig.end()) {
            std::cout << weightConfig.size() << endl;
            for(auto param: weightConfig){
                printf( "[%d]:[%s], ", param.first, param.second);
            }
            printf("ERR:: IMP_PED_FALLING_DETECTION->init() still missing models\n");
            return RET_CODE::ERR_INIT_PARAM_FAILED;
        }
    RET_CODE ret = RET_CODE::FAILED;
    ped_detect_modelpath = weightConfig[InitParam::BASE_MODEL];
    sk_detect_modelpath = weightConfig[InitParam::SUB_MODEL];

    //ped detection
    ret = m_ped_detectHandle->init(ped_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    //sk detection
    ret = m_sk_detectHandle->init(sk_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- IMP_PED_FALLING_DETECTION::init";
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_PED_FALLING_DETECTION::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> IMP_PED_FALLING_DETECTION::init";
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

RET_CODE IMP_PED_FALLING_DETECTION::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> IMP_PED_FALLING_DETECTION::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    RET_CODE ret = RET_CODE::FAILED;
    float expand_scale = 1.2;
    VecObjBBox ped_bboxes;
    ret = m_ped_detectHandle->run(tvimage, ped_bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_ped_detectHandle return %d\n", __FILE__, __LINE__, ret);
        return ret;
    }

    // //filter
    // for(auto &&box: det_bboxes){
    //     if(box.objtype == CLS_TYPE::PEDESTRIAN) ped_bboxes.push_back(box);
    // }

    if(ped_bboxes.empty()) return RET_CODE::SUCCESS;
    ret = m_sk_detectHandle->run(tvimage,ped_bboxes,0.1,0.1);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_sk_detectHandle return %d\n", __FILE__, __LINE__, ret);
        return ret;
    }

    filter_valid_pose(tvimage, ped_bboxes, bboxes);

    LOGI << "<- IMP_PED_FALLING_DETECTION::run";
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_PED_FALLING_DETECTION::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}



void IMP_PED_FALLING_DETECTION::filter_valid_pose(TvaiImage &tvimage, VecObjBBox &bboxes_in, VecObjBBox &bboxes_out){
    for(auto &&box: bboxes_in){
        uPoint head = get_head_center(box);
        uPoint body = get_upper_body_center(box);
        uPoint cam = body;
        cam.y = body.y - 20;
        float angle = calc_angle(head, cam, body);
        if( angle > m_threshold_angle_of_body ){
            //摔倒规则通过
            box.objtype = m_cls;
            box.Pts.pts = {head, box.Pts.pts[5], box.Pts.pts[6]};
            bboxes_out.push_back(box);
        }
    // #ifndef MLU220 //只有MLU270的情况下,才返回行人数据供分析
    //     else {
    //         box.objtype = CLS_TYPE::PEDESTRIAN;
    //         bboxes_out.push_back(box);
    //     }
    // #endif        
    }

}


/*******************************************************************************
 * IMP_PED_BENDING_DETECTION
 * 行人弯腰检测
 * threshold = 行人的阈值
 * chaffee.chen@ucloud.cn 2022-11-07
*******************************************************************************/
void IMP_PED_BENDING_DETECTION::filter_valid_pose(TvaiImage &tvimage, VecObjBBox &bboxes_in, VecObjBBox &bboxes_out){
    for(auto &&box: bboxes_in){
        if(!is_valid_position(tvimage, box)) continue;
        float hw_ratio = ((float)box.rect.height) / box.rect.width;
        if( hw_ratio > 2) continue;
        if(box.Pts.pts.empty() || box.Pts.pts.size()!=17) continue;
        uPoint head = get_head_center(box);
        uPoint body = get_upper_body_center(box);
        uPoint cam = body;
        cam.y = body.y - 20;
        float angle = calc_angle(head, cam, body);
        if( angle > m_threshold_angle_of_body ){
            //规则通过
            box.objtype = CLS_TYPE::PEDESTRIAN_BEND;
            box.Pts.pts = {head, body, box.Pts.pts[5], box.Pts.pts[6] };
            bboxes_out.push_back(box);
        }
    // #ifndef MLU220 //只有MLU270的情况下,才返回行人数据供分析
    //     else {
    //         box.objtype = CLS_TYPE::PEDESTRIAN;
    //         bboxes_out.push_back(box);
    //     }
    // #endif        
    }

}


bool IMP_PED_BENDING_DETECTION::is_valid_position(TvaiImage &tvimage, BBox &boxIn){
    int H = tvimage.height;
    int W = tvimage.width;
    float ratio = 0.01;
    int minW = ratio*W;
    int maxW = W - minW;
    int minH = ratio*H;
    int maxH = H - minH;

    int x0 = boxIn.rect.x;
    int y0 = boxIn.rect.y;
    int x1 = boxIn.rect.x + boxIn.rect.width;
    int y1 = boxIn.rect.y + boxIn.rect.height;

    if( x0 < minW || x0 > maxW) return false;
    if( x1 < minW || x1 > maxW) return false;
    if( y0 < minH || y0 > maxH) return false;
    if( y1 < minH || y1 > maxH) return false;
    return true;    
}


RET_CODE IMP_PED_BENDING_DETECTION::init(std::map<InitParam, ucloud::WeightData> &weightConfig){
    LOGI << "-> IMP_PED_BENDING_DETECTION::init";
    WeightData ped_detect_modelpath ,sk_detect_modelpath;
    if(weightConfig.find(InitParam::BASE_MODEL)==weightConfig.end() || \
        weightConfig.find(InitParam::SUB_MODEL)==weightConfig.end()) {
            std::cout << weightConfig.size() << endl;
            for(auto param: weightConfig){
                printf( "[%d]:[%s], ", param.first, param.second);
            }
            printf("ERR:: IMP_PED_BENDING_DETECTION->init() still missing models\n");
            return RET_CODE::ERR_INIT_PARAM_FAILED;
        }
    RET_CODE ret = RET_CODE::FAILED;
    ped_detect_modelpath = weightConfig[InitParam::BASE_MODEL];
    sk_detect_modelpath = weightConfig[InitParam::SUB_MODEL];

    //ped detection
    ret = m_ped_detectHandle->init(ped_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    //sk detection
    ret = m_sk_detectHandle->init(sk_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- IMP_PED_BENDING_DETECTION::init";
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_PED_BENDING_DETECTION::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> IMP_PED_BENDING_DETECTION::init";
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

RET_CODE IMP_PED_BENDING_DETECTION::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> IMP_PED_BENDING_DETECTION::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    RET_CODE ret = RET_CODE::FAILED;
    float expand_scale = 1.2;
    VecObjBBox det_bboxes, ped_bboxes;
    ret = m_ped_detectHandle->run(tvimage, det_bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_ped_detectHandle return %d\n", __FILE__, __LINE__, ret);
        return ret;
    }

    //filter
    for(auto &&box: det_bboxes){
        if(box.objtype == CLS_TYPE::PEDESTRIAN) ped_bboxes.push_back(box);
    }

    if(ped_bboxes.empty()) return RET_CODE::SUCCESS;
    ret = m_sk_detectHandle->run(tvimage,ped_bboxes,0.1,0.1);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_sk_detectHandle return %d\n", __FILE__, __LINE__, ret);
        return ret;
    }

    filter_valid_pose(tvimage, ped_bboxes, bboxes);

    LOGI << "<- IMP_PED_BENDING_DETECTION::run";
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_PED_BENDING_DETECTION::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}