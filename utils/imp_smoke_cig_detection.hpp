#ifndef _IMP_SMOKE_CIG_DETECTION_HPP_
#define _IMP_SMOKE_CIG_DETECTION_HPP_

#include "module_base.hpp"
#include "basic.hpp"
#include "module_track.hpp"
#include "module_yolo.hpp"
#include "module_retinaface.hpp"

/*******************************************************************************
 * SMOKE_CIG_DETECTION
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
class IMP_SMOKE_CIG_DETECTION: public ucloud::AlgoAPI{
public:
    IMP_SMOKE_CIG_DETECTION(){
        // m_face_detectHandle = std::make_shared<RETINAFACE_DETECTION_BYTETRACK>();
        m_face_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::FACE_DETECTOR);
        m_cig_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::CIG_DETECTOR_NO_TRACK);
    }
    ~IMP_SMOKE_CIG_DETECTION(){}
    RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

private:
    float m_face_threshold = 0.7;

    ucloud::AlgoAPISPtr m_face_detectHandle = nullptr;
    ucloud::AlgoAPISPtr m_cig_detectHandle = nullptr;
    
    ucloud::CLS_TYPE m_cls = ucloud::CLS_TYPE::SMOKING;

};


#endif