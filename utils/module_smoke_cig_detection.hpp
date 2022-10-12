#ifndef _MODULE_SMOKE_CIG_DETECTION_HPP_
#define _MODULE_SMOKE_CIG_DETECTION_HPP_

#include "module_base.hpp"
#include "basic.hpp"
#include "module_track.hpp"
#include "module_yolo.hpp"
#include "module_retinaface.hpp"

/*******************************************************************************
 * SMOKE_CIG_DETECTION
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
class SMOKE_CIG_DETECTION: public ucloud::AlgoAPI{
public:
    SMOKE_CIG_DETECTION(){
        m_face_detectHandle = std::make_shared<RETINAFACE_DETECTION_BYTETRACK>();
        m_cig_detectHandle = std::make_shared<YOLO_DETECTION_NAIVE>();
    }
    ~SMOKE_CIG_DETECTION(){}
    RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

private:
    float m_face_threshold = 0.7;

    ucloud::AlgoAPISPtr m_face_detectHandle = nullptr;
    std::shared_ptr<ucloud::AlgoAPI> m_cig_detectHandle = nullptr;
    
    ucloud::CLS_TYPE m_cls = ucloud::CLS_TYPE::SMOKING;

};


#endif