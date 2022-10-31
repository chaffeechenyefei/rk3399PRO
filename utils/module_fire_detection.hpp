#ifndef _MODULE_FIRE_DETECTION_HPP_
#define _MODULE_FIRE_DETECTION_HPP_
#include "module_base.hpp"
#include "module_classify.hpp"
#include "module_yolo.hpp"

/*******************************************************************************
 * FireDetector
 * YOLO_DETECTION_BYTETRACK + Classification
 * YOLO_DETECTION_BYTETRACK: 火焰检测器
 * Classification: 火焰分类
*******************************************************************************/
class FireDetector:public ucloud::AlgoAPI{
 public:
    FireDetector(){//直观看到用了哪些类实现的
        LOGI<<"-> FireDetector";
        // m_detectHandle = std::make_shared<YOLO_DETECTION_BYTETRACK>();
        m_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::FIRE_DETECTOR);
        m_clsHandle = std::make_shared<Classification>();
    }
    virtual ~FireDetector(){};
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold=0.5, float nms_threshold=0.5);
    //实际使用m_clsHandle管理的cls_type
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
private:
    std::shared_ptr<AlgoAPI> m_detectHandle = nullptr;
    std::shared_ptr<AlgoAPI> m_clsHandle = nullptr;
    float m_fire_classify_threshold = 0.6;
    float m_trust_threshold = 0.7;
};
#endif