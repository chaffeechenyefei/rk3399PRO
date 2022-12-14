#ifndef _IMP_LICPLATE_DETECTION_HPP_
#define _IMP_LICPLATE_DETECTION_HPP_

#include "../libai_core.hpp"
#include "module_base.hpp"
/*******************************************************************************
 * IMP_XXX: 表示所有模块都通过ucloud::AICoreFactory::getAlgoAPI来实现, run时仅进行模块之间的衔接处理.
*******************************************************************************/


/*******************************************************************************
 * IMP_LICPLATE_DETECTOR
*******************************************************************************/
class IMP_LICPLATE_DETECTOR:public ucloud::AlgoAPI{
 public:
    IMP_LICPLATE_DETECTOR(){//直观看到用了哪些类实现的
        LOGI<<"-> IMP_LICPLATE_DETECTOR";
        // m_detectHandle = std::make_shared<YOLO_DETECTION_BYTETRACK>();
        m_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::LICPLATE_DETECTOR);
        m_clsHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::LICPLATE_RECOGNIZER_ONLY);
    }
    virtual ~IMP_LICPLATE_DETECTOR(){};
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold=0.5, float nms_threshold=0.5);
    
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
private:
    std::shared_ptr<AlgoAPI> m_detectHandle = nullptr;
    std::shared_ptr<AlgoAPI> m_clsHandle = nullptr;
};
#endif