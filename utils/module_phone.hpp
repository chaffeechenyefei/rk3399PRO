#ifndef _MODULE_PHONE_HPP_
#define _MODULE_PHONE_HPP_
#include "module_base.hpp"
#include "module_classify.hpp"
#include "module_yolo.hpp"

/*******************************************************************************
 * PhoneDetector
 * YOLO_DETECTION_BYTETRACK + Classification
 * YOLO_DETECTION_BYTETRACK: 使用人车非通用检测
 * Classification: 打电话检测
*******************************************************************************/
class PhoneDetector:public ucloud::AlgoAPI{
 public:
    PhoneDetector(){//直观看到用了哪些类实现的
        LOGI<<"-> PhoneDetector";
        m_ped_detectHandle = std::make_shared<YOLO_DETECTION_BYTETRACK>();
        m_clsHandle = std::make_shared<Classification>();
    }
    virtual ~PhoneDetector(){};
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold=0.5, float nms_threshold=0.5);
    //实际使用m_clsHandle管理的cls_type
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);
private:
    void transform_box_to_ped_box(ucloud::VecObjBBox &in_boxes);
    std::shared_ptr<AlgoAPI> m_ped_detectHandle = nullptr;
    std::shared_ptr<AlgoAPI> m_clsHandle = nullptr;
    float m_ped_threshold = 0.4;
    float m_phone_theshold = 0.5;
};
#endif