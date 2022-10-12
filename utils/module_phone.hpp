#ifndef _MODULE_PHONE_HPP_
#define _MODULE_PHONE_HPP_
#include "module_base.hpp"
#include "module_classify.hpp"
#include "module_yolo.hpp"

class PhoneDetector:public ucloud::AlgoAPI{
 public:
    PhoneDetector();
    virtual ~PhoneDetector();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold=0.5, float nms_threshold=0.5);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss, int select_idx=1);
private:
    void transform_box_to_ped_box(ucloud::VecObjBBox &in_boxes, ucloud::VecObjBBox &out_boxes,int &imheight,int &imwidth);
    std::shared_ptr<YOLO_DETECTION> m_ped_detectHandle=nullptr;
    std::shared_ptr<Classification> m_clsHandle = nullptr;
    float m_ped_threshold = 0.4;
    float m_phone_theshold = 0.5;
    int m_select = 1;
    std::vector<ucloud::CLS_TYPE> m_clss;
    std::map<ucloud::CLS_TYPE, int> m_unique_clss_map;
    int m_nc;///number classes
};
#endif