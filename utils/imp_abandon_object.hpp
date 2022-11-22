#ifndef _IMP_ABANDON_OBJECT_HPP_
#define _IMP_ABANDON_OBJECT_HPP_
#include "module_base.hpp"
#include "basic.hpp"
#include "module_abandon_object.hpp"
#include <opencv2/opencv.hpp>



class IMP_ABANDON_DETECTOR:public ucloud::AlgoAPI{
public:
    IMP_ABANDON_DETECTOR(){
        LOGI<<"IMP_ABANDON_DETECTOR";
        m_abandon_ptr = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::ABANDON_OBJECT_DETECTOR);       
    };
    virtual ~IMP_ABANDON_DETECTOR(){};
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    // virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold=0.5, float nms_threshold=0.5);
    // virtual RET_CODE run(TvaiImage& tvimage, TvaiRect roi ,ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    virtual ucloud::RET_CODE run(ucloud::BatchImageIN &batch_tvimages, ucloud::VecObjBBox &bboxes);
    void updateBG(ucloud::BatchImageIN &batch_tvimages); 
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
private:
    std::shared_ptr<ucloud::AlgoAPI> m_abandon_ptr=nullptr;
    std::vector<float> m_Bg_weight{0.2,0.35,0.45};
    ucloud::TvaiImage m_BackGround;
    cv::Mat m_BGmat;
    bool m_upbg=false;

};

#endif