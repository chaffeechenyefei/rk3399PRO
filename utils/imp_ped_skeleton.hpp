#ifndef _IMP_PED_SKELETON_HPP_
#define _IMP_PED_SKELETON_HPP_

#include "basic.hpp"
#include "module_track.hpp"
#include "module_yolo.hpp"
#include "module_posenet.hpp"


class IMP_PED_FALLING_DETECTION;//行人摔倒检测
class IMP_PED_BENDING_DETECTION;//行人弯腰检测

/*******************************************************************************
 * IMP_PED_FALLING_DETECTION
 * 行人摔倒检测
 * threshold = 行人的阈值
 * chaffee.chen@ucloud.cn 2022-11-07
*******************************************************************************/
class IMP_PED_FALLING_DETECTION: public ucloud::AlgoAPI{
public:
    IMP_PED_FALLING_DETECTION(){
        m_ped_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::PED_FALL_DETECTOR);
        m_sk_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::SKELETON_DETECTOR);
    }
    virtual ~IMP_PED_FALLING_DETECTION(){}
    virtual RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold, float nms_threshold);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

protected:
    //过滤出符合的体态
    virtual void filter_valid_pose(TvaiImage &tvimage, VecObjBBox &bboxes_in, VecObjBBox &bboxes_out);

private:
    float m_threshold_angle_of_body = 30;//degree
    // float m_ped_threshold = 0.7;
    ucloud::AlgoAPISPtr m_ped_detectHandle = nullptr;
    ucloud::AlgoAPISPtr m_sk_detectHandle = nullptr;
    
    ucloud::CLS_TYPE m_cls = ucloud::CLS_TYPE::PEDESTRIAN_FALL;

};

/*******************************************************************************
 * IMP_PED_BENDING_DETECTION
 * 行人弯腰检测
 * threshold = 行人的阈值
 * chaffee.chen@ucloud.cn 2022-11-07
*******************************************************************************/
class IMP_PED_BENDING_DETECTION: public ucloud::AlgoAPI{
public:
    IMP_PED_BENDING_DETECTION(){
        m_ped_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::GENERAL_DETECTOR);
        m_sk_detectHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::SKELETON_DETECTOR);
    }
    virtual ~IMP_PED_BENDING_DETECTION(){};
    virtual RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold, float nms_threshold);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
protected:
    virtual void filter_valid_pose(TvaiImage &tvimage, VecObjBBox &bboxes_in, VecObjBBox &bboxes_out);

    bool is_valid_position(TvaiImage &tvimage, BBox &boxIn);

private:
    ucloud::CLS_TYPE m_cls = ucloud::CLS_TYPE::PEDESTRIAN_BEND;
    float m_threshold_angle_of_body = 50;//degree

    ucloud::AlgoAPISPtr m_ped_detectHandle = nullptr;
    ucloud::AlgoAPISPtr m_sk_detectHandle = nullptr;
};


#endif