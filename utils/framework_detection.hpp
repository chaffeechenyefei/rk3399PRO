/**
 * 套个壳, 方便各种检测网络叠加不同跟踪器
*/
#ifndef _FRAMEWORK_DETECTION_HPP_
#define _FRAMEWORK_DETECTION_HPP_

#include "../libai_core.hpp"
#include "module_track.hpp"
#include <mutex>
#include "basic.hpp"


/*******************************************************************************
目录
*******************************************************************************/
// class AnyDetectionV4DeepSort;//任意检测模型+DeepSort
class AnyDetectionV4ByteTrack;//任意检测模型+ByteTrack
class PipelineNaive;//任意模型管道式组合


/*******************************************************************************
AnyDetection + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
class AnyDetectionV4ByteTrack:public ucloud::AlgoAPI{
public:
    AnyDetectionV4ByteTrack();
    virtual RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual RET_CODE init(const std::string &modelpath);
    virtual ~AnyDetectionV4ByteTrack(){};
    virtual RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);

    /** -----------------non AlgoAPI-------------------**/
    virtual RET_CODE set_trackor(TRACKMETHOD trackmethod);
    virtual RET_CODE set_detector(ucloud::AlgoAPI* ptr);

protected:
    float clip_threshold(float x);
    float clip_nms_threshold(float x);
    ucloud::AlgoAPISPtr m_detector = nullptr;
    std::shared_ptr<TrackPoolAPI<BYTETRACKPARM>> m_trackor = nullptr;

    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    int m_fps = 5;
    int m_nn_buf = 10;
};


/*******************************************************************************
PipelineNaive
chaffee.chen@2022-10-09
*******************************************************************************/
class PipelineNaive: public ucloud::AlgoAPI{
public:
    PipelineNaive(){}
    virtual ~PipelineNaive(){}

    virtual RET_CODE run(TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

    /**push_back 
     * handles, threshold, nms_threshold and 是否使用run传入的阈值参数 fixed_threshold=true表示不使用
     * 需要放入成熟的AlgoAPISPtr
     * **/
    virtual RET_CODE push_back(ucloud::AlgoAPISPtr apihandle, bool fixed_threshold=true, float threshold=0.55, float nms_threshold=0.6){
        m_handles.push_back(apihandle);
        m_thresholds.push_back(threshold);
        m_nms_thresholds.push_back(nms_threshold);
        if(!fixed_threshold) unfixed_thresholds_index = m_handles.size()-1;
    }

protected:
    std::vector<ucloud::AlgoAPISPtr> m_handles;
    std::vector<float> m_thresholds;
    std::vector<float> m_nms_thresholds;
    int unfixed_thresholds_index = -1;

};


#endif