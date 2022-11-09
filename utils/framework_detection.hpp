/**
 * 套个壳, 方便各种检测网络叠加不同跟踪器
*/
#ifndef _FRAMEWORK_DETECTION_HPP_
#define _FRAMEWORK_DETECTION_HPP_

// #include "../libai_core.hpp"
#include "module_track.hpp"
#include <mutex>
#include "basic.hpp"
#include "module_base.hpp"
#include <functional>
#include <string>


/*******************************************************************************
目录
*******************************************************************************/
// class AnyDetectionV4DeepSort;//任意检测模型+DeepSort
class AnyDetectionV4ByteTrack;//任意检测模型+ByteTrack
class PipelineNaive;//任意模型管道式组合
class AnyModelWithBBox;//通用推理模型, 输入tvimage和VecBBox,针对VecBBox给的TvaiRect进行推理, 多态postprocess
class AnyModelWithTvaiImage; //通用推理模型, 多态postprocess



/*******************************************************************************
AnyDetection + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
class AnyDetectionV4ByteTrack:public ucloud::AlgoAPI{
public:
    AnyDetectionV4ByteTrack();
    virtual RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual RET_CODE init(const std::string &modelpath);
    virtual RET_CODE init(ucloud::WeightData weightConfig);
    virtual ~AnyDetectionV4ByteTrack(){};
    virtual RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE run(TvaiImage &tvimage, ucloud::VecObjBBox &bboxes,std::string &filename,float threshold=0.55,float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);

    /** -----------------non AlgoAPI-------------------**/
    virtual RET_CODE set_trackor(TRACKMETHOD trackmethod);
    virtual RET_CODE set_detector(ucloud::AlgoAPI* ptr);
    virtual RET_CODE set_anchor(std::vector<float> &anchors);//yolo

protected:
    float clip_threshold(float x);
    float clip_nms_threshold(float x);
    ucloud::AlgoAPISPtr m_detector = nullptr;
    std::shared_ptr<TrackPoolAPI<BYTETRACKPARM>> m_trackor = nullptr;

    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;
    std::vector<float> m_anchors;
    int m_fps = 5;
    int m_nn_buf = 10;
#ifdef TIMING
    Timer m_Tk;
#endif
};


/*******************************************************************************
PipelineNaive
chaffee.chen@2022-10-09
*******************************************************************************/
typedef std::function<ucloud::RET_CODE(ucloud::VecBBoxIN&, ucloud::VecBBoxOUT&)> FilterFuncPtr;
class PipelineNaive: public ucloud::AlgoAPI{
public:
    PipelineNaive(){}
    virtual ~PipelineNaive(){}

    virtual RET_CODE run(TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    /*******************************************************************************
     * push_back
     * handles, threshold, nms_threshold and 是否使用run传入的阈值参数 fixed_threshold=true表示不使用
     * 需要放入成熟的AlgoAPISPtr
    *******************************************************************************/
    virtual RET_CODE push_back(ucloud::AlgoAPISPtr apihandle, bool fixed_threshold=true, float threshold=0.55, float nms_threshold=0.6, FilterFuncPtr filter_func=nullptr ){
        m_handles.push_back(apihandle);
        m_thresholds.push_back(threshold);
        m_nms_thresholds.push_back(nms_threshold);
        m_filter_funcs.push_back(filter_func);
        if(!fixed_threshold) unfixed_thresholds_index = m_handles.size()-1;
    }
#ifdef TIMING
    Timer m_Tk;
#endif  
protected:
    std::vector<ucloud::AlgoAPISPtr> m_handles;
    std::vector<float> m_thresholds;
    std::vector<float> m_nms_thresholds;
    std::vector<FilterFuncPtr> m_filter_funcs;
    int unfixed_thresholds_index = -1;

};


/*******************************************************************************
 * AnyModelWithBBox RGB input only
 * 适用类型:
 *  1. 模型推理结果(outputs)的数据指针不会通过VecObjBBox传到外部
 *  2. 运行run时, 只对VecObjBBox中的有效rect区域进行推理
 *  3. 使用时, 需要对postprocess进行重载
 * chaffee.chen@2022-11-03
*******************************************************************************/
class AnyModelWithBBox: public ucloud::AlgoAPI{
public:
    AnyModelWithBBox();
    virtual ~AnyModelWithBBox();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    /*******************************************************************************
     * run 对bboxes中的每个区域进行特征提取, 并将结果更新到bboxes中(objtype,tvaifeature)
    *******************************************************************************/
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas ,ucloud::BBox &bbox, float aX, float aY);

protected:
    std::shared_ptr<BaseModel> m_net = nullptr;
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;
    
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;
    PRE_PARAM m_param_img2tensor;

    int m_InpNum = 0;
    int m_OutNum = 0;

    std::vector<int> m_OutEleNums;
    std::vector<std::vector<int>> m_OutEleDims;

#ifdef TIMING
    Timer m_TK;
#endif
};


/*******************************************************************************
 * AnyModelWithTvaiImage RGB input only
 * chaffee.chen@2022-11-03
*******************************************************************************/
class AnyModelWithTvaiImage: public ucloud::AlgoAPI{
public:
    AnyModelWithTvaiImage();
    virtual ~AnyModelWithTvaiImage();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi ,ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, ucloud::TvaiRect roi, float threshold, float nms_threshold ,ucloud::VecObjBBox &bboxes,float aX, float aY);

    float clip_threshold(float x);
    float clip_nms_threshold(float x);

protected:
    std::shared_ptr<BaseModel> m_net = nullptr;
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;
    
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;
    PRE_PARAM m_param_img2tensor;

    int m_InpNum = 0;
    int m_OutNum = 0;

    std::vector<int> m_OutEleNums;
    std::vector<std::vector<int>> m_OutEleDims;

    //当传入的参数超过边界时,采用默认数值
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

#ifdef TIMING
    Timer m_TK;
#endif
};

#endif