#ifndef _MODULE_YOLO_U_HPP_
#define _MODULE_YOLO_U_HPP_

#include "module_base.hpp"
#include "basic.hpp"
#include "module_track.hpp"
#include <atomic>
#include "framework_detection.hpp"

/*******************************************************************************
 * general_infer_uint8_nhwc_to_uint8的使用范例
*******************************************************************************/


// class YOLO_DETECTION;//带跟踪
class YOLO_DETECTION_UINT8;//不带跟踪
class YOLO_DETECTION_UINT8_BYTETRACK;

/*******************************************************************************
 * YOLO_DETECTION_UINT8 无跟踪, 纯检测
 * yolov5系列支持, 需要配合特定export_rknn.py输出的结果
 * chaffee.chen@ucloud.cn 2022-10-27
*******************************************************************************/
class YOLO_DETECTION_UINT8: public ucloud::AlgoAPI{
public:
/**
 * public API
 */
    YOLO_DETECTION_UINT8();
    virtual ~YOLO_DETECTION_UINT8();
    virtual void release();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);

    ucloud::RET_CODE set_anchor(std::vector<float> &anchors);
/**
 * non-public API
 */
//对roi区域进行检测
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi ,ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
protected:
    /*******************************************************************************
     * uint8_t
    *******************************************************************************/
    virtual ucloud::RET_CODE postprocess(std::vector<uint8_t*> &output_datas, std::vector<float> &output_scales, std::vector<uint32_t> &output_zp,float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    virtual ucloud::RET_CODE postprocess(std::vector<uint8_t*> &output_datas, std::vector<float> &output_scales, std::vector<uint32_t> &output_zp, ucloud::TvaiRect roi, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    int post_process_forked_rknn(
        std::vector<uint8_t*> output_datas ,std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales, 
        float conf_threshold, VecObjBBox &result);
    /*******************************************************************************
     * float
    *******************************************************************************/
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, ucloud::TvaiRect roi, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    int post_process_forked_rknn(
        std::vector<float*> output_datas, 
        float conf_threshold, VecObjBBox &result); 

    float clip_threshold(float x);
    float clip_nms_threshold(float x);

    

       


private:
    std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;//前处理模块
    
    DATA_SHAPE m_InpSp;
    int m_InpNum = 1;//输入Tensor数量
    // int m_OtpNum = 4;//输出Tensor数量 3 layer+1 anchor_gir
    int m_OtpNum = 4;//输出Tensor数量 3 xy, wh, conf+prob
    int m_nc = 0;
    int m_nl = 3;//3 layers
    std::vector<int> m_OutEleNums;//输出Tensor元素总数
    /**
     * 输出Tensor的维度:
     * 3x[1 na h w d ] flatten -> [1, na*h*w*d ]
     * 1x[1 nl na 2] flatten -> [1, nl*na*2]
     * ------------ OR -------------------
     * xy[1,D,2] wh[1,D,2] conf[1,D,NC+1] <====current situation
     **/
    std::vector<std::vector<int>> m_OutEleDims;
    PRE_PARAM m_param_img2tensor;
    std::vector<int> m_strides;

    std::vector<ucloud::CLS_TYPE> m_clss;
    std::map<ucloud::CLS_TYPE, int> m_unique_clss_map;

    //当传入的参数超过边界时,采用默认数值
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    std::vector<float> m_anchors;


#ifdef TIMING
    Timer m_Tk;
#endif    
};

/*******************************************************************************
 * YOLO_DETECTION_UINT8_BYTETRACK
*******************************************************************************/
class YOLO_DETECTION_UINT8_BYTETRACK:public AnyDetectionV4ByteTrack{
public:
    YOLO_DETECTION_UINT8_BYTETRACK(){
        m_detector = std::make_shared<YOLO_DETECTION_UINT8>();
    }
    virtual ~YOLO_DETECTION_UINT8_BYTETRACK(){}
};

#endif