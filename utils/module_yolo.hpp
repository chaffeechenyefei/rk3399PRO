#ifndef _MODULE_YOLO_HPP_
#define _MODULE_YOLO_HPP_

#include "module_base.hpp"
#include "basic.hpp"
#include "module_track.hpp"
#include <atomic>
#include "framework_detection.hpp"
#include <string>

// class YOLO_DETECTION;//带跟踪
class YOLO_DETECTION_NAIVE;//不带跟踪
class YOLO_DETECTION_BYTETRACK;//跟踪框架



/*******************************************************************************
 * YOLO_DETECTION_NAIVE 无跟踪, 纯检测
 * yolov5系列支持, 需要配合特定export_rknn.py输出的结果
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
class YOLO_DETECTION_NAIVE: public ucloud::AlgoAPI{
public:
/**
 * public API
 */
    YOLO_DETECTION_NAIVE();
    virtual ~YOLO_DETECTION_NAIVE();
    virtual void release();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, std::string &filename, float threshold=0.5, float nms_threshold=0.6);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);
/**
 * non-public API
 */
//对roi区域进行检测
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi ,ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
protected:

    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, ucloud::TvaiRect roi, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY,const vector<vector<int> > anchors,const vector<int> strides);
    /** mode=0: Detect Layer标准输出
     * 模型输出Tensor的维度:
     * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1]
     * dim0 = 2,NC+1
     * dim1 = L
     * dim2 = 1
     **/    
    virtual bool check_output_dims_1LX();
    virtual ucloud::RET_CODE rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes); 
    virtual ucloud::RET_CODE rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes, const vector<vector<int> > &anchors, const vector<int> &strides);
    
    float clip_threshold(float x);
    float clip_nms_threshold(float x);


private:
    std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;//前处理模块
    
    DATA_SHAPE m_InpSp;
    int m_InpNum = 1;//输入Tensor数量
    // int m_OtpNum = 4;//输出Tensor数量 3 layer+1 anchor_gir
    int m_OtpNum = 3;//输出Tensor数量 3 xy, wh, conf+prob
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

   

#ifdef TIMING
    Timer m_Tk;
#endif    
};

/*******************************************************************************
 * YOLO_DETECTION_BYTETRACK
*******************************************************************************/
class YOLO_DETECTION_BYTETRACK:public AnyDetectionV4ByteTrack{
public:
    YOLO_DETECTION_BYTETRACK(){
        m_detector = std::make_shared<YOLO_DETECTION_NAIVE>();
    }
    virtual ~YOLO_DETECTION_BYTETRACK(){}
};


/*******************************************************************************
 * YOLO_DETECTION 内部带跟踪器 弃用
 * yolov5系列支持, 需要配合特定export_rknn.py输出的结果
 * chaffee.chen@ucloud.cn 2022-09-19
*******************************************************************************/
// class YOLO_DETECTION: public ucloud::AlgoAPI{
// public:
// /**
//  * public API
//  */
//     YOLO_DETECTION();
//     virtual ~YOLO_DETECTION();
//     virtual void release();
//     virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
//     virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
//     // virtual ucloud::RET_CODE set_param(float threshold, float nms_threshold);
//     virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
//     virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);
// /**
//  * non-public API
//  */
// protected:
//     virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage& tvimage ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
//     virtual ucloud::RET_CODE postprocess_drm(std::vector<float*> &output_datas, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
//     virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aspect_ratios);
//     virtual ucloud::RET_CODE postprocess_opencv(std::vector<float*> &output_datas, float threshold ,float nms_threshold, ucloud::VecObjBBox &bboxes, std::vector<float> &aspect_ratios);
//     /** mode=0: Detect Layer标准输出
//      * 模型输出Tensor的维度:
//      * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1]
//      * dim0 = 2,NC+1
//      * dim1 = L
//      * dim2 = 1
//      **/    
//     virtual bool check_output_dims_1LX();
//     virtual ucloud::RET_CODE rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes);
//     /** mode=1: Detect Layer仅进行了sigmoid
//      * 模型输出Tensor的维度:
//      * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1] anchors_grid [nl,na,2]
//      * dim0 = 2,NC+1
//      * dim1 = L
//      * dim2 = 1
//      **/
//     virtual bool check_output_dims_1LX2(){return true;};
//     virtual ucloud::RET_CODE rknn_output_to_boxes_1LX2( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes);  
//     /** mode=2: Detect Layer仅进行了sigmoid, 且不进行permute
//      * 模型输出Tensor的维度:
//      * wywhs [1,na*no,h,w]xnl(3) anchors_grid [nl,na,2]
//      * dim0 = w
//      * dim1 = h
//      * dim2 = na*no
//      **/    
//     virtual ucloud::RET_CODE rknn_output_to_boxes_1LX3( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes);  
//     float clip_threshold(float x);
//     float clip_nms_threshold(float x);

// protected:
//     std::shared_ptr<TrackPoolAPI<BYTETRACKPARM>> m_track = nullptr;

// private:
//     std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
//     std::shared_ptr<ImageUtil> m_drm = nullptr;
    
//     BYTETRACKPARM m_track_param;
//     DATA_SHAPE m_InpSp;
//     int m_InpNum = 1;//输入Tensor数量
//     // int m_OtpNum = 4;//输出Tensor数量 3 layer+1 anchor_gir
//     int m_OtpNum = 3;//输出Tensor数量 3 xy, wh, conf+prob
//     int m_nc = 0;
//     int m_nl = 3;//3 layers
//     std::vector<int> m_OutEleNums;//输出Tensor元素总数
//     /**
//      * 输出Tensor的维度:
//      * 3x[1 na h w d ] flatten -> [1, na*h*w*d ]
//      * 1x[1 nl na 2] flatten -> [1, nl*na*2]
//      * ------------ OR -------------------
//      * xy[1,D,2] wh[1,D,2] conf[1,D,NC+1] <====current situation
//      **/
//     std::vector<std::vector<int>> m_OutEleDims;
//     PRE_PARAM m_param_img2tensor;
//     std::vector<int> m_strides;

//     std::vector<ucloud::CLS_TYPE> m_clss;
//     std::map<ucloud::CLS_TYPE, int> m_unique_clss_map;

//     //当传入的参数超过边界时,采用默认数值
//     float m_default_threshold = 0.4;
//     float m_default_nms_threshold = 0.4;

//     // int m_fps = 25;
//     // int m_nn_buf = 30;
//     int m_fps = 5;
//     int m_nn_buf = 10;

// #ifdef TIMING
//     Timer m_Tk;
// #endif    
// };

#endif