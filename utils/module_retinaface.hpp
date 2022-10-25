#ifndef _MODULE_RETINAFACE_HPP_
#define _MODULE_RETINAFACE_HPP_
#include "framework_detection.hpp"
#include "module_base.hpp"
#include "basic.hpp"
#include "module_track.hpp"
#include <atomic>

class RETINAFACE_DETECTION; //人脸检测无跟踪
class RETINAFACE_DETECTION_BYTETRACK;//人脸检测+跟踪

/*******************************************************************************
 * RETINAFACE_DETECTION
 * chaffee.chen@ucloud.cn 2022-10-11
*******************************************************************************/
class RETINAFACE_DETECTION: public ucloud::AlgoAPI{
public:
/**
 * public API
 */
    RETINAFACE_DETECTION();
    virtual ~RETINAFACE_DETECTION();
    virtual void release();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

    static float get_box_expand_ratio(){return m_expand_ratio;}
/**
 * non-public API
 */
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold,ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY);
    virtual ucloud::RET_CODE rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, ucloud::VecObjBBox &bboxes);
    
    float clip_threshold(float x);
    float clip_nms_threshold(float x);

    void gen_prior_box();

private:
    std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;
    
    DATA_SHAPE m_InpSp;
    int m_InpNum = 1;//输入Tensor数量
    int m_OtpNum = 3;//输出Tensor数量 3 locs, confs, landmss
    std::vector<int> m_OutEleNums;//输出Tensor元素总数
    float m_Var[2] = {0.1, 0.2};
    float* m_Anchors = nullptr;
    /**
     * 输出Tensor的维度:
     * 3x[1 na h w d ] flatten -> [1, na*h*w*d ]
     * 1x[1 nl na 2] flatten -> [1, nl*na*2]
     * ------------ OR -------------------
     * xy[1,D,2] wh[1,D,2] conf[1,D,NC+1] <====current situation
     **/
    std::vector<std::vector<int>> m_OutEleDims;
    PRE_PARAM m_param_img2tensor;

    ucloud::CLS_TYPE m_clss = ucloud::CLS_TYPE::FACE ;

    //当传入的参数超过边界时,采用默认数值
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    static constexpr float m_expand_ratio = 1.3;//返回的人脸检测框扩大比例


#ifdef TIMING
    Timer m_Tk;
#endif    
};

/*******************************************************************************
 * RETINAFACE_DETECTION_BYTETRACK
 * chaffee.chen@ucloud.cn 2022-10-11
*******************************************************************************/
class RETINAFACE_DETECTION_BYTETRACK:public AnyDetectionV4ByteTrack{
public:
    RETINAFACE_DETECTION_BYTETRACK(){
        m_detector = std::make_shared<RETINAFACE_DETECTION>();
    }
    virtual ~RETINAFACE_DETECTION_BYTETRACK(){}
};

#endif