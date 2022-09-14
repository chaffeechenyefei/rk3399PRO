#ifndef _MODULE_YOLO_HPP_
#define _MODULE_YOLO_HPP_

#include "module_base.hpp"
#include "basic.hpp"

class YOLO_DETECTION: public ucloud::AlgoAPI{
public:
/**
 * public API
 */
    YOLO_DETECTION();
    virtual ~YOLO_DETECTION();
    virtual void release();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes);
    virtual ucloud::RET_CODE set_param(float threshold, float nms_threshold);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);
/**
 * non-public API
 */
protected:
    virtual ucloud::RET_CODE preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aspect_ratios);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, ucloud::VecObjBBox &bboxes, std::vector<float> &aspect_ratios);
    virtual ucloud::RET_CODE rknn_output_to_boxes( std::vector<float*> &output_datas,std::vector<ucloud::VecObjBBox> &bboxes);
    /**
     * [1 na h w d ] -> [ d w h na 1] -> [d w h na] -> dim0:d dim1:w dim2:h dim3:na
     */
    virtual bool check_output_dims();

private:
    std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
    DATA_SHAPE m_InpSp;
    int m_InpNum = 1;//输入Tensor数量
    int m_OtpNum = 4;//输出Tensor数量 3 layer+1 anchor_gir
    int m_nc = 0;
    int m_nl = 3;//3 layers
    std::vector<int> m_OutEleNums;//输出Tensor元素总数
    /**
     * 输出Tensor的维度:reversed [3 52 92 14] 3:nl, h,w, 14:xywh+nc
     * [1 na h w d ] -> [ d w h na 1] -> [d w h na] -> dim0:d dim1:w dim2:h dim3:na
     **/
    std::vector<std::vector<int>> m_OutEleDims;
    PRE_PARAM m_param_img2tensor;
    std::vector<int> m_strides;

    std::vector<ucloud::CLS_TYPE> m_clss;
    std::map<ucloud::CLS_TYPE, int> m_unique_clss_map;
    float m_threshold = 0.6;
    float m_nms_threshold = 0.6;

#ifdef TIMING
    Timer m_Tk;
#endif    
};


#endif