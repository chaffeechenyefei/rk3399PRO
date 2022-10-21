#ifndef _MODULE_CLASSIFY_HPP_
#define _MODULE_CLASSIFY_HPP_
#include <vector>
#include "module_base.hpp"
#include "basic.hpp"
#include <string>

/*******************************************************************************
 * Classification
 * 通用分类器
 * lihui.liu@2022-10-13
*******************************************************************************/
class Classification: public ucloud::AlgoAPI{
public:
    Classification();
    virtual ~Classification();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    /*******************************************************************************
     * run 对bboxes中的每个区域进行分类, 并将结果更新到bboxes中(objtype, objectness, confidence)
     * PARAM:
     *  threshold只有超过阈值的分类结果才会更新到bboxes中
    *******************************************************************************/
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, std::string &filename, float threshold=0.5, float nms_threshold=0.5);

    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
    /*******************************************************************************
     * get_class_type 返回剔除占位类型OTHERS后的有效分类类别
    *******************************************************************************/    
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    /*******************************************************************************
     * set_output_cls_order
     * 例如 {OTHERS, FIRE, OTHERS} 长度必须与模型输出一致 否则后续运行会FAILED
     * 表示输出的 dim0 = OTHERS, dim1 = FIRE, dim2 = OTHERS
     * OTHERS仅表示占位, 仅dim1会被输出
     *******************************************************************************/
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);

protected:
    // /***whole image preprocess with drm**/
    // virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage& tvimage ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    // /***whole image preprocess with opencv**/
    // virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    // /***image with preprocess with roi+drm**/
    // virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage& tvimage , ucloud::TvaiRect roi,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    // /***image preprocess with roi+opencv**/
    // virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi, std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,ucloud::BBox &bbox);

private:
    std::shared_ptr<BaseModel> m_net = nullptr;
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;
    
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;
    PRE_PARAM m_param_img2tensor;

    int m_InpNum = 1;
    int m_OutNum = 1;
    
    // std::map<int,ucloud::CLS_TYPE> m_output_dim_to_clss; //未来稀疏化的输出可以使用
    std::vector<ucloud::CLS_TYPE> m_clss; //稠密型, 每个输出维度都要赋予类型, 默认OTHERS是占位的

    std::vector<int> m_OutEleNums;
    std::vector<std::vector<int>> m_OutEleDims;

#ifdef TIMING
    Timer m_TK;
#endif
};


#endif