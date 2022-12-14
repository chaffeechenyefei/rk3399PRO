#ifndef _MODULE_LPRNET_HPP_
#define _MODULE_LPRNET_HPP_
#include <vector>
#include "module_base.hpp"
#include "basic.hpp"
#include <string>

/*******************************************************************************
 * LPRNET
*******************************************************************************/
static std::vector<std::string> LICPLATE_CHARS = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                        "新",
                        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                        "W", "X", "Y", "Z", "I", "O", "-"};


class LPRNET: public ucloud::AlgoAPI{
public:
    LPRNET();
    virtual ~LPRNET();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    /*******************************************************************************
     * run 对bboxes中的每个区域进行分类, 并将结果更新到bboxes中(objtype, objectness, confidence)
     * PARAM:
     *  threshold只有超过阈值的分类结果才会更新到bboxes中
    *******************************************************************************/
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);

    /*******************************************************************************
     * trival
    *******************************************************************************/    
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss);

protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, float threshold ,ucloud::BBox &bbox);

private:
    std::shared_ptr<BaseModel> m_net = nullptr;
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;
    
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;
    PRE_PARAM m_param_img2tensor;

    int m_InpNum = 1;
    int m_OutNum = 1;
    
    std::vector<int> m_OutEleNums;
    std::vector<std::vector<int>> m_OutEleDims;

#ifdef TIMING
    Timer m_TK;
#endif
};


#endif