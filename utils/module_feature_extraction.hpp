#ifndef _MODULE_FEATURE_EXTRACTION_
#define _MODULE_FEATURE_EXTRACTION_
#include <vector>
#include "module_base.hpp"
#include "basic.hpp"
#include <string>

/*******************************************************************************
 * FeatureExtractor
 * 通用特征提取器, 注意VecObjBBox中含有指针, 需要通过AICoreFactory::releaseVecObjBBox进行释放
 * chaffee.chen@2022-11-03
*******************************************************************************/
class FeatureExtractor: public ucloud::AlgoAPI{
public:
    FeatureExtractor();
    virtual ~FeatureExtractor();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    /*******************************************************************************
     * run 对bboxes中的每个区域进行特征提取, 并将结果更新到bboxes中(objtype,tvaifeature)
    *******************************************************************************/
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas ,ucloud::BBox &bbox);

    void unit_norm(float* ptr, int dims);

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