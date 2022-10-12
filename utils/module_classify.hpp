#ifndef _MODULE_CLASSIFY_HPP_
#define _MODULE_CLASSIFY_HPP_
#include <vector>
#include "module_base.hpp"
#include "basic.hpp"


class Classification: public ucloud::AlgoAPI{
public:
    Classification();
    virtual ~Classification();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
    virtual ucloud::RET_CODE set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss,int select_idx=1);
    virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage &tvimage,std::vector<unsigned char*> &input_datas);
    virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage &tvimage,std::vector<unsigned char*> &input_datas,std::vector<cv::Rect> &rois);
private:
    std::shared_ptr<BaseModel> m_net = nullptr;
    std::shared_ptr<ImageUtil> m_drm = nullptr;
    
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;
    PRE_PARAM m_param_img2tensor;
    std::vector<ucloud::CLS_TYPE> m_clss;
    int m_InpNum = 1;
    int m_OutNum = 1;
    int m_nc = 0;
    int m_select = 1;//选择的特征为度
    std::vector<int> m_OutEleNums;
    std::vector<std::vector<int>> m_OutEleDims;
    std::map<ucloud::CLS_TYPE, int> m_unique_clss_map;

#ifdef TIMING
    Timer m_TK;
#endif
};


#endif