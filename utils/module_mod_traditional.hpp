#ifndef _MODULE_MOD_TRADITIONAL_HPP_
#define _MODULE_MOD_TRADITIONAL_HPP_
#include <vector>
#include "module_base.hpp"
#include "basic.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include "module_nn_match.hpp"

/*******************************************************************************
 * BACKGROUND_SEGMENTATION
 * 通用分类器
 * chaffee.chen@2022-11-10
*******************************************************************************/
class BACKGROUND_SEGMENTATION: public ucloud::AlgoAPI{
public:
    BACKGROUND_SEGMENTATION(){};
    virtual ~BACKGROUND_SEGMENTATION(){};
    virtual ucloud::RET_CODE init();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam,std::string> &modelpath);
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
    /*******************************************************************************
     * get_class_type 返回剔除占位类型OTHERS后的有效分类类别
    *******************************************************************************/    
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);

protected:
    ucloud::RET_CODE create_model(int uuid_cam);//BG Substraction
    ucloud::RET_CODE create_trackor(int uuid_cam);//tracklets
    virtual ucloud::RET_CODE postprocess(cv::Mat cvInp, ucloud::VecObjBBox &bboxes, float aX, float aY, 
        float threshold, int uuid_cam);
    virtual ucloud::RET_CODE postfilter(ucloud::TvaiImage &tvimage, ucloud::VecObjBBox &ins, ucloud::VecObjBBox &outs, float threshold);
    virtual ucloud::RET_CODE trackprocess(ucloud::TvaiImage &tvimage, ucloud::VecObjBBox &ins);

private:
    std::shared_ptr<PreProcess_CPU_DRM_Model> m_cv_preprocess_net = nullptr;        
    std::vector<ucloud::CLS_TYPE> m_clss = {ucloud::CLS_TYPE::FALLING_OBJ, ucloud::CLS_TYPE::FALLING_OBJ_UNCERTAIN};
    PRE_PARAM m_param_img2tensor;

#ifdef OPENCV3
    std::map<int,cv::Ptr<cv::BackgroundSubtractor>> m_Models;
#else
    std::map<int,cv::shared_ptr<cv::BackgroundSubtractor>> m_Models;
#endif
    std::map<int,std::shared_ptr<BoxTraceSet>> m_Trackors;

#ifdef TIMING
    Timer m_TK;
#endif
};


#endif