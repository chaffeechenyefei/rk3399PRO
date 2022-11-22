#ifndef _MODULE_ABANDON_OBJECT_HPP_
#define _MODULE_ABANDON_OBJECT_HPP_

#include "framework_detection.hpp"
#include <queue>
#include <opencv2/opencv.hpp>
#include  "module_nn_match.hpp"

class ABANDON_OBJECT_DETECTION:public AnyModelWithTvaiImage{
public:
    ABANDON_OBJECT_DETECTION(){};
    virtual ~ABANDON_OBJECT_DETECTION(){};
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.5);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi ,ucloud::VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    virtual ucloud::RET_CODE get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss);
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, ucloud::TvaiRect roi, float threshold, float nms_threshold ,ucloud::VecObjBBox &bboxes,float aX, float aY);
    virtual ucloud::RET_CODE trackprocess(ucloud::TvaiImage &tvimage, ucloud::VecObjBBox &ins,bool same);
    ucloud::RET_CODE updateBG(const Mat cur); 
    // ucloud::RET_CODE calc_diff(const Mat cur,Mat &diff);
    ucloud::RET_CODE create_trackor(int uuid_cam);//tracklets
    // ucloud::RET_CODE setBG
private:
    std::vector<ucloud::CLS_TYPE> m_clss = {ucloud::CLS_TYPE::ABADNON_STATIC, ucloud::CLS_TYPE::ABADNON_MOVE,ucloud::CLS_TYPE::OTHERS};
    std::queue<cv::Mat> m_batchBg;
    cv::Mat m_bg; 
    int m_Bg_num=5;
    std::map<int,std::shared_ptr<BoxTraceSet> > m_Trackors;
    std::vector<float> m_Bg_weight{0.1,0.15,0.2,0.25,0.3};

};

#endif 
