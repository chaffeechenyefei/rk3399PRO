#ifndef _MODULE_SMOKE_CLOUD_HPP_
#define _MODULE_SMOKE_CLOUD_HPP_

#include "framework_detection.hpp"


/*******************************************************************************
 * SMOKE_CLOUD_DETECTION
 * chaffee.chen@2022-11-15
 * backbone: resnet34
 * input 224x224 rgb
 * output [1,1,14,14]
*******************************************************************************/
class SMOKE_CLOUD_DETECTION: public AnyModelWithTvaiImage{
public:
    SMOKE_CLOUD_DETECTION(){}
    virtual ~SMOKE_CLOUD_DETECTION(){}
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas, 
                                        ucloud::TvaiRect roi, 
                                        float threshold, float nms_threshold ,
                                        ucloud::VecObjBBox &bboxes,float aX, float aY);

};


#endif