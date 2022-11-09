#ifndef _MODULE_POSENET_
#define _MODULE_POSENET_
#include <vector>
#include "module_base.hpp"
#include "framework_detection.hpp"
#include "basic.hpp"
#include <string>

/*******************************************************************************
 * PoseNet
 * chaffee.chen@2022-11-03
*******************************************************************************/
class PoseNet: public AnyModelWithBBox{
public:
    PoseNet(){};
    virtual ~PoseNet(){};
protected:
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas ,ucloud::BBox &bbox, float aX, float aY);

#ifdef TIMING
    Timer m_TK;
#endif
};


#endif