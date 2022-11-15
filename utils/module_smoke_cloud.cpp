#include "module_smoke_cloud.hpp"
#include <opencv2/opencv.hpp>

using namespace ucloud;
using namespace cv;

static inline float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

/*******************************************************************************
 * SMOKE_CLOUD_DETECTION
 * chaffee.chen@2022-11-15
 * backbone: resnet34
 * input 224x224 rgb
 * output [1,1,14,14]
*******************************************************************************/

ucloud::RET_CODE SMOKE_CLOUD_DETECTION::postprocess(std::vector<float*> &output_datas, 
                                        ucloud::TvaiRect roi, 
                                        float threshold, float nms_threshold ,
                                        ucloud::VecObjBBox &bboxes,float aX, float aY)
{
    LOGI << "-> SMOKE_CLOUD_DETECTION::postprocess";
    if(output_datas.empty()) return RET_CODE::SUCCESS;
    if(output_datas[0]==nullptr) return RET_CODE::SUCCESS;
    //[1,1,14,14]
    //dim3 dim2 dim1 dim0
    //n c h w
    int w = m_OutEleDims[0][0];
    int h = m_OutEleDims[0][1];
    if(w!=14||h!=14){
        printf("**[%s][%d] w(%d),h(%d) != 14\n", __FILE__, __LINE__, w, h);
        return RET_CODE::FAILED;
    }
    // float *ptrTmp = output_datas[0];
    // for(int i=0; i < w*h; i++){
    //     *ptrTmp = sigmoid(*ptrTmp);
    //     ptrTmp++;
    // }
    cv::Mat cv_output(h,w,CV_32FC1, output_datas[0]);
    // float minV{1},maxV{0};
    // for(int r=0; r<h; r++){
    //     float* ptr = cv_output.ptr<float>(r);
    //     for(int c=0; c<w; c++){
    //         if(ptr[c]>maxV) maxV = ptr[c];
    //         if(ptr[c]<minV) minV = ptr[c];
    //     }
    // }
    // printf("cv_output %1.3f, %1.3f\n", minV, maxV);
    cv::Mat cv_mask = cv_output > threshold;
    cv::resize(cv_mask, cv_mask, cv::Size(m_param_img2tensor.model_input_shape.w, m_param_img2tensor.model_input_shape.h));

    std::vector<std::vector<Point>> vec_cv_contours;
    findContours(cv_mask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // printf("vec_cv_contours %d\n", vec_cv_contours.size());
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.objtype = CLS_TYPE::SMOKE_CLOUD;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aX); bbox.rect.width = ((1.0*rect.width) / aX);
        bbox.rect.y = ((1.0*rect.y) / aY); bbox.rect.height = ((1.0*rect.height) / aY);
        bboxes.push_back(bbox);
    }
    LOGI << "<- SMOKE_CLOUD_DETECTION::postprocess";
    return RET_CODE::SUCCESS;
}