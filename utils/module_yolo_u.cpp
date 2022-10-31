#include "module_yolo_u.hpp"
#include <opencv2/opencv.hpp>
using namespace ucloud;
using namespace std;
// using namespace cv;

/*******************************************************************************
 * 内部函数
*******************************************************************************/
static inline int get_unique_cls_num(std::vector<CLS_TYPE>& output_clss, std::map<CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::set<CLS_TYPE> unique_cls;
    unique_cls.insert(output_clss.begin(), output_clss.end());
    int i = 0;
    for(auto &&unicls: unique_cls){
        unique_cls_order.insert(std::pair<CLS_TYPE,int>(unicls,i++));
    }
    return unique_cls.size();    
}

/*******************************************************************************
 * YOLO_DETECTION_NAIVE 无跟踪, 纯检测
 * yolov5系列支持, 需要配合特定export_rknn.py输出的结果
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
ucloud::RET_CODE YOLO_DETECTION_UINT8::set_anchor(std::vector<float> &anchors){
    m_anchors = anchors;
    if(m_anchors.size()==3*3*2 || m_anchors.size()==4*3*2)
        return RET_CODE::SUCCESS;
    else {
        printf("**Err[%s][%d] anchor size(%d) not supported\n", __FILE__, __LINE__, m_anchors.size());
        return RET_CODE::FAILED;
    }
    return RET_CODE::SUCCESS;
}


YOLO_DETECTION_UINT8::YOLO_DETECTION_UINT8(){
    LOGI << "-> YOLO_DETECTION_UINT8";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
    LOGI << "<- YOLO_DETECTION_UINT8";
}

YOLO_DETECTION_UINT8::~YOLO_DETECTION_UINT8(){
    LOGI << "-> ~YOLO_DETECTION_UINT8";
    release();
}

void YOLO_DETECTION_UINT8::release(){
    m_OutEleNums.clear();
    m_OutEleDims.clear();
}

RET_CODE YOLO_DETECTION_UINT8::init(std::map<InitParam, WeightData> &weightConfig){
    LOGI << "-> YOLO_DETECTION_UINT8::init";
    RET_CODE ret = RET_CODE::SUCCESS;
    if( weightConfig.find(InitParam::BASE_MODEL) == weightConfig.end() ){
        LOGI << "base model not found in weightConfig";
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    // m_net->release();
    bool useDRM = false;
    #ifdef TIMING    
    m_Tk.start();
    #endif
    ret = m_net->base_init(weightConfig[InitParam::BASE_MODEL].pData, weightConfig[InitParam::BASE_MODEL].size, useDRM);
    #ifdef TIMING    
    m_Tk.end("model loading");
    #endif
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO的体现, 都只取index0的数据
    if(m_InpNum != m_net->get_input_shape().size()){
        printf("**[%s][%d], m_InpNum[%d]!=m_net->get_input_shape().size()[%d]\n", __FILE__, __LINE__, m_InpNum, m_net->get_input_shape().size());
        return RET_CODE::FAILED;
    }
    if(m_OtpNum != m_net->get_output_shape().size()){
        printf("**[%s][%d], m_OtpNum[%d]!=m_net->get_output_shape().size()[%d]\n", __FILE__, __LINE__, m_OtpNum, m_net->get_output_shape().size());
        // return RET_CODE::FAILED;
    }
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //图像前处理参数
    m_param_img2tensor.keep_aspect_ratio = true;//保持长宽比, opencv有效, drm无效
    m_param_img2tensor.pad_both_side = false;//仅进行单边(右下)补齐, drm无效
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//转换成RGB格式
    m_param_img2tensor.model_input_shape = m_InpSp;//resize的需求尺寸

    m_strides = {8,16,32,64};
    LOGI << "<- YOLO_DETECTION_UINT8::init";
    return ret;
}

RET_CODE YOLO_DETECTION_UINT8::init(std::map<ucloud::InitParam, std::string> &modelpath){
    LOGI << "-> YOLO_DETECTION_UINT8::init";
    std::map<InitParam, WeightData> weightConfig;
    for(auto &&modelp: modelpath){
        int szBuf = 0;
        unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
        weightConfig[modelp.first] = WeightData{tmpBuf,szBuf};
    }
    RET_CODE ret = init(weightConfig);
    for(auto &&wC: weightConfig){
        free(wC.second.pData);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- YOLO_DETECTION_UINT8::init";
    return ret;
}


float YOLO_DETECTION_UINT8::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float YOLO_DETECTION_UINT8::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

RET_CODE YOLO_DETECTION_UINT8::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_DETECTION_UINT8::run";
    RET_CODE ret = RET_CODE::SUCCESS;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);

    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        ret = RET_CODE::SUCCESS;
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    std::vector<unsigned char*> input_datas;
    std::vector<float> aX, aY;
    // std::vector<uint8_t*> output_datas;
    // std::vector<uint32_t> output_zp;
    // std::vector<float> output_scales;
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
#ifdef USEDRM
    ret = m_cv_preprocess_net->preprocess_drm(tvimage, m_param_img2tensor, input_datas, aX, aY);
#else
    ret = m_cv_preprocess_net->preprocess_opencv(tvimage, m_param_img2tensor, input_datas, aX, aY);
#endif
#ifdef TIMING    
    m_Tk.end("preprocess");
#endif
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif
    // ret = m_net->general_infer_uint8_nhwc_to_uint8(input_datas, output_datas, output_scales, output_zp);
    ret = m_net->general_infer_uint8_nhwc_to_float(input_datas, output_datas);
#ifdef TIMING    
    m_Tk.end("general_infer_uint8_nhwc_to_float");
#endif    
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        return ret;
    }

#ifdef TIMING    
    m_Tk.start();
#endif
    // ret = postprocess(output_datas, output_scales, output_zp, threshold, nms_threshold, bboxes, aX, aY);
    ret = postprocess(output_datas, threshold, nms_threshold, bboxes, aX, aY);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif    
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        for(auto &&t: output_datas) free(t);
        return ret;
    }

    for(auto &&t: output_datas){
        free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }

    LOGI << "<- YOLO_DETECTION_UINT8::run";
    return RET_CODE::SUCCESS;
}


RET_CODE YOLO_DETECTION_UINT8::run(TvaiImage& tvimage, TvaiRect roi , VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_DETECTION_UINT8::run with roi";
    RET_CODE ret = RET_CODE::SUCCESS;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    roi = get_valid_rect(roi, tvimage.width, tvimage.height);

    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        ret = RET_CODE::SUCCESS;
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    std::vector<unsigned char*> input_datas;
    std::vector<float> aX, aY;
    // std::vector<uint8_t*> output_datas;
    // std::vector<uint32_t> output_zp;
    // std::vector<float> output_scales;
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
#ifdef USEDRM
    ret = m_cv_preprocess_net->preprocess_drm(tvimage, roi, m_param_img2tensor ,input_datas, aX, aY);
#else
    ret = m_cv_preprocess_net->preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
#endif
#ifdef TIMING    
    m_Tk.end("preprocess");
#endif
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif
    // ret = m_net->general_infer_uint8_nhwc_to_uint8(input_datas, output_datas, output_scales, output_zp);
    ret = m_net->general_infer_uint8_nhwc_to_float(input_datas, output_datas);
#ifdef TIMING    
    m_Tk.end("general_infer_uint8_nhwc_to_float");
#endif    
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        return ret;
    }

#ifdef TIMING    
    m_Tk.start();
#endif
    // ret = postprocess(output_datas, output_scales, output_zp, roi, threshold, nms_threshold, bboxes, aX, aY);
    ret = postprocess(output_datas, roi, threshold, nms_threshold, bboxes, aX, aY);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif    
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        for(auto &&t: output_datas) free(t);
        return ret;
    }

    for(auto &&t: output_datas){
        free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }

    LOGI << "<- YOLO_DETECTION_UINT8::run with roi";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE YOLO_DETECTION_UINT8::postprocess(std::vector<uint8_t*> &output_datas, 
    std::vector<float> &output_scales, 
    std::vector<uint32_t> &output_zp,
    float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION_UINT8::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    post_process_forked_rknn(output_datas, output_zp, output_scales, threshold, vecBox);
    int n = vecBox.size();
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION_UINT8::postprocess";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_DETECTION_UINT8::postprocess(std::vector<uint8_t*> &output_datas, 
    std::vector<float> &output_scales, std::vector<uint32_t> &output_zp, 
    ucloud::TvaiRect roi, 
    float threshold ,float nms_threshold,
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION_UINT8::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    post_process_forked_rknn(output_datas, output_zp, output_scales, threshold, vecBox);
    int n = vecBox.size();
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    shift_box_from_roi_to_org(vecBox_after_nms, roi);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION_UINT8::postprocess";
    return RET_CODE::SUCCESS;
}



ucloud::RET_CODE YOLO_DETECTION_UINT8::postprocess(std::vector<float*> &output_datas, 
    float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION_UINT8::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    post_process_forked_rknn(output_datas, threshold, vecBox);
    int n = vecBox.size();
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION_UINT8::postprocess";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_DETECTION_UINT8::postprocess(std::vector<float*> &output_datas, 
    ucloud::TvaiRect roi, 
    float threshold ,float nms_threshold,
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION_UINT8::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    post_process_forked_rknn(output_datas, threshold, vecBox);
    int n = vecBox.size();
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    shift_box_from_roi_to_org(vecBox_after_nms, roi);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION_UINT8::postprocess";
    return RET_CODE::SUCCESS;
}



RET_CODE YOLO_DETECTION_UINT8::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
    m_nc = output_clss.size();
    m_clss = output_clss;
    get_unique_cls_num(output_clss, m_unique_clss_map);
    return RET_CODE::SUCCESS;
}

RET_CODE YOLO_DETECTION_UINT8::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&uq: m_unique_clss_map){
        valid_clss.push_back(uq.first);
    }
    return RET_CODE::SUCCESS;
}


static float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static uint8_t qnt_f32_to_affine(float f32, uint32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

static int process_forked_rknn(uint8_t *input, float *anchor, int grid_h, int grid_w, int stride,
                   VecObjBBox &bboxes, std::vector<ucloud::CLS_TYPE> &clss,
                   float threshold, uint32_t zp, float scale, int nc)
{
    int PROP_BOX_SIZE = nc+5;
    int OBJ_CLASS_NUM = nc;
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    uint8_t thres_u8 = qnt_f32_to_affine(thres, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                uint8_t box_objectness = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_objectness >= thres_u8)
                {   
                    BBox box;
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    uint8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    box.x = box_x;
                    box.y = box_y;
                    box.w = box_w;
                    box.h = box_h;

                    box.x0 = box_x;
                    box.y0 = box_y;
                    box.x1 = box_x + box_w;
                    box.y1 = box_y + box_h;

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float max_confidence = sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale));
                    float objectness = sigmoid(deqnt_affine_to_f32(box_objectness, zp, scale));
                    box.confidence = objectness*max_confidence;
                    box.objectness = objectness;
                    box.objtype = clss[maxClassId];
                    validCount++;
                    bboxes.push_back(box);
                }
            }
        }
    }
    return validCount;
}

int YOLO_DETECTION_UINT8::post_process_forked_rknn( 
    std::vector<uint8_t*> output_datas ,std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales, 
    float conf_threshold, VecObjBBox &result)
{   
    int model_in_h = m_InpSp.h;
    int model_in_w = m_InpSp.w;
    uint8_t* anchors = output_datas[3];
    int n_anchors = 2*3*m_nl;//6*3=18
    float* anchors_fp32 = reinterpret_cast<float*>(anchors);

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    uint8_t* input0 = output_datas[0];
    validCount0 = process_forked_rknn(input0, &anchors_fp32[0] , grid_h0, grid_w0,
        stride0, result , m_clss, conf_threshold, qnt_zps[0], qnt_scales[0], m_nc);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    uint8_t* input1 = output_datas[1];
    validCount1 = process_forked_rknn(input1, &anchors_fp32[6], grid_h1, grid_w1,
        stride1, result , m_clss, conf_threshold, qnt_zps[1], qnt_scales[1], m_nc);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    uint8_t* input2 = output_datas[2];
    validCount2 = process_forked_rknn(input2, &anchors_fp32[12], grid_h2, grid_w2,
        stride2, result , m_clss, conf_threshold, qnt_zps[2], qnt_scales[2], m_nc);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    return validCount;
}



static int process_forked_rknn(float *input, float *anchor, int grid_h, int grid_w, int stride,
                   VecObjBBox &bboxes, std::vector<ucloud::CLS_TYPE> &clss,
                   float threshold, int nc)
{
    int PROP_BOX_SIZE = nc+5;
    int OBJ_CLASS_NUM = nc;
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float u_threshold = unsigmoid(threshold);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_objectness = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_objectness >= u_threshold)
                {   
                    BBox box;
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float *in_ptr = input + offset;
                    float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
                    float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
                    float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
                    float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    box.x = box_x;
                    box.y = box_y;
                    box.w = box_w;
                    box.h = box_h;

                    box.x0 = box_x;
                    box.y0 = box_y;
                    box.x1 = box_x + box_w;
                    box.y1 = box_y + box_h;

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float max_confidence = sigmoid(maxClassProbs);
                    float objectness = sigmoid(box_objectness);
                    box.confidence = objectness*max_confidence;
                    box.objectness = objectness;
                    box.objtype = clss[maxClassId];
                    validCount++;
                    bboxes.push_back(box);
                }
            }
        }
    }
    return validCount;
}



int YOLO_DETECTION_UINT8::post_process_forked_rknn( 
    std::vector<float*> output_datas, 
    float conf_threshold, VecObjBBox &result)
{   
    int model_in_h = m_InpSp.h;
    int model_in_w = m_InpSp.w;
    // float* anchors = output_datas[3];
    // float anchors[] = { 5.81641, 4.39062,  10.78906,  8.45312, 18.59375, 14.20312, 34.31250, 23.07812, 24.43750,  58.09375,  86.62500,  57.87500,  67.00000, 167.50000, 185.12500, 153.87500, 410.00000, 465.50000};
    int n_anchors = 2*3*m_nl;//6*3=18
    int validCount = 0;

    float *anchors = &(m_anchors[0]);
    // memcpy(anchors, &(m_anchors[0]), n_anchors*sizeof(float)); 
    for(int i = 0; i < output_datas.size(); i++){
        // stride 8
        int stride0 = m_strides[i];
        int grid_h0 = model_in_h / stride0;
        int grid_w0 = model_in_w / stride0;
        float* input0 = output_datas[i];
        validCount += process_forked_rknn(input0, &anchors[6*i] , grid_h0, grid_w0,
            stride0, result , m_clss, conf_threshold, m_nc);
    }
    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    return validCount;
}

