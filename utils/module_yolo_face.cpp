#include "module_yolo_face.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
using namespace ucloud;
using namespace std;
// using namespace cv;


static inline float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static inline float unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

/*******************************************************************************
 * 内部函数
*******************************************************************************/
static inline int get_unique_cls_num(std::vector<CLS_TYPE>& output_clss, std::map<CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::vector<CLS_TYPE> unique_cls;
    for(auto i=output_clss.begin(); i !=output_clss.end(); i++){
        bool conflict = false;
        for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
            if( *i == *iter ){
                conflict = true;
                break;
            }
        }
        if(!conflict) unique_cls.push_back(*i);
    }
    for(int i=0; i < unique_cls.size(); i++ ){
        unique_cls_order.insert(std::pair<CLS_TYPE,int>(unique_cls[i],i));
    }
    return unique_cls.size();
}

/*******************************************************************************
 * YOLO_FACE_DETECTION_NAIVE 无跟踪, 纯检测
 * yolov5系列支持, 需要配合特定export_rknn.py输出的结果
 * chaffee.chen@ucloud.cn 2022-10-12
*******************************************************************************/
YOLO_FACE_DETECTION_NAIVE::YOLO_FACE_DETECTION_NAIVE(){
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE";
}

YOLO_FACE_DETECTION_NAIVE::~YOLO_FACE_DETECTION_NAIVE(){
    LOGI << "-> ~YOLO_FACE_DETECTION_NAIVE";
    release();
}

void YOLO_FACE_DETECTION_NAIVE::release(){
    m_OutEleNums.clear();
    m_OutEleDims.clear();
    m_anchors.clear();
}

RET_CODE YOLO_FACE_DETECTION_NAIVE::init(std::map<InitParam, WeightData> &weightConfig){
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::init";
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
    // m_decode_mode = 0;
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //图像前处理参数
    m_param_img2tensor.keep_aspect_ratio = true;//保持长宽比, opencv有效, drm无效
    m_param_img2tensor.pad_both_side = false;//仅进行单边(右下)补齐, drm无效
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//转换成RGB格式
    m_param_img2tensor.model_input_shape = m_InpSp;//resize的需求尺寸

    m_strides = {8,16,32,64};
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::init";
    return ret;
}

RET_CODE YOLO_FACE_DETECTION_NAIVE::init(std::map<ucloud::InitParam, std::string> &modelpath){
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::init";
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
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::init";
    return ret;
}

ucloud::RET_CODE YOLO_FACE_DETECTION_NAIVE::set_anchor(std::vector<float> &anchors){
    LOGI << "->  YOLO_FACE_DETECTION_NAIVE::set_anchor";
    m_anchors = anchors;
    if(m_anchors.size()==3*3*2 || m_anchors.size()==4*3*2){
        set_decode(4);
        LOGI << "m_decode_mode = " << m_decode_mode;
        return RET_CODE::SUCCESS;
    }
    else {
        printf("**Err[%s][%d] anchor size(%d) not supported\n", __FILE__, __LINE__, m_anchors.size());
        return RET_CODE::FAILED;
    }
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE YOLO_FACE_DETECTION_NAIVE::set_decode(int mode){
    m_decode_mode = mode;
    return RET_CODE::SUCCESS;
}


float YOLO_FACE_DETECTION_NAIVE::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float YOLO_FACE_DETECTION_NAIVE::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

RET_CODE YOLO_FACE_DETECTION_NAIVE::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::run";
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
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
    if(tvimage.width%8!=0 || tvimage.height%2!=0)
        ret = m_cv_preprocess_net->preprocess_opencv(tvimage, m_param_img2tensor, input_datas, aX, aY);
    else
        ret = m_cv_preprocess_net->preprocess_drm(tvimage, m_param_img2tensor, input_datas, aX, aY);
// #ifdef USEDRM
//     ret = m_cv_preprocess_net->preprocess_drm(tvimage, m_param_img2tensor, input_datas, aX, aY);
// #else
//     ret = m_cv_preprocess_net->preprocess_opencv(tvimage, m_param_img2tensor, input_datas, aX, aY);
// #endif
#ifdef TIMING    
    m_Tk.end("preprocess");
#endif
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif
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
    ret = postprocess(output_datas, threshold, nms_threshold, bboxes, aX, aY, m_decode_mode);
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

    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::run";
    return RET_CODE::SUCCESS;
}


RET_CODE YOLO_FACE_DETECTION_NAIVE::run(TvaiImage& tvimage, TvaiRect roi , VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::run with roi";
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
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
// #ifdef USEDRM
//     ret = m_cv_preprocess_net->preprocess_drm(tvimage, roi, m_param_img2tensor ,input_datas, aX, aY);
// #else
//     ret = m_cv_preprocess_net->preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
// #endif
    if(tvimage.width%8!=0 || tvimage.height%2!=0)
        ret = m_cv_preprocess_net->preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);
    else
        ret = m_cv_preprocess_net->preprocess_drm(tvimage, roi,  m_param_img2tensor, input_datas, aX, aY);

#ifdef TIMING    
    m_Tk.end("preprocess");
#endif
    if(ret!=RET_CODE::SUCCESS) return ret;

#ifdef TIMING    
    m_Tk.start();
#endif
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
    ret = postprocess(output_datas, roi ,threshold, nms_threshold, bboxes, aX, aY, m_decode_mode);
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

    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::run with roi";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_FACE_DETECTION_NAIVE::postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY, int decode_mode)
{
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    RET_CODE retcode = RET_CODE::SUCCESS;

    retcode = rknn_output_to_boxes_mode4(output_datas, threshold, vecBox);

    if(retcode!=RET_CODE::SUCCESS){
        printf("**ERR[%s][%d] postprocess decode err\n", __FILE__, __LINE__);
        return retcode;
    }

    int n = 0;
    for(auto &&box: vecBox){
        n+=box.size();
    }
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::postprocess";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_FACE_DETECTION_NAIVE::postprocess(std::vector<float*> &output_datas, TvaiRect roi, float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY, int decode_mode)
{
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    RET_CODE retcode = RET_CODE::SUCCESS;

    retcode = rknn_output_to_boxes_mode4(output_datas, threshold, vecBox);

    if(retcode!=RET_CODE::SUCCESS){
        printf("**ERR[%s][%d] postprocess decode err\n", __FILE__, __LINE__);
        return retcode;
    }
    
    int n = 0;
    for(auto &&box: vecBox){
        n+=box.size();
    }
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aX[0], aY[0]);
    shift_box_from_roi_to_org(vecBox_after_nms, roi);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::postprocess";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_FACE_DETECTION_NAIVE::rknn_output_to_boxes_mode4( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes){
    LOGI << "-> YOLO_FACE_DETECTION_NAIVE::rknn_output_to_boxes_mode4";
    if(m_anchors.empty()) {
        printf("**ERR [%s][%d] m_anchors should be set\n", __FILE__, __LINE__);
        return RET_CODE::FAILED;
    }
    if(output_datas.size()!=3){
        printf("**[%s][%d] output_datas.size = %d\n",__FILE__, __LINE__, output_datas.size());
        return RET_CODE::FAILED;
    }
    std::vector<std::vector<int>> output_dims = m_net->get_output_dims();
    int NC = m_nc;
    int layers_num = output_datas.size();
    int anchors_num = 3;
    int input_h = m_param_img2tensor.model_input_shape.h;
    int input_w = m_param_img2tensor.model_input_shape.w;
    int prior_box_size = 5+NC+5*2;//xywh(4),objectness(1), xy,xy,...(5*2) ,num_classes(NC), 
    int anchor_len = m_anchors.size()/layers_num;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }
    float thres_sigmoid = unsigmoid(threshold);

    for (int layer_id=0;layer_id<layers_num;layer_id++){
        float *layer = output_datas[layer_id];
        int grid_h = input_h/m_strides[layer_id];
        int grid_w = input_w/m_strides[layer_id];
        int grid_len = grid_h*grid_w;
        int stride = m_strides[layer_id];
        float* anchor= &(m_anchors[anchor_len*layer_id]);

        for (int a =0;a<anchors_num;a++){          
            for (int i=0;i<grid_h;i++){
                for (int j=0;j<grid_w;j++){
                    float box_objectness = layer[(prior_box_size * a + 4) * grid_len + i * grid_w + j];
                   
                    if (box_objectness >= thres_sigmoid){
                        BBox fbox; 
                        int offset = (prior_box_size * a) * grid_len + i * grid_w + j;
                        float *in_ptr = layer + offset;
                        float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
                        float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
                        float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
                        float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;

                        fbox.Pts.type = LandMarkType::LICPLATE;
                        fbox.Pts.refcoord = RefCoord::IMAGE_ORIGIN;
                        for(int npt = 0; npt < 5; npt++){
                            float ptx = in_ptr[ (5+ 2*npt) * grid_len] * (float)anchor[a * 2] + j*(float)stride;
                            float pty = in_ptr[ (5+ 2*npt+1) * grid_len] * (float)anchor[a * 2 + 1] + i*(float)stride;
                            fbox.Pts.pts.push_back(uPoint(ptx, pty));
                        }

                        box_x = (box_x + j) * (float)stride;
                        box_y = (box_y + i) * (float)stride;
                        box_w = box_w * box_w * (float)anchor[a * 2];
                        box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);
                        
                        fbox.x = box_x;
                        fbox.y = box_y;
                        fbox.w = box_w;
                        fbox.h = box_h;

                        fbox.x0 = box_x;
                        fbox.y0 = box_y;
                        fbox.x1 = box_x + box_w;
                        fbox.y1 = box_y + box_h;

                        float maxClassProbs = in_ptr[15 * grid_len];
                        int maxid = 0; 
                        for (int k = 1; k < NC; ++k)
                        {
                            float prob = in_ptr[(15 + k) * grid_len];
                            if (prob > maxClassProbs)
                            {
                                maxid = k;
                                maxClassProbs = prob;
                            }
                        }
                        float max_confidence = sigmoid(maxClassProbs);
                        float objectness = sigmoid(box_objectness);
                        fbox.objectness = objectness;
                        fbox.confidence = objectness*max_confidence;;
                        fbox.quality = objectness;
                        if (maxid < 0 || m_clss.empty())
                            fbox.objtype = CLS_TYPE::UNKNOWN;
                        else
                            fbox.objtype = m_clss[maxid];
                        if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                            bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);

                    }
                }
                // fd<<"\n";
            }
        }   
        // fd.close();
    }
    LOGI << "<- YOLO_FACE_DETECTION_NAIVE::rknn_output_to_boxes_mode4";
    return RET_CODE::SUCCESS;
}



RET_CODE YOLO_FACE_DETECTION_NAIVE::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
    m_nc = output_clss.size();
    m_clss = output_clss;
    get_unique_cls_num(output_clss, m_unique_clss_map);
    return RET_CODE::SUCCESS;
}

RET_CODE YOLO_FACE_DETECTION_NAIVE::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&m: m_clss){
        bool FLAG_exsit_class = false;
        for( auto &&n:valid_clss){
            if( m==n){
                FLAG_exsit_class = true;
                break;
            }
        }
        if(!FLAG_exsit_class) valid_clss.push_back(m);
    }
    return RET_CODE::SUCCESS;
}



