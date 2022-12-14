#include "module_retinaface.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace ucloud;
using namespace std;


static void base_transform_xyxy_xyhw_face(std::vector<BBox> &vecbox, float expand_ratio ,float aX, float aY){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aX);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aY);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aX;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aY;

        float wh = MAX(w,h);

        float _x0 = cx - wh/2;
        float _y0 = cy - wh/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(wh);
        vecbox[i].rect.height = int(wh);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aX;
            vecbox[i].Pts.pts[j].y /= aY;
        }
    }
};

/**
 * RETINAFACE_DETECTION
 * chaffee.chen@ucloud.cn 2022-10-11
 */
RETINAFACE_DETECTION::RETINAFACE_DETECTION(){
    LOGI << "-> RETINAFACE_DETECTION";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
}

RETINAFACE_DETECTION::~RETINAFACE_DETECTION(){
    LOGI << "-> RETINAFACE_DETECTION";
    release();
}

void RETINAFACE_DETECTION::release(){
    m_OutEleNums.clear();
    m_OutEleDims.clear();
    if(m_Anchors) free(m_Anchors);
    m_Anchors = nullptr;
}

void RETINAFACE_DETECTION::gen_prior_box(){
    
    int min_sizes[] = {16, 32, 64, 128, 256, 512};
    int steps[] = {8, 16, 32};
    int feature_maps[6] = {0};
    float H = m_InpSp.h;
    float W = m_InpSp.w;
    int total_ele = 0;
    for(int i = 0; i < 3 ; i++){
        feature_maps[2*i] = std::ceil(H/steps[i]);
        feature_maps[2*i+1] = std::ceil(W/steps[i]);
        printf("feature_map size [%d,%d]\n", feature_maps[2*i] , feature_maps[2*i+1]);
        total_ele += 2 * feature_maps[2*i]*feature_maps[2*i+1];//2* each layer has 2 min size
    }
    LOGI << "** total x4 anchors " << total_ele;
    if(m_Anchors) free(m_Anchors);
    m_Anchors = (float*)malloc(sizeof(float)*total_ele*4);
    int ind = 0;
    for(int l=0;l<3; l++){//layer of feature maps
        int f0 = feature_maps[2*l];
        int f1 = feature_maps[2*l+1];
        for(int i = 0; i < f0; i++ ){//for i, j in product(range(f[0]), range(f[1]))
            for(int j = 0; j < f1; j++){
                for(int m = 0; m < 2; m++ ){//for min_size in min_sizes:
                    float min_size = min_sizes[2*l+m];
                    float s_kx = min_size/W;//s_kx = min_size / self.image_size[1]
                    float s_ky = min_size/H;//s_ky = min_size / self.image_size[0]
                    float dense_cx = ((float)((j+0.5)*steps[l])) / W; //dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    float dense_cy = ((float)((i+0.5)*steps[l])) / H; //dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    m_Anchors[ind++] = dense_cx;
                    m_Anchors[ind++] = dense_cy;
                    m_Anchors[ind++] = s_kx;
                    m_Anchors[ind++] = s_ky;
                }
            }
        }//for i, j in product(range(f[0]), range(f[1]))
    }//layer of feature maps
    // for(int i = 0; i < 10; i++){
    //     printf("%f,", m_Anchors[i]);
    // }
    // printf("\n");

}

RET_CODE RETINAFACE_DETECTION::init(std::map<ucloud::InitParam, std::string> &modelpath){
    LOGI << "-> RETINAFACE_DETECTION::init";
    RET_CODE ret = RET_CODE::SUCCESS;
    if( modelpath.find(InitParam::BASE_MODEL) == modelpath.end() ){
        LOGI << "base model not found in modelpath";
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    // m_net->release();
    bool useDRM = false;
    #ifdef TIMING    
    m_Tk.start();
    #endif
    ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], useDRM);
    #ifdef TIMING    
    m_Tk.end("model loading");
    #endif
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO?????????, ?????????index0?????????
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //?????????????????????
    m_param_img2tensor.keep_aspect_ratio = true;//???????????????, opencv??????, drm??????
    m_param_img2tensor.pad_both_side = false;//???????????????(??????)??????, drm??????
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::BGR;//?????????RGB??????
    m_param_img2tensor.model_input_shape = m_InpSp;//resize???????????????

    // m_strides = {8,16,32,64};
    // if(!check_output_dims_1LX()){
    //     printf("output dims check failed\n");
    //     return RET_CODE::ERR_MODEL_NOT_MATCH;
    // }
    gen_prior_box();//m_Anchors
    LOGI << "<- RETINAFACE_DETECTION::init";
    return ret;
}


RET_CODE RETINAFACE_DETECTION::init(std::map<InitParam, ucloud::WeightData> &weightConfig){
    LOGI << "-> RETINAFACE_DETECTION::init";
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
    ret = m_net->base_init(weightConfig[InitParam::BASE_MODEL].pData, weightConfig[InitParam::BASE_MODEL].size ,useDRM);
    #ifdef TIMING    
    m_Tk.end("model loading");
    #endif
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO?????????, ?????????index0?????????
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //?????????????????????
    m_param_img2tensor.keep_aspect_ratio = true;//???????????????, opencv??????, drm??????
    m_param_img2tensor.pad_both_side = false;//???????????????(??????)??????, drm??????
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::BGR;//?????????RGB??????
    m_param_img2tensor.model_input_shape = m_InpSp;//resize???????????????

    // m_strides = {8,16,32,64};
    // if(!check_output_dims_1LX()){
    //     printf("output dims check failed\n");
    //     return RET_CODE::ERR_MODEL_NOT_MATCH;
    // }
    gen_prior_box();//m_Anchors
    LOGI << "<- RETINAFACE_DETECTION::init";
    return ret;
}


float RETINAFACE_DETECTION::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float RETINAFACE_DETECTION::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

RET_CODE RETINAFACE_DETECTION::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> RETINAFACE_DETECTION::run";
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
    

#ifdef TIMING    
    m_Tk.end("preprocess");
#endif
    if(ret!=RET_CODE::SUCCESS) return ret;

    // return ret;

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

    LOGI << "<- RETINAFACE_DETECTION::run";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE RETINAFACE_DETECTION::postprocess(std::vector<float*> &output_datas, float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> RETINAFACE_DETECTION::postprocess_drm";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    rknn_output_to_boxes_1LX(output_datas, threshold, vecBox);
    int n = vecBox.size();
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox, nms_threshold , NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw_face(vecBox_after_nms, m_expand_ratio, aX[0], aY[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- RETINAFACE_DETECTION::postprocess_drm";
    return RET_CODE::SUCCESS;
}

RET_CODE RETINAFACE_DETECTION::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = {m_clss};
    return RET_CODE::SUCCESS;
}

/**
 * ??????Tensor?????????:
 * boxes[1,L,4] scores[1,L,1] landmarks[1,L,10]
 * dim0 = 4,1,10
 * dim1 = L
 * dim2 = 1
 **/  
ucloud::RET_CODE RETINAFACE_DETECTION::rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, ucloud::VecObjBBox &bboxes){
    LOGI << "<- RETINAFACE_DETECTION::rknn_output_to_boxes_1LX";
    int stepboxes = 4;
    int steplmks = 10;
    int stepanchors = 4;
    int L = m_OutEleDims[0][1];
    float imgW = m_InpSp.w;
    float imgH = m_InpSp.h;

    int cnt = 0;
    float *ptrBoxes = output_datas[0];
    float *ptrScores = output_datas[1];
    float *ptrLmks = output_datas[2];
    float *ptrAnchors = m_Anchors;
    for(int i = 0; i < L; i++){
        float objectness = *ptrScores++;
        // printf("objectness:%f\n", objectness);
        if( objectness<threshold ) {
            ptrBoxes += stepboxes;
            ptrLmks += steplmks;
            ptrAnchors += stepanchors;
            continue;
        } else{ //threshold
            cnt++;
            BBox fbox;
            float cx = *ptrBoxes++;
            float cy = *ptrBoxes++;
            float cw = *ptrBoxes++; 
            float ch = *ptrBoxes++;

            float p0 = *ptrAnchors++;
            float p1 = *ptrAnchors++;
            float p2 = *ptrAnchors++;
            float p3 = *ptrAnchors++;
            /* 
            bx1 = priors[:,:, :2] + loc[:,:, :2] * variances[0] * priors[:,:, 2:]
            loc[:,:, :2] = [cx, cy]
            bx1 ~ [cx, cy]
            priors[:,:, :2] = [p0,p1]
            */
            cx = p0 + cx*m_Var[0]*p2;
            cy = p1 + cy*m_Var[0]*p3;
            /* 
            bx2 = priors[:,:, 2:] * torch.exp(loc[:,:, 2:] * variances[1])
            loc[:,:, 2:] = [cw,ch]
            priors[:,:, 2:] = [p2,p3]
            */
            cw = p2* std::exp(cw*m_Var[1]);
            ch = p3* std::exp(ch*m_Var[1]);
            /* 
            boxes = boxes * self.scale#[B,N,4]
            */
            cx *= imgW;
            cw *= imgW;
            cy *= imgH;
            ch *= imgH;
            fbox.x0 = cx - cw/2;
            fbox.y0 = cy - ch/2;
            fbox.x1 = cx + cw/2;
            fbox.y1 = cy + ch/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = cw; fbox.h = ch;
            /* 
            landms = torch.cat((priors[:,:, :2] + pre[:,:, :2] * variances[0] * priors[:,:, 2:],
                priors[:,:, :2] + pre[:,:, 2:4] * variances[0] * priors[:,:, 2:],
                priors[:,:, :2] + pre[:,:, 4:6] * variances[0] * priors[:,:, 2:],
                priors[:,:, :2] + pre[:,:, 6:8] * variances[0] * priors[:,:, 2:],
                priors[:,:, :2] + pre[:,:, 8:10] * variances[0] * priors[:,:, 2:],
                ), dim=2)
            */                        
            for(int nlmk=0; nlmk<5; nlmk++){
                float x = *ptrLmks++;
                float y = *ptrLmks++;
                x = p0 + x*m_Var[0]*p2;
                y = p1 + y*m_Var[0]*p3;
                x *= imgW;
                y *= imgH;
                fbox.Pts.pts.push_back(uPoint(x,y));
            }
            fbox.Pts.type = LandMarkType::FACE_5PTS;
            fbox.Pts.refcoord = RefCoord::IMAGE_ORIGIN;
            fbox.objectness = objectness;
            fbox.confidence = objectness;
            fbox.quality = objectness;//++quality using max_confidence instead for object detection
            fbox.objtype = m_clss;
            bboxes.push_back(fbox);
        } //threshold
    }
    // printf("[%d] over threshold: %f\n",cnt, m_threshold);

    LOGI << "-> RETINAFACE_DETECTION::rknn_output_to_boxes_1LX";
    return RET_CODE::SUCCESS;
}



