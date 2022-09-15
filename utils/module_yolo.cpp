#include "module_yolo.hpp"
#include <opencv2/opencv.hpp>
using namespace ucloud;
using namespace std;
// using namespace cv;


YOLO_DETECTION::YOLO_DETECTION(){
    LOGI << "-> YOLO_DETECTION";
    m_net = std::make_shared<BaseModel>();
}

YOLO_DETECTION::~YOLO_DETECTION(){
    LOGI << "-> YOLO_DETECTION";
    release();
}

void YOLO_DETECTION::release(){
    m_OutEleNums.clear();
    m_OutEleDims.clear();
}

RET_CODE YOLO_DETECTION::init(std::map<ucloud::InitParam, std::string> &modelpath){
    LOGI << "-> YOLO_DETECTION::init";
    RET_CODE ret = RET_CODE::SUCCESS;
    if( modelpath.find(InitParam::BASE_MODEL) == modelpath.end() ){
        LOGI << "base model not found in modelpath";
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    // m_net->release();
    bool useDRM = false;
    ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], useDRM);
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO的体现, 都只取index0的数据
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //图像前处理参数
    m_param_img2tensor.keep_aspect_ratio = true;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;
    m_param_img2tensor.model_input_shape = m_InpSp;

    m_strides = {8,16,32,64};
    if(!check_output_dims()){
        printf("output dims check failed\n");
        return RET_CODE::ERR_MODEL_NOT_MATCH;
    }
    LOGI << "<- NaiveModel::init";
    return ret;
}

/**
 * 输出Tensor的维度:
 * [1 na h w d ] flatten -> [1, na*h*w*d ] dim0: na*h*w*d dim1:1
 * [1 nl na 2] flatten -> [1, nl*na*2] dim0: nl*na*2 dim1:1
 **/
bool YOLO_DETECTION::check_output_dims(){
    int NA = 3;
    int D = m_nc+5;
    for(int i = 0; i < m_OtpNum - 1 ; i++ ){//minus 1,因为最后一个是grid_anchors
        int dim1_expected = m_InpSp.h / m_strides[i] * m_InpSp.w / m_strides[i] * NA * D;
        if (dim1_expected != m_OutEleDims[i][0] ){
            printf("m_InpSp.h / m_strides[i] = %d, m_InpSp.w / m_strides[i] = %d, \
                    NA = %d, D = %d, \
                    m_OutEleDims[i][0] = %d not equals\n", \
                    m_InpSp.h / m_strides[i], m_InpSp.w / m_strides[i], NA, D , m_OutEleDims[i][0] \
                    );
            return false;
        }
    }
    return true;
}

RET_CODE YOLO_DETECTION::run(TvaiImage& tvimage, VecObjBBox &bboxes){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_DETECTION::run";
    RET_CODE ret = RET_CODE::SUCCESS;

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
    std::vector<float> aspect_ratios;
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
    ret = preprocess(tvimage, input_datas, aspect_ratios);
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
    ret = postprocess(output_datas, bboxes, aspect_ratios);
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

    LOGI << "<- YOLO_DETECTION::run";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_DETECTION::preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aspect_ratio  ){
    LOGI << "-> YOLO_DETECTION::preprocess";
    // Mat im(tvimage.height,tvimage.width,CV_8UC3, tvimage.pData);
    // Mat resized_im;
    // cv::resize(im, resized_im, Size(m_InpSp.w,m_InpSp.h));
    // unsigned char *tmp = (unsigned char *)malloc(resized_im.cols*resized_im.rows*3);
    // memcpy(tmp, resized_im.data, resized_im.cols*resized_im.rows*3);
    // input_datas.push_back(tmp);
    std::vector<cv::Mat> dst;
    std::vector<float> aX,aY;
    std::vector<cv::Rect> roi = {cv::Rect(0,0,tvimage.width,tvimage.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, roi, 
        dst, m_param_img2tensor, aX, aY);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        cv::imwrite("preprocess_img.png", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        // memset(data,255,ele.total()*3);//ATT.
        for(int i = 0; i < ele.total()*3; i++)
            data[i] = 255;
        input_datas.push_back(data);
    }
    aspect_ratio = aX;
    LOGI << "<- YOLO_DETECTION::preprocess";
    return ret;
}

ucloud::RET_CODE YOLO_DETECTION::postprocess(std::vector<float*> &output_datas, VecObjBBox &bboxes, std::vector<float> &aspect_ratios){
    LOGI << "-> YOLO_DETECTION::postprocess";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    // rknn_output_to_boxes_c_data_layer(output_datas, vecBox);
    rknn_output_to_boxes_python_data_layer(output_datas, vecBox);
    int n = 0;
    for(auto &&box: vecBox){
        n+=box.size();
    }
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox,m_nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratios[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION::postprocess";
    return RET_CODE::SUCCESS;
}


/**
 * [1 na h w d ]
 * [ 1 3:nl 3:na 2:(xy) ]
 */
ucloud::RET_CODE YOLO_DETECTION::rknn_output_to_boxes_python_data_layer( std::vector<float*> &output_datas,std::vector<ucloud::VecObjBBox> &bboxes){
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_python_data_layer";
    int NC = m_nc;
    int NL = m_nl;//3
    int NA = 3;
    int D = NC + 5;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }

    float* anchor_grid = output_datas[NL];
    for(int nl = 0; nl < NL; nl++){
        for(int na = 0; na < NA; na++ ){
            printf("%f,%f ", anchor_grid[nl*NA*2+na*2+0] , anchor_grid[nl*NA*2+na*2+1]);
        }
        printf("\n");
    }
    printf("\n");
    int cnt = 0;
    for(int nl = 0; nl < NL; nl++){
        //layers
        float *layer = output_datas[nl];
        int H = m_InpSp.h/m_strides[nl];
        int W = m_InpSp.w/m_strides[nl];
        float* pp = layer;
        printf("layer[%d] :", nl );
        for(int p = 0; p < 10; p++ ){
            printf("%f, ", *pp++);
        }
        printf("\n");
        //num of anchors: na
        for(int na = 0; na < NA; na++){
            for(int h = 0; h < H; h++){//feature map h
                for(int w=0; w < W; w++){ //w
                    // float *tmp = layer+nl*NA*H*W*D+na*H*W*D+h*W*D+w*D;
                    float objectness = *(layer+4);
                    // printf("objectness:%f\n", objectness);
                    if( objectness<m_threshold ) {
                        layer += D;
                        continue;
                    } else{ //threshold
                        cnt++;
                        BBox fbox;
                        float *tmp = layer;
                        float cx = *tmp++;
                        float cy = *tmp++;
                        float cw = *tmp++; 
                        float ch = *tmp++;
                        cx = (cx*2 - 0.5 + w )*m_strides[nl];
                        cy = (cy*2 - 0.5 + h )*m_strides[nl];
                        cw = (cw*cw*4)* anchor_grid[nl*NA*2+na*2+0];   //anchor_grid[nl*NA*2+na*2+0];
                        ch = (ch*ch*4)* anchor_grid[nl*NA*2+na*2+1];   //anchor_grid[nl*NA*2+na*2+1];
                        fbox.x0 = cx - cw/2;
                        fbox.y0 = cy - ch/2;
                        fbox.x1 = cx + cw/2;
                        fbox.y1 = cy + ch/2;
                        fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
                        tmp++;//skip objectness
                        fbox.objectness = objectness;
                        int maxid = -1;
                        float max_confidence = 0;
                        float* confidence = tmp;
                        argmax(confidence, NC , maxid, max_confidence);
                        fbox.confidence = objectness*max_confidence;
                        fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
                        layer += D;
                        if (maxid < 0 || m_clss.empty())
                            fbox.objtype = CLS_TYPE::UNKNOWN;
                        else
                            fbox.objtype = m_clss[maxid];
                        if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                            bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);
                    } //threshold
                } //w
            }//feature map h
        }//num of anchors: na
        printf("[%d] over threshold: %d\n",cnt, m_threshold);
    }
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_python_data_layer";
    return RET_CODE::SUCCESS;
}

/**
 * anchor_grid 
 * python data layer [ 1 3:nl 3:na 2:(xy) ]
 * c data layer [(xy), na , nl ]
 */
static inline float get_anchor_grid_from_c_data_layer(float* anchor_grid, int nl, int na , int xy ,int NA, int NL){
    return anchor_grid[ xy*NA*NL+na*NL+nl ];
}
/**
 * output layer [1 na h w d ] -> [ d w h na 1] -> [d w h na] -> dim0:d dim1:w dim2:h dim3:na 
 * python data layer [1 na h w d ]
 * c data layer [d w h na]
 */
static inline float get_xywhoc_from_c_data_layer(float *output, int na, int h, int w, int d, int NA, int H, int W){
    return output[ d*W*H*NA + w*H*NA + h*NA + na];
}
ucloud::RET_CODE YOLO_DETECTION::rknn_output_to_boxes_c_data_layer( std::vector<float*> &output_datas, std::vector<ucloud::VecObjBBox> &bboxes){
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_c_data_layer";
    int NC = m_clss.size();
    int NL = m_nl;//3
    int NA = 3;
    int D = NC + 5;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }
    float* confidence = (float*)malloc(sizeof(float)*NC);
    float* anchor_grid = output_datas[NL];
    // printf("anchor_grid:");
    // for(int nl = 0; nl < NL; nl++){
    //     printf("[layer:%d] = ", nl);
    //     for(int na = 0; na < NA; na++){
    //         printf("%f,%f ", \
    //             anchor_grid[nl*NA*2+na*2], anchor_grid[nl*NA*2+na*2+1]);
    //         // get_anchor_grid_from_c_data_layer(anchor_grid, nl, na , 0 , NA, NL), \
    //         // get_anchor_grid_from_c_data_layer(anchor_grid, nl, na , 1 , NA, NL));
    //     }
    //     printf("\n");
    // }

    for(int nl = 0; nl < NL; nl++){
        //layers
        float *layer = output_datas[nl];
        int H = m_InpSp.h/m_strides[nl];
        int W = m_InpSp.w/m_strides[nl];
        //num of anchors: na
        for(int na = 0; na < NA; na++){
            for(int h = 0; h < H; h++){//feature map h
                for(int w=0; w < W; w++){ //w
                    float objectness = get_xywhoc_from_c_data_layer(layer,na,h,w,4,NA,H,W);
                    // printf("objectness:%f\n", objectness);
                    if( objectness<m_threshold ) {
                        continue;
                    } else{ //threshold
                        BBox fbox;
                        float cx = get_xywhoc_from_c_data_layer(layer,na,h,w,0,NA,H,W);
                        float cy = get_xywhoc_from_c_data_layer(layer,na,h,w,1,NA,H,W);
                        float cw =  get_xywhoc_from_c_data_layer(layer,na,h,w,2,NA,H,W);
                        float ch =  get_xywhoc_from_c_data_layer(layer,na,h,w,3,NA,H,W);
                        cx = (cx*2 - 0.5 + w )*m_strides[nl];
                        cy = (cy*2 - 0.5 + h )*m_strides[nl];
                        cw = (cw*cw*4)* anchor_grid[nl*NA*2+na*2+0];//get_anchor_grid_from_c_data_layer(anchor_grid, nl, na , 0 , NA, NL);   //anchor_grid[nl*NL*NA*2+na*NA*2+0];
                        ch = (ch*ch*4)* anchor_grid[nl*NA*2+na*2+1];//get_anchor_grid_from_c_data_layer(anchor_grid, nl, na , 1 , NA, NL);   //anchor_grid[nl*NL*NA*2+na*NA*2+1];
                        fbox.x0 = cx - cw/2;
                        fbox.y0 = cy - ch/2;
                        fbox.x1 = cx + cw/2;
                        fbox.y1 = cy + ch/2;
                        fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
                        // tmp++;//skip objectness
                        fbox.objectness = objectness;
                        int maxid = -1;
                        float max_confidence = 0;
                        for(int i=0; i < NC; i++){
                            confidence[i] = get_xywhoc_from_c_data_layer(layer,na,h,w,i+5,NA,H,W);
                        }
                        argmax(confidence, NC , maxid, max_confidence);
                        fbox.confidence = objectness*max_confidence;
                        fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
                        // layer += D;
                        if (maxid < 0 || m_clss.empty())
                            fbox.objtype = CLS_TYPE::UNKNOWN;
                        else
                            fbox.objtype = m_clss[maxid];
                        if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                            bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);
                    } //threshold
                } //w
            }//feature map h
        }//num of anchors: na
    }
    free(confidence);
    LOGI << "-> YOLO_DETECTION::rknn_output_to_boxes_c_data_layer";
    return RET_CODE::SUCCESS;
}


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

RET_CODE YOLO_DETECTION::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
    m_nc = output_clss.size();
    m_clss = output_clss;
    get_unique_cls_num(output_clss, m_unique_clss_map);
    return RET_CODE::SUCCESS;
}

RET_CODE YOLO_DETECTION::set_param(float threshold, float nms_threshold){
    if(float_in_range(threshold,1,0))
        m_threshold = threshold;
    else
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    if(float_in_range(nms_threshold,1,0))
        m_nms_threshold = nms_threshold;
    else
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    return RET_CODE::SUCCESS;    
}

RET_CODE YOLO_DETECTION::get_class_type(std::vector<CLS_TYPE> &valid_clss){
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