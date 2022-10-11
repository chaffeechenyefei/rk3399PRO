#include "module_yolo.hpp"
#include <opencv2/opencv.hpp>
using namespace ucloud;
using namespace std;
// using namespace cv;


YOLO_DETECTION::YOLO_DETECTION(){
    LOGI << "-> YOLO_DETECTION";
    m_net = std::make_shared<BaseModel>();
    m_drm = std::make_shared<ImageUtil>();
    m_track_param = {0.5, 0.5+0.1f};
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
    #ifdef TIMING    
    m_Tk.start();
    #endif
    ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], useDRM);
    #ifdef TIMING    
    m_Tk.end("model loading");
    #endif
    if(ret!=RET_CODE::SUCCESS) return ret;
    //SISO的体现, 都只取index0的数据
    assert(m_InpNum == m_net->get_input_shape().size());
    assert(m_OtpNum == m_net->get_output_shape().size());
    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    //图像前处理参数
    m_param_img2tensor.keep_aspect_ratio = true;//保持长宽比, opencv有效, drm无效
    m_param_img2tensor.pad_both_side = false;//仅进行单边(右下)补齐, drm无效
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//转换成RGB格式
    m_param_img2tensor.model_input_shape = m_InpSp;//resize的需求尺寸

    m_strides = {8,16,32,64};
    // if(!check_output_dims_1LX()){
    //     printf("output dims check failed\n");
    //     return RET_CODE::ERR_MODEL_NOT_MATCH;
    // }
    
    m_track = std::make_shared<ByteTrackNoReIDPool>(m_fps,m_nn_buf);
    // m_track = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
    LOGI << "<- NaiveModel::init";
    return ret;
}

/**
 * 输出Tensor的维度:
 * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1]
 * dim0 = 2,NC+1
 * dim1 = L
 * dim2 = 1
 **/  
bool YOLO_DETECTION::check_output_dims_1LX(){
    int L = -1;
    if(m_OtpNum != 3){
        printf("m_OtpNum(%d) is not 3(cxcy,wh,conf)\n",m_OtpNum);
        return false;
    }
    if(m_OutEleDims[2][0]!=m_nc+1){
        printf( "NC(%d)+1 is not m_OutEleDims[2][0](%d)\n", m_nc, m_OutEleDims[2][0]);
        return false;
    }
    for(int i =0 ; i < m_OtpNum; i++){
        if(L==-1) L = m_OutEleDims[i][1];
        if(L!=m_OutEleDims[i][1]){
            printf("L = %d, m_OutEleDims[%d][1] = %d not matched\n",L, i , m_OutEleDims[i][1]);
            return false;
        }
    }
    return true;
}

float YOLO_DETECTION::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float YOLO_DETECTION::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

RET_CODE YOLO_DETECTION::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    // return run_drm(tvimage, bboxes);
    LOGI << "-> YOLO_DETECTION::run";
    RET_CODE ret = RET_CODE::SUCCESS;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    m_track_param = {threshold, threshold+0.1f};

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
    std::vector<float> aspect_ratios, aX, aY;
    std::vector<float*> output_datas;

#ifdef TIMING    
    m_Tk.start();
#endif
    // ret = preprocess(tvimage, input_datas, aspect_ratios);
    ret = preprocess_drm(tvimage, input_datas, aX, aY);
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
    // ret = postprocess(output_datas, threshold, nms_threshold, bboxes, aspect_ratios);
    ret = postprocess_drm(output_datas, threshold, nms_threshold, bboxes, aX, aY);
#ifdef TIMING    
    m_Tk.end("postprocess");
#endif    
    if(ret!=RET_CODE::SUCCESS) {
        for(auto &&t: input_datas) free(t);
        for(auto &&t: output_datas) free(t);
        return ret;
    }

#ifdef TIMING    
    m_Tk.start();
#endif
    if(m_track){
        m_track->update(tvimage, bboxes, m_track_param);
        m_track->clear();
    }
        

#ifdef TIMING    
    m_Tk.end("tracking");
#endif  
    

    for(auto &&t: output_datas){
        free(t);
    }
    for(auto &&t: input_datas){
        free(t);
    }

    LOGI << "<- YOLO_DETECTION::run";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE YOLO_DETECTION::preprocess_opencv(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, std::vector<float> &aspect_ratio  ){
    LOGI << "-> YOLO_DETECTION::preprocess_opencv";
    // Mat im(tvimage.height,tvimage.width,CV_8UC3, tvimage.pData);
    // Mat resized_im;
    // cv::resize(im, resized_im, Size(m_InpSp.w,m_InpSp.h));
    // unsigned char *tmp = (unsigned char *)malloc(resized_im.cols*resized_im.rows*3);
    // memcpy(tmp, resized_im.data, resized_im.cols*resized_im.rows*3);
    // input_datas.push_back(tmp);
    bool use_subpixel = false;
    std::vector<cv::Mat> dst;
    std::vector<float> aX,aY;
    std::vector<cv::Rect> roi = {cv::Rect(0,0,tvimage.width,tvimage.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, roi, 
        dst, m_param_img2tensor, aX, aY, use_subpixel);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        // cv::imwrite("preprocess_img.png", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        // memset(data,255,ele.total()*3);//ATT.
        // for(int i = 0; i < ele.total()*3; i++) //check the output of net to ensure result on PC and Emb equals.
        //     data[i] = 255;
        input_datas.push_back(data);
    }
    aspect_ratio = aX;
    LOGI << "<- YOLO_DETECTION::preprocess_opencv";
    return ret;
}

ucloud::RET_CODE YOLO_DETECTION::preprocess_drm(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas, 
    std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION::preprocess_drm";
    bool valid_input_format = true;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        break;
    default:
        valid_input_format = false;
        break;
    }
    if(!valid_input_format) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;

    unsigned char* data = (unsigned char*)std::malloc(3*m_InpSp.w*m_InpSp.w);
    RET_CODE uret = m_drm->init(tvimage);
    if(uret!=RET_CODE::SUCCESS) return uret;
    // int ret = m_drm->resize(tvimage,m_InpSp, data);
    int ret = m_drm->resize(tvimage, m_param_img2tensor, data);
    input_datas.push_back(data);
    aX.push_back( (float(m_InpSp.w))/tvimage.width );
    aY.push_back( (float(m_InpSp.h))/tvimage.height );

    // cv::Mat cvimage_show( cv::Size(m_InpSp.w, m_InpSp.h), CV_8UC3, data);
    // cv::cvtColor(cvimage_show, cvimage_show, cv::COLOR_RGB2BGR);
    // cv::imwrite("preprocess_drm.jpg", cvimage_show);

    LOGI << "<- YOLO_DETECTION::preprocess_drm";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE YOLO_DETECTION::postprocess_drm(std::vector<float*> &output_datas, float threshold ,float nms_threshold, 
    ucloud::VecObjBBox &bboxes, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> YOLO_DETECTION::postprocess_drm";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    rknn_output_to_boxes_1LX(output_datas, threshold, vecBox);
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
    LOGI << "<- YOLO_DETECTION::postprocess_drm";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE YOLO_DETECTION::postprocess_opencv(std::vector<float*> &output_datas, float threshold ,float nms_threshold, 
    VecObjBBox &bboxes, std::vector<float> &aspect_ratios)
{
    LOGI << "-> YOLO_DETECTION::postprocess_opencv";
    if(output_datas.empty()) return RET_CODE::ERR_POST_EXE;

    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    // rknn_output_to_boxes_c_data_layer(output_datas, vecBox);
    // rknn_output_to_boxes_python_data_layer(output_datas, vecBox);
    rknn_output_to_boxes_1LX(output_datas, threshold, vecBox);
    int n = 0;
    for(auto &&box: vecBox){
        n+=box.size();
    }
    LOGI << "rknn_output_to_boxes " << n;
    base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratios[0]);
    bboxes = vecBox_after_nms;
    // LOGI << "after filter " << bboxes.size() << std::endl;
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YOLO_DETECTION::postprocess_opencv";
    return RET_CODE::SUCCESS;
}

/** mode=2: Detect Layer仅进行了sigmoid, 且不进行permute
 * 模型输出Tensor的维度:
 * wywhs [1,na*no,h,w]xnl(3) anchors_grid [nl,na,2]
 * dim0 = w
 * dim1 = h
 * dim2 = na*no
 **/
ucloud::RET_CODE YOLO_DETECTION::rknn_output_to_boxes_1LX3( std::vector<float*> &output_datas, float threshold, 
    std::vector<ucloud::VecObjBBox> &bboxes)
{
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_1LX3";
    int NL = m_nl;
    int NA = m_OutEleDims[3][1];
    int NC = m_nc;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }

    int cnt = 0;
    float *ptrAnchorGrid = output_datas[3];

    for(int nl=0; nl < NL; nl++){ // Layer
        float* ptrXYWHS = output_datas[nl]; //[1,na,no,h,w]
        for(int na=0; na < NA; na++){ //num of anchors
            int H = m_InpSp.h / m_strides[nl];
            int W = m_InpSp.w / m_strides[nl];
            int hwStep = H*W;
            float agX = ptrAnchorGrid[ nl*NA*2 + na*2 + 0];
            float agY = ptrAnchorGrid[ nl*NA*2 + na*2 + 1];
            float* ptrNoHW = ptrXYWHS+na*(NC+5)*hwStep; //[no,h,w]
            float* ptrX = ptrNoHW;
            float* ptrY = ptrNoHW + hwStep;
            float* ptrW = ptrNoHW + 2*hwStep;
            float* ptrH = ptrNoHW + 3*hwStep;
            float* ptrObj = ptrNoHW + 4*hwStep; //[h,w] objectness
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    float objectness = *ptrObj++;
                    if(objectness<threshold){
                        ptrX++; ptrY++; ptrW++; ptrH++;
                        continue;
                    } else {
                        cnt++;
                        BBox fbox;
                        float cx = *ptrX++;
                        float cy = *ptrY++;
                        float cw = *ptrW++; 
                        float ch = *ptrH++;
                        cx = (cx*2 - 0.5 + w)*m_strides[nl];
                        cy = (cy*2 - 0.5 + h)*m_strides[nl];
                        cw = cw*cw*4*agX;
                        ch = ch*ch*4*agY;
                        fbox.x0 = cx - cw/2;
                        fbox.y0 = cy - ch/2;
                        fbox.x1 = cx + cw/2;
                        fbox.y1 = cy + ch/2;
                        fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = cw; fbox.h = ch;
                        fbox.objectness = objectness;
                        int maxid = -1;
                        float max_confidence = 0;
                        float* confidence = (float*)malloc(sizeof(float)*NC);
                        for(int i = 0; i < NC; i++){
                            confidence[i] = *(ptrNoHW+(i+5)*hwStep);
                        }
                        argmax(confidence, NC , maxid, max_confidence);
                        free(confidence);
                        fbox.confidence = objectness*max_confidence;
                        fbox.quality = max_confidence;//++quality using max_confidence instead for object detection

                        if (maxid < 0 || m_clss.empty())
                            fbox.objtype = CLS_TYPE::UNKNOWN;
                        else
                            fbox.objtype = m_clss[maxid];
                        if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                            bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);
                    }//if objectness < m_threshold
                }//w
            }//h
        }//num of anchors

    }// Layer
    LOGI << "-> YOLO_DETECTION::rknn_output_to_boxes_1LX3";
    return RET_CODE::SUCCESS;
}  

/** mode=1: Detect Layer仅进行了sigmoid
 * 输入Tensor的维度:
 * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1] anchors_grid [nl,na,2]
 * dim0 = 2,NC+1
 * dim1 = L
 * dim2 = 1
 **/  
ucloud::RET_CODE YOLO_DETECTION::rknn_output_to_boxes_1LX2( std::vector<float*> &output_datas, float threshold, 
    std::vector<ucloud::VecObjBBox> &bboxes)
{
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_1LX2";
    int stepxywh = 2;
    int stepConf = m_nc + 1;
    int L = m_OutEleDims[0][1];
    int NL = m_nl;
    int NA = m_OutEleDims[3][1];
    int NC = m_nc;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }

    int cnt = 0;
    float *ptrXY = output_datas[0];//center x,y
    float *ptrWH = output_datas[1];
    float *ptrProb = output_datas[2];//objectness+prob of classes
    float *ptrAnchorGrid = output_datas[3];

    for(int nl=0; nl < NL; nl++){ // Layer
        for(int na=0; na < NA; na++){ //num of anchors
            int H = m_InpSp.h / m_strides[nl];
            int W = m_InpSp.w / m_strides[nl];
            float agX = ptrAnchorGrid[ nl*NA*2 + na*2 + 0];
            float agY = ptrAnchorGrid[ nl*NA*2 + na*2 + 1];
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    float objectness = *ptrProb;
                    if(objectness< threshold){
                        ptrXY += stepxywh;
                        ptrWH += stepxywh;
                        ptrProb += stepConf;
                        continue;
                    } else {
                        cnt++;
                        BBox fbox;
                        float cx = *ptrXY++;
                        float cy = *ptrXY++;
                        float cw = *ptrWH++; 
                        float ch = *ptrWH++;
                        cx = (cx*2 - 0.5 + w)*m_strides[nl];
                        cy = (cy*2 - 0.5 + h)*m_strides[nl];
                        cw = cw*cw*4*agX;
                        ch = ch*ch*4*agY;
                        fbox.x0 = cx - cw/2;
                        fbox.y0 = cy - ch/2;
                        fbox.x1 = cx + cw/2;
                        fbox.y1 = cy + ch/2;
                        fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = cw; fbox.h = ch;
                        fbox.objectness = objectness;
                        int maxid = -1;
                        float max_confidence = 0;
                        float* confidence = ptrProb+1;
                        argmax(confidence, NC , maxid, max_confidence);
                        fbox.confidence = objectness*max_confidence;
                        fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
                        ptrProb += stepConf;
                        if (maxid < 0 || m_clss.empty())
                            fbox.objtype = CLS_TYPE::UNKNOWN;
                        else
                            fbox.objtype = m_clss[maxid];
                        if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                            bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);
                    }//if objectness < m_threshold
                }//w
            }//h
        }//num of anchors

    }// Layer
    LOGI << "-> YOLO_DETECTION::rknn_output_to_boxes_1LX2";
    return RET_CODE::SUCCESS;
}

/**
 * 输入Tensor的维度:
 * xy[1,L,2] wh[1,L,2] conf[1,L,NC+1]
 * dim0 = 2,NC+1
 * dim1 = L
 * dim2 = 1
 **/  
ucloud::RET_CODE YOLO_DETECTION::rknn_output_to_boxes_1LX( std::vector<float*> &output_datas, float threshold, std::vector<ucloud::VecObjBBox> &bboxes){
    LOGI << "<- YOLO_DETECTION::rknn_output_to_boxes_1LX";
    int stepxywh = 2;
    int stepConf = m_nc + 1;
    int L = m_OutEleDims[0][1];
    int NC = m_nc;
    cout<<"NC "<<NC<<endl;
    for (int i=0; i<m_unique_clss_map.size(); i++){
        bboxes.push_back(VecObjBBox());
    }

    int cnt = 0;
    float *ptrXY = output_datas[0];//center x,y
    float *ptrWH = output_datas[1];
    float *ptrProb = output_datas[2];//objectness+prob of classes
    for(int i = 0; i < L; i++){
        float objectness = *ptrProb;
        // printf("objectness:%f\n", objectness);
        if( objectness<threshold ) {
            ptrXY += stepxywh;
            ptrWH += stepxywh;
            ptrProb += stepConf;
            continue;
        } else{ //threshold
            cnt++;
            BBox fbox;
            float cx = *ptrXY++;
            float cy = *ptrXY++;
            float cw = *ptrWH++; 
            float ch = *ptrWH++;
            fbox.x0 = cx - cw/2;
            fbox.y0 = cy - ch/2;
            fbox.x1 = cx + cw/2;
            fbox.y1 = cy + ch/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = cw; fbox.h = ch;
            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = ptrProb+1;
            argmax(confidence, NC , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            ptrProb += stepConf;
            if (maxid < 0 || m_clss.empty())
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = m_clss[maxid];
            if(m_unique_clss_map.find(fbox.objtype)!=m_unique_clss_map.end())
                bboxes[m_unique_clss_map[fbox.objtype]].push_back(fbox);
        } //threshold
    }
    // printf("[%d] over threshold: %f\n",cnt, m_threshold);

    LOGI << "-> YOLO_DETECTION::rknn_output_to_boxes_1LX";
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

// RET_CODE YOLO_DETECTION::set_param(float threshold, float nms_threshold){
//     if(float_in_range(threshold,1,0))
//         m_threshold = threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     if(float_in_range(nms_threshold,1,0))
//         m_nms_threshold = nms_threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return RET_CODE::SUCCESS;    
// }

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






