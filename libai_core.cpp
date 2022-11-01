#include "libai_core.hpp"
#include <opencv2/opencv.hpp>
#include "utils/basic.hpp"
#include "utils/module_siso.hpp"
#include "utils/module_yolo.hpp"
#include "utils/module_classify.hpp"
#include "utils/module_phone.hpp"
#include "utils/module_retinaface.hpp"
#include "utils/module_smoke_cig_detection.hpp"
#include "utils/module_fire_detection.hpp"
#include "utils/module_yolo_u.hpp"
#include <iostream>

using namespace cv;
using namespace ucloud;
using std::cout;
using std::endl;
using std::vector;
using cv::Scalar;
using cv::Size;


std::vector<float> default_anchors = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
// std::vector<float> anchors = { 5.81641,   4.39062,  10.78906,   8.45312,  18.59375,  14.20312,  34.31250,  23.07812,  24.43750,  58.09375,  86.62500,  57.87500,  67.00000, 167.50000, 185.12500, 153.87500, 410.00000, 465.50000};





/*--------------AICoreFactory API------------------*/
AICoreFactory::AICoreFactory(){LOGI << "AICoreFactory Constructor";}
AICoreFactory::~AICoreFactory(){}

AlgoAPISPtr AICoreFactory::getAlgoAPI(AlgoAPIName apiName){
#ifdef BUILD_VERSION
    printf("\033[31m------------------------LIBAI_CORE-------------------------------\033[0m\n");
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
    printf("\033[31mcurrent version of libai_core.so is \"%s\"\033[0m\n", BUILD_VERSION);
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
#endif     
    AlgoAPISPtr apiHandle = nullptr;
    switch (apiName)
    {
/*******************************************************************************
内部测试
*******************************************************************************/
    case AlgoAPIName::UDF_JSON:
    {
        YOLO_DETECTION_NAIVE* _ptr_ = new YOLO_DETECTION_NAIVE();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
        _ptr_->set_output_cls_order(model_output_clss);
        apiHandle.reset(_ptr_);
    }
        break;
    case AlgoAPIName::RESERVED1://classification单独测试
    {
        Classification* _ptr_ = new Classification();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::OTHERS, CLS_TYPE::PHONING, CLS_TYPE::OTHERS};
        _ptr_->set_output_cls_order(model_output_clss);
        apiHandle.reset(_ptr_);
    }
        break;    
    case AlgoAPIName::GENERAL_DETECTOR_FAST_LOAD:
    {
        YOLO_DETECTION_UINT8 *_ptr_ = new YOLO_DETECTION_UINT8();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
        _ptr_->set_output_cls_order(model_output_clss);
        std::vector<float> anchors = { 5.81641,   4.39062,  10.78906,   8.45312,  18.59375,  14.20312,  34.31250,  23.07812,  24.43750,  58.09375,  86.62500,  57.87500,  67.00000, 167.50000, 185.12500, 153.87500, 410.00000, 465.50000};
        _ptr_->set_anchor(anchors);
        apiHandle.reset(_ptr_);
    }
        break;
/*******************************************************************************
正式接口
*******************************************************************************/           
    /**
     * 人脸检测
     */
    case AlgoAPIName::FACE_DETECTOR:
    {
        printf("AlgoAPIName::FACE_DETECTOR\n");
        RETINAFACE_DETECTION_BYTETRACK* _ptr_ = new RETINAFACE_DETECTION_BYTETRACK();
        apiHandle.reset(_ptr_);
    }
        break;            
    /**
     * 人车非通用检测 
     */
    case AlgoAPIName::GENERAL_DETECTOR:
    {
        printf("AlgoAPIName::GENERAL_DETECTOR\n");
        YOLO_DETECTION_BYTETRACK* _ptr_ = new YOLO_DETECTION_BYTETRACK();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
        _ptr_->set_output_cls_order(model_output_clss);
        std::vector<float> anchors = { 5.81641,   4.39062,  10.78906,   8.45312,  18.59375,  14.20312,  34.31250,  23.07812,  24.43750,  58.09375,  86.62500,  57.87500,  67.00000, 167.50000, 185.12500, 153.87500, 410.00000, 465.50000};
        _ptr_->set_anchor(anchors);
        apiHandle.reset(_ptr_);
    }
        break;
    /**
     * 安全帽检测
     */
    case AlgoAPIName::SAFETY_HAT_DETECTOR:
    {
        printf("AlgoAPIName::SAFETY_HAT_DETECTOR\n");
        YOLO_DETECTION_BYTETRACK* _ptr_ = new YOLO_DETECTION_BYTETRACK();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::PED_SAFETY_HAT, CLS_TYPE::PED_HEAD};
        _ptr_->set_anchor(default_anchors);
        _ptr_->set_output_cls_order(model_output_clss);
        apiHandle.reset(_ptr_);
    }
        break;        
    /**
     * 火焰检测
     */
    case AlgoAPIName::FIRE_DETECTOR:
    {
        printf("AlgoAPIName::FIRE_DETECTOR\n");
        YOLO_DETECTION_BYTETRACK* _ptr_ = new YOLO_DETECTION_BYTETRACK();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::FIRE};
        _ptr_->set_output_cls_order(model_output_clss);
        _ptr_->set_anchor(default_anchors);
        apiHandle.reset(_ptr_);
    }
        break;        
    case AlgoAPIName::FIRE_DETECTOR_X://火焰检测加强版
    {
        printf("AlgoAPIName::FIRE_DETECTOR_X\n");
        FireDetector* _ptr_ = new FireDetector();
        // vector<CLS_TYPE> model_output_clss = {CLS_TYPE::FIRE};
        // _ptr_->set_output_cls_order(model_output_clss);
        apiHandle.reset(_ptr_);
    }
        break;              
    /*
    * 打电话
    */
    case AlgoAPIName::PHONING_DETECTOR:{
        printf("AlgoAPIName::PHONING_DETECTOR\n");
        PhoneDetector* _ptr_ = new PhoneDetector();
        //LS_TYPE::OTHERS,CLS_TYPE::PHONING,CLS_TYPE::PHONE_PLAY
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::OTHERS, CLS_TYPE::PHONING, CLS_TYPE::OTHERS};
        _ptr_->set_output_cls_order(model_output_clss);
        apiHandle.reset(_ptr_); 
    }
    break;   
    /*
    * 抽烟
    */
    case AlgoAPIName::SMOKING_DETECTOR:{
        printf("AlgoAPIName::SMOKING_DETECTOR\n");
        SMOKE_CIG_DETECTION* _ptr_ = new SMOKE_CIG_DETECTION();
        apiHandle.reset(_ptr_); 
    }
    break;    
    /*
    * 香烟检测
    */
    case AlgoAPIName::CIG_DETECTOR_NO_TRACK:{
        printf("AlgoAPIName::CIG_DETECTOR_NO_TRACK\n");
        YOLO_DETECTION_NAIVE* _ptr_ = new YOLO_DETECTION_NAIVE();
        vector<CLS_TYPE> cls_types = {CLS_TYPE::SMOKING};
        _ptr_->set_output_cls_order(cls_types);
        _ptr_->set_anchor(default_anchors);
        apiHandle.reset(_ptr_); 
    }
    break;           

    default:
        std::cout << "ERROR: Current API is not ready yet!" << std::endl;
        break;
    }
    return apiHandle;
}

/*--------------Read/Write API------------------*/
unsigned char* ucloud::readImg_to_RGB(std::string filepath, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_BGR(std::string filepath, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_RGB(std::string filepath, int w, int h,int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    float ar = 1.0;
    im = resize(im,Size(w,h), false, ar );         
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_BGR(std::string filepath, int w, int h, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    float ar = 1.0;
    im = resize(im,Size(w,h), false, ar );        
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_NV21(std::string filepath, int &width, int &height, int &stride){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv21_with_stride(im, width, height, stride, 2);
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_NV12(std::string filepath, int &width, int &height, int &stride){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv12_with_stride(im, width, height, stride, 2);
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_NV21(std::string filepath, int w, int h,int &width, int &height, int &stride){
    Mat im = imread(filepath);
    if(im.empty()){
        printf("%s not found\n", filepath.c_str());
        return nullptr;
    }
    float ar = 1.0;
    im = resize(im,Size(w,h), false, ar );
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv21_with_stride(im, width, height, stride, 2);
    return dst_ptr;    
}

unsigned char* ucloud::readImg_to_NV12(std::string filepath, int w, int h,int &width, int &height, int &stride){
    Mat im = imread(filepath);
    float ar = 1.0;
    im = resize(im,Size(w,h), false, ar );
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv12_with_stride(im, width, height, stride, 2);
    return dst_ptr;    
}

void ucloud::writeImg(std::string filepath , unsigned char* img, int width, int height, bool overwrite){
    static int image_cnt = 0;
    std::string _filepath = filepath;
    if (!overwrite){
        while(exists_file(_filepath)){
            _filepath = filepath + "_" + std::to_string(image_cnt) + ".jpg";
            image_cnt++;
        }
    }
    Mat im(height, width, CV_8UC3, img);
    imwrite(_filepath, im);
}

void ucloud::freeImg(unsigned char** imgPtr){
    free(*imgPtr);
    *imgPtr = nullptr;
}

static int global_rand_color[256] = {0};
static bool global_rand_color_init = false;
static int global_landmark_color[] = {//bgr::5
    255,0,0,
    0,255,0,
    0,0,255,
    255,255,0,
    0,255,255,
};
void ucloud::drawImg(unsigned char* img, int width, int height, VecObjBBox &bboxs, 
    bool disp_landmark, bool disp_label, bool use_rand_color, int color_for_trackid_or_cls){
    int thickness = (width/640) + 2;
    //第一次载入, 初始化全局颜色
    if(!global_rand_color_init){
        for(int i = 0; i < sizeof(global_rand_color)/sizeof(int); i++ ){
            global_rand_color[i] = rand()%255;
        }
        global_rand_color_init = true;
    }
    if(!bboxs.empty()){
        int* rand_color = (int*)malloc(bboxs.size()*3*sizeof(int));
        if(use_rand_color){
            for(int i = 0; i < bboxs.size()*3; i++ )
                rand_color[i] = rand()%255;
        }
        Mat im(height,width,CV_8UC3, img);
        for(int i = 0; i < bboxs.size(); i++){
            Scalar color = use_rand_color ? Scalar(rand_color[i*3],rand_color[i*3+1],rand_color[i*3+2]): Scalar(0,255,0) ;
            if(bboxs[i].track_id >=0 && color_for_trackid_or_cls == 0){
                int gi = bboxs[i].track_id%(sizeof(global_rand_color)/sizeof(int)/3);
                color = Scalar(global_rand_color[gi*3],global_rand_color[gi*3+1],global_rand_color[gi*3+2]);
                // std::cout << gi << "," << global_rand_color[gi*3] << std::endl;
            } else if(bboxs[i].track_id >=0 && color_for_trackid_or_cls == 1){
                int gi = int(bboxs[i].objtype)%(sizeof(global_rand_color)/sizeof(int)/3);
                color = Scalar(global_rand_color[gi*3],global_rand_color[gi*3+1],global_rand_color[gi*3+2]);
                // std::cout << gi << "," << global_rand_color[gi*3] << std::endl;
            } else if(color_for_trackid_or_cls == 1 ){
                color = Scalar(0,255,0);
            }

            TvaiRect _rect = bboxs[i].rect;
            rectangle(im, Rect(_rect.x,_rect.y,_rect.width,_rect.height), color,thickness);
            if (disp_label){
                std::string track_id = "";
                if(bboxs[i].track_id >= 0 )
                    track_id = ": " + std::to_string(bboxs[i].track_id);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << bboxs[i].confidence;
                std::string score = " ," + stream.str();
                putText(im, std::to_string(bboxs[i].objtype)+track_id+score, Point(_rect.x, _rect.y+25), FONT_ITALIC, 0.8, color, thickness-1);
            }
                
            if (disp_landmark){
                for(int j = 0; j < bboxs[i].Pts.pts.size(); j++){
                    int gj = j%5;
                    cv::Scalar lmk_color = Scalar(global_landmark_color[3*gj],global_landmark_color[3*gj+1],global_landmark_color[3*gj+2]);
                    circle(im, Point2f(bboxs[i].Pts.pts[j].x, bboxs[i].Pts.pts[j].y),3, lmk_color,2);
                }
            }
        }
        free(rand_color);
    }
}


unsigned char* ucloud::yuv_reader(std::string filename, int w, int h, bool trans2bgr){
    std::ifstream fin(filename, std::ios::binary);
    int l = fin.tellg();
    fin.seekg(0, std::ios::end);
    int m = fin.tellg();
    fin.seekg(0,std::ios::beg);
    // cout << "file size " << (m-l) << " bytes" << endl;
    assert(m-l == w*h/2*3);
    int stride = w;
    int wh = w*h;
    unsigned char* yuvdata = (unsigned char*)malloc(int(wh/2*3)*sizeof(unsigned char));
    fin.read( reinterpret_cast<char*>(yuvdata) , int(wh/2*3)*sizeof(unsigned char));
    fin.close();
    if(!trans2bgr)
        return yuvdata;
    else{
        unsigned char* bgrdata = (unsigned char*)malloc(int(wh*3)*sizeof(unsigned char));
        cv::Mat img(h/2*3,w,CV_8UC1,yuvdata);
        cv::Mat bgr(h,w,CV_8UC3, bgrdata);
        cv::cvtColor(img, bgr, CV_YUV2BGR_NV21);
        free(yuvdata);
        return bgrdata;
    }
}

unsigned char* ucloud::rgb_reader(std::string filename, int w, int h){
    std::ifstream fin(filename, std::ios::binary);
    int l = fin.tellg();
    fin.seekg(0, std::ios::end);
    int m = fin.tellg();
    fin.seekg(0,std::ios::beg);
    assert(m-l == w*h*3);
    int stride = w;
    int wh = w*h;
    unsigned char* rgbdata = (unsigned char*)malloc(int(wh*3)*sizeof(unsigned char));
    fin.read( reinterpret_cast<char*>(rgbdata) , int(wh*3)*sizeof(unsigned char));
    fin.close();
    return rgbdata;
}

/*--------------视频读写------------------*/
//视频读取基于opencv
void vidReader::release(){
    if(handle_t!=nullptr){
        VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
        vid->release();    
        handle_t = nullptr;
        m_len = 0;
    }
}

bool vidReader::init(std::string filename){
    release();
    VideoCapture* vid = new VideoCapture();
    bool ret = vid->open(filename);
    if(!ret) {
        std::cout << "video open failed"<<std::endl;
        vid->release();
        return ret;
    }
    m_len = vid->get(CV_CAP_PROP_FRAME_COUNT);
    handle_t = reinterpret_cast<void*>(vid);
    return ret;
}

unsigned char* vidReader::getbgrImg(int &width, int &height){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl; return nullptr;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);
    width = img.cols;
    height = img.rows;
    unsigned char* buf = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
    memcpy(buf, img.data, img.total()*3);
    return buf;
}

unsigned char* vidReader::getyuvImg(int &width, int &height, int &stride){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl; return nullptr;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);
    unsigned char* dst_ptr = nullptr;
    dst_ptr = BGR2YUV_nv21_with_stride(img, width, height, stride, 2);
    return dst_ptr;
}

VIDOUT* vidReader::getImg(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);

    unsigned char* bgrbuf = (unsigned char*)malloc(img.total()*3*sizeof(unsigned char));
    memcpy(bgrbuf, img.data, img.total()*3);

    int width, height, stride;
    unsigned char* yuvbuf = nullptr;
    yuvbuf = BGR2YUV_nv21_with_stride(img, width, height, stride, 2);

    VIDOUT* rett = new VIDOUT();
    rett->bgrbuf = bgrbuf;
    rett->yuvbuf = yuvbuf;
    rett->w = width;
    rett->h = height;
    rett->s = stride;
    rett->_w = img.cols;
    rett->_h = img.rows;
    return rett;
}

int vidReader::width(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FRAME_WIDTH);
}

int vidReader::height(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FRAME_HEIGHT);    
}

int vidReader::fps(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FPS);
}

bool vidWriter::init(std::string filename, int width, int height, int fps){
    release();
    VideoWriter* vid = new VideoWriter();
    bool ret = vid->open(filename, CV_FOURCC('D','I','V','X'), fps, Size(width, height));
    // bool ret = vid->open(filename, CV_FOURCC('H','2','6','4'), fps, Size(width, height));
    if(!ret) {
        vid->release();
        return ret;
    }
    m_fps = fps;
    m_height = height;
    m_width = width;
    handle_t = reinterpret_cast<void*>(vid);
    return ret;
}

void vidWriter::release(){
    if(handle_t!=nullptr){
        VideoWriter* vid = reinterpret_cast<VideoWriter*>(handle_t);
        vid->release();    
        handle_t = nullptr;
    }    
}

void vidWriter::writeImg(unsigned char* buf, int bufw, int bufh){
    VideoWriter* vid = reinterpret_cast<VideoWriter*>(handle_t);
    Mat img( Size(bufw, bufh),CV_8UC3, buf);
    Mat img_fit;
    resize(img, img_fit, Size(m_width, m_height));
    vid->write(img_fit);
}

/*--------------Clocker------------------*/
Clocker::Clocker(){
    ctx = new Timer();
}

Clocker::~Clocker(){
    delete reinterpret_cast<Timer*>(ctx);
}

void Clocker::start(){
    Timer* cTx = reinterpret_cast<Timer*>(ctx);
    cTx->start();
}

double Clocker::end(std::string title, bool display){
    Timer* cTx = reinterpret_cast<Timer*>(ctx);
    return cTx->end(title, display);
}
/*--------------END Clocker------------------*/


/*******************************************************************************
输入argc argv的解析
chaffee.chen@2022-10-20
*******************************************************************************/
/*--------------BEGIN ArgParser------------------*/
bool ArgParser::add_argument(const std::string &keyword, int default_value , const std::string &helpword){
    //if(std::is_same<typename std::decay<T>::type,int>::value)
    std::string valType = "int";
    m_cmd_int.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + std::to_string(default_value);
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

bool ArgParser::add_argument(const std::string &keyword, float default_value , const std::string &helpword){
    //if(std::is_same<typename std::decay<T>::type,int>::value)
    std::string valType = "float";
    m_cmd_float.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + std::to_string(default_value);
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

// bool ArgParser::add_argument(const std::string &keyword, bool default_value , const std::string &helpword){
//     //if(std::is_same<typename std::decay<T>::type,int>::value)
//     std::string valType = "bool";
//     m_cmd_bool.insert(std::make_pair(keyword, default_value));
//     std::string _helpword = helpword+" ["+valType+"], default value = " + (default_value?"true":"false");
//     m_cmd_help.insert(std::make_pair(keyword, _helpword));
//     printf("+ %s\n", _helpword.c_str());
//     return true;
// }

bool ArgParser::add_argument(const std::string &keyword, const std::string &default_value , const std::string &helpword){
    std::string valType = "string";
    m_cmd_str.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + default_value;
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

bool ArgParser::parser(int argc, char* argv[]){
    printf("------------------------------------------\n");
    printf("parser\n");
    printf("------------------------------------------\n");
    for(int i = 0; i < argc; i++){
        std::string keyword = std::string(argv[i]);
        // printf("%s ", keyword.c_str());
        if(keyword=="-help"||keyword=="--help"){
            print_help();
            return false;
        }
    }
    printf("start parsering...\n");
    for(int i = 0 ; i < argc -1; i++ ){
        std::string keyword = std::string(argv[i]);
        // printf("%s \n", argv[i]);
        if(m_cmd_help.find(keyword)==m_cmd_help.end()){
            // printf("%s command is not known, which will be skipped.\n", keyword.c_str());
            continue;
        }
        if(m_cmd_int.find(keyword)!=m_cmd_int.end()){
            m_cmd_int[keyword] = std::atoi(argv[i+1]);
            printf("set [%s] to %d\n", keyword.c_str() ,m_cmd_int[keyword]);
        }
        if(m_cmd_str.find(keyword)!=m_cmd_str.end()){
            m_cmd_str[keyword] = std::string(argv[i+1]);
            printf("set [%s] to %s\n", keyword.c_str() ,m_cmd_str[keyword].c_str());
        }
        // if(m_cmd_bool.find(keyword)!=m_cmd_bool.end()){
        //     m_cmd_bool[keyword] = std::atoi(argv[i+1]) > 0 ? true:false ;
        // }
        if(m_cmd_float.find(keyword)!=m_cmd_float.end()){
            m_cmd_float[keyword] = std::atof(argv[i+1]);
            printf("set [%s] to %.3f\n", keyword.c_str() ,m_cmd_float[keyword]);
        }
    }
    // print_help();
    printf("------------------------------------------\n");
    return true;
}

void ArgParser::print_help(){
    printf("------------------------------------------\n");
    printf("help\n");
    printf("------------------------------------------\n");
    for(auto&& cmd: m_cmd_help){
        printf("%s : %s\n", cmd.first.c_str(), cmd.second.c_str());
    }
    printf("------------------------------------------\n");
}

float ArgParser::get_value_float(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return 0;
    }
    if(m_cmd_float.find(keyword)!=m_cmd_float.end()){
        return m_cmd_float[keyword];
    }
    return 0;
}

int ArgParser::get_value_int(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return 0;
    }
    if(m_cmd_int.find(keyword)!=m_cmd_int.end()){
        return m_cmd_int[keyword];
    }
    return 0;
}

// bool ArgParser::get_value_bool(const std::string &keyword){
//     if(m_cmd_help.find(keyword)==m_cmd_help.end()){
//         printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
//         return 0;
//     }
//     if(m_cmd_bool.find(keyword)!=m_cmd_bool.end()){
//         return m_cmd_bool[keyword];
//     }
//     return 0;
// }

std::string ArgParser::get_value_string(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return "";
    }
    if(m_cmd_str.find(keyword)!=m_cmd_str.end()){
        return m_cmd_str[keyword];
    }
    return "";
}