#include "libai_core.hpp"
#include <opencv2/opencv.hpp>
#include "utils/basic.hpp"
#include "utils/module_siso.hpp"
#include "utils/module_yolo.hpp"
#include <iostream>

using namespace cv;
using namespace ucloud;
using std::cout;
using std::endl;
using std::vector;

/*--------------AICoreFactory API------------------*/
AICoreFactory::AICoreFactory(){LOGI << "AICoreFactory Constructor";}
AICoreFactory::~AICoreFactory(){}

AlgoAPISPtr AICoreFactory::getAlgoAPI(AlgoAPIName apiName){
    AlgoAPISPtr apiHandle = nullptr;
    switch (apiName)
    {
    case AlgoAPIName::UDF_JSON:
        apiHandle = std::make_shared<NaiveModel>();
        break;
    case AlgoAPIName::GENERAL_DETECTOR:
    {
        YOLO_DETECTION* _ptr_ = new YOLO_DETECTION();
        vector<CLS_TYPE> model_output_clss = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
        _ptr_->set_output_cls_order(model_output_clss);
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

unsigned char* ucloud::readImg_to_NV21(std::string filepath, int &width, int &height, int &stride){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv21_with_stride(im, width, height, stride, 2);
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
                stream << std::fixed << std::setprecision(2) << bboxs[i].objectness;
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

double Clocker::end(std::string title){
    Timer* cTx = reinterpret_cast<Timer*>(ctx);
    return cTx->end(title);
}