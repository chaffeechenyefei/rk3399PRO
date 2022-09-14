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