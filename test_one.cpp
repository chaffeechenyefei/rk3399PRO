/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <thread>

#include "libai_core.hpp"
#include "config.hpp"

using namespace std;
using namespace ucloud;

std::mutex cmutex;
int W = 0;
int H = 0;
float user_threshold = -1;
bool use_string_to_init = true;

void create_thread_for_yolo_task(int thread_id, TASKNAME taskid ,string datapath, int num_loops_each_thread, bool use_track=false ){
    RET_CODE retcode = RET_CODE::FAILED;
    float threshold, nms_threshold;
    AlgoAPIName apiName, apiSubName;
    std::map<InitParam, std::string> init_param;
    int use_batch = 0;
    bool flag_parser = task_parser(taskid, threshold, nms_threshold, apiName, init_param, use_batch);
    if(!flag_parser) {
        std::cout << "parser failed" << std::endl;
        return;
    }
    if(user_threshold>0){
        threshold = user_threshold;
        printf("threshold is set to %1.3f by user input\n", threshold);
    }

    double tm_cost = 0;
    int num_result = 0;
    int max_track_id = -1;
    std::cout << "yolo::loading model for thread #" << thread_id << endl;
    //Get Algo API
    AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
    std::cout << "AICoreFactory done!" << endl;
    //Initial model with loading weights
    if(use_string_to_init)
        retcode = ptrMainHandle->init(init_param);
    else{
        printf("**using weight config to init\n");
        std::map<InitParam, WeightData> weightConfig;
        for(auto &&param: init_param){
            int tmpSz = 0;
            unsigned char* tmpPtr = readfile(param.second.c_str(), &tmpSz);
            WeightData tmp{tmpPtr, tmpSz};
            weightConfig[param.first] = tmp;
        }
        retcode = ptrMainHandle->init(weightConfig);
        for(auto &&param: weightConfig){
            free(param.second.pData);
        }
    }
    if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo initial failed" << endl; return; }
    //Set model parameters
    // ptrMainHandle->set_param(threshold, nms_threshold);

    VecObjBBox show_bboxes;
    int width, height, stride;
    for(int i = 0; i < num_loops_each_thread; i++){
        VecObjBBox bboxes; 
        std::string imgname =datapath;
        printf("loading %s\n", imgname.c_str());
        unsigned char* imgBuf = nullptr;
        int inputdata_sz = 0;
        TvaiImage tvimage;
        if(imgname.find(".yuv") != std::string::npos){//如果输入图像后缀是YUV则直接读取
            imgBuf = yuv_reader(imgname, W, H);
            width = W; height = H; stride = W;
            inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
            tvimage = {TVAI_IMAGE_FORMAT_NV21,width,height,stride,imgBuf, inputdata_sz};
        } else {
            imgBuf = readImg_to_NV21(imgname, W, H, width, height, stride);
            inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
            tvimage = {TVAI_IMAGE_FORMAT_NV21,width,height,stride,imgBuf, inputdata_sz};
        }
        //将图像resize到1280x720, 模拟摄像头输入
        auto start = chrono::system_clock::now();
        RET_CODE _ret_ = ptrMainHandle->run(tvimage, bboxes, threshold , nms_threshold);
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        num_result += bboxes.size();
        free(imgBuf);
        if(i == num_loops_each_thread-1) show_bboxes = bboxes;
    }
    printf("total %d bboxes returned\n", show_bboxes.size());
    for(auto &&box : show_bboxes){
        printf("id[%d], type[%d], x,y,w,h = %d,%d,%d,%d, confidence = %.3f, objectness = %.3f \n", 
        box.track_id, box.objtype,
        box.rect.x, box.rect.y, box.rect.width, box.rect.height,
        box.confidence, box.objectness
        );
    }

    unsigned char* imgBuf = nullptr;
    if(datapath.find(".yuv") != std::string::npos){//如果输入图像后缀是YUV则直接读取
        imgBuf = yuv_reader(datapath, W, H , true);
    } else{
        imgBuf = readImg_to_BGR(datapath, W, H, width, height);
    }
    
    if(imgBuf){
        drawImg(imgBuf, width, height, show_bboxes, true, true, false, 1);
        writeImg("result.jpg", imgBuf, width , height);
        free(imgBuf);
    }
    return;
}

/*-------------------------------------------
                  Main Function
./test_one {datapath} {taskid} {num loops}
-------------------------------------------*/
int main(int argc, char **argv)
{  
    ArgParser myParser;
    myParser.add_argument("-data","test.jpg","input image");
    myParser.add_argument("-task",1, "taskid");
    myParser.add_argument("-loop",1, "loop times");
    myParser.add_argument("-threshold",-1, "threshold(if less than 0, value from task parser will be applied.)");
    myParser.add_argument("-w", 1280, "input image width");
    myParser.add_argument("-h", 720, "input image height");
    myParser.add_argument("-list",0, "list all the task");
    myParser.add_argument("-init",0,"0: std::string for init, 1:WeightConfig for init");
    if(!myParser.parser(argc, argv)) return -1;

    if(myParser.get_value_int("-list")>0){
        print_all_task();
        return -1;
    }

    bool use_track = true;
    int num_loops = myParser.get_value_int("-loop");
    string datapath = myParser.get_value_string("-data");
    TASKNAME taskid = TASKNAME(myParser.get_value_int("-task"));     
    use_string_to_init = myParser.get_value_int("-init") == 0 ? true:false;
    W = myParser.get_value_int("-w");
    H = myParser.get_value_int("-h");
    user_threshold = myParser.get_value_float("-threshold");

    printf("=======================\n");
    printf("=======================\n");
    create_thread_for_yolo_task(0, taskid , datapath, num_loops, use_track);

    // pthread_exit(NULL);
    return 0;
};