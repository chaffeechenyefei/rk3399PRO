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

    double tm_cost = 0;
    int num_result = 0;
    int max_track_id = -1;
    std::cout << "yolo::loading model for thread #" << thread_id << endl;
    //Get Algo API
    AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
    std::cout << "AICoreFactory done!" << endl;
    //Initial model with loading weights
    retcode = ptrMainHandle->init(init_param);
    if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo initial failed" << endl; return; }
    //Set model parameters
    // ptrMainHandle->set_param(threshold, nms_threshold);

    VecObjBBox show_bboxes;
    int width, height, stride;
    for(int i = 0; i < num_loops_each_thread; i++){
        VecObjBBox bboxes; 
        std::string imgname =datapath;
        printf("loading %s\n", imgname.c_str());
        //将图像resize到1280x720, 模拟摄像头输入
        unsigned char* imgBuf = readImg_to_NV21(imgname, 1280, 720, width, height, stride);
        int inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
        TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,imgBuf, inputdata_sz};

        auto start = chrono::system_clock::now();
        RET_CODE _ret_ = ptrMainHandle->run(tvimage, bboxes, threshold , nms_threshold);
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        num_result += bboxes.size();
        free(imgBuf);
        if(i == num_loops_each_thread-1) show_bboxes = bboxes;
    }
    for(auto &&box : show_bboxes){
        printf("id[%d], type[%d], x,y,w,h = %d,%d,%d,%d, confidence = %.3f, objectness = %.3f \n", 
        box.track_id, box.objtype,
        box.rect.x, box.rect.y, box.rect.width, box.rect.height,
        box.confidence, box.objectness
        );
    }
    unsigned char* imgBuf = readImg_to_BGR(datapath, 1280, 720, width, height);
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
    bool use_track = true;
    int num_loops = 1;
    string datapath;
    TASKNAME taskid = TASKNAME::PED_CAR_NONCAR;     

    if(argc>=2){
        string _tmp(argv[1]);
        datapath = _tmp;
    }
    if(argc >= 3){
        int _taskid = atoi(argv[2]);
        taskid = TASKNAME(_taskid);
    }
    if(argc >= 4){
        num_loops = atoi(argv[3]);
    }

    printf("=======================\n");
    printf("=======================\n");
    create_thread_for_yolo_task(0, taskid , datapath, num_loops, use_track);

    // pthread_exit(NULL);
    return 0;
};