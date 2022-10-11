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
    thread thread_source([=](){
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

        ifstream infile;
        string filename = datapath + "/list.txt";
        infile.open(filename, std::ios::in);
        string imgname;
        vector<string> vec_imgnames;
        while(infile >> imgname){
            std::string imgname_full = datapath + "/" + imgname;
            vec_imgnames.push_back(imgname_full);
        }
        infile.close();
        printf("total [%d] images listed...\n", vec_imgnames.size());

        for(int i = 0; i < num_loops_each_thread; i++){
            if(i%200 == 1 ){
                std::lock_guard<std::mutex> lk(cmutex);
                printf("#[%02d]-[%05d][%.2f%%]: per cost = %f, for %d targets with max_track_id %d\n", \
                thread_id, i, ((float)i)/num_loops_each_thread*100 , (float)(tm_cost/i), num_result, max_track_id);
                fflush(stdout);
            }
            VecObjBBox bboxes; 
            int width, height, stride;

            imgname = vec_imgnames[i%vec_imgnames.size()];
            //将图像resize到1280x720, 模拟摄像头输入
            unsigned char* imgBuf = readImg_to_NV21(imgname, 1280, 720, width, height, stride);
            int inputdata_sz = 3*width*height/2*sizeof(unsigned char);
            TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,imgBuf, inputdata_sz};

            auto start = chrono::system_clock::now();
            RET_CODE _ret_ = ptrMainHandle->run(tvimage, bboxes, threshold , nms_threshold);
            auto end = chrono::system_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
            tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
            num_result += bboxes.size();
            // { //结果输出到txt
            //     std::string txtname = imgname + ".txt";
            //     ofstream fd;
            //     fd.open(txtname);
            //     if(fd.is_open()){
            //         ostringstream line("");
            //         for(auto &&box: bboxes){
            //             float x = ((float)(box.rect.x));
            //             float y = ((float)box.rect.y);
            //             float w = ((float)box.rect.width);
            //             float h = ((float)box.rect.height);
            //             line << box.objtype << " " << x << " " << y << " " << w << " " << h << " " << box.objectness <<"\n";
            //         }
            //         fd.write(line.str().c_str(), line.str().length());
            //     }
            //     fd.close();
            // }
            // std::cout << bboxes.size() << std::endl;
            for(auto &&box:bboxes){
                if(box.track_id > max_track_id) max_track_id = box.track_id;
            }

            free(imgBuf);
        }

        {
            std::lock_guard<std::mutex> lk(cmutex);
            printf("#[%02d]-[%05d]-[finished]: per cost = %f\n", thread_id, num_loops_each_thread , (float)(tm_cost/num_loops_each_thread));
            fflush(stdout);
        }
        return;
        
    });//end of thread
    thread_source.detach();
}

/*-------------------------------------------
                  Main Function
./test_thread {datapath} {taskid} {thread_num} {num_times_each_loop}
-------------------------------------------*/
int main(int argc, char **argv)
{  
    bool use_track = true;
    int thread_num = 1;
    int num_loops_each_thread = 1000;
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
        thread_num = atoi(argv[3]);
    }
    if(argc >= 5){
        num_loops_each_thread = atoi(argv[4]);
    }    

    printf("=======================\n");
    printf("Createing [%d] threads each loop [%d] times\n", thread_num, num_loops_each_thread);
    printf("=======================\n");
    for(int thread_id = 0; thread_id < thread_num; thread_id++)
        create_thread_for_yolo_task(thread_id, taskid , datapath, num_loops_each_thread, use_track);

    pthread_exit(NULL);
    return 0;
};