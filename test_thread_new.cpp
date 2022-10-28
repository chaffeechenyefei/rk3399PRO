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
        AlgoAPIName apiName;
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
        // retcode = ptrMainHandle->init(init_param);
        std::map<InitParam, WeightData> weightConfig;
        for(auto &&param: init_param){
            int tmpSz = 0;
            unsigned char* tmpPtr = readfile(param.second.c_str(), &tmpSz);
            printf("data addr:%d, bufsize:%d\n", tmpPtr , tmpSz);
            WeightData tmp{tmpPtr, tmpSz};
            weightConfig[param.first] = tmp;
        }
        {
            std::lock_guard<std::mutex> lk(cmutex);
            retcode = ptrMainHandle->init(weightConfig);
        }

        for(auto &&param: weightConfig){
            free(param.second.pData);
        }        
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
            if(i%50 == 1 ){
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
            int inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
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

    ArgParser myParser;
    myParser.add_argument("-t1", 1, "task1");
    myParser.add_argument("-t2", -1, "task2");
    myParser.add_argument("-t3", -1, "task3");
    myParser.add_argument("-t4", -1, "task4");
    myParser.add_argument("-data","data/image/","image path");
    myParser.add_argument("-loop",1, "loop times");
    myParser.add_argument("-list",0, "list all the task");
    if(!myParser.parser(argc, argv)) return -1;

    if(myParser.get_value_int("-list")>0){
        print_all_task();
        return -1;
    }

    std::vector<TASKNAME> taskids;
    if(myParser.get_value_int("-t1") >= 0){
        taskids.push_back(TASKNAME(myParser.get_value_int("-t1")));
    }
    if(myParser.get_value_int("-t2") >= 0){
        taskids.push_back(TASKNAME(myParser.get_value_int("-t2")));
    }
    if(myParser.get_value_int("-t3") >= 0){
        taskids.push_back(TASKNAME(myParser.get_value_int("-t3")));
    }    
    if(myParser.get_value_int("-t4") >= 0){
        taskids.push_back(TASKNAME(myParser.get_value_int("-t4")));
    }

    bool use_track = true;
    int thread_num = taskids.size();
    int num_loops_each_thread = myParser.get_value_int("-loop");
    string datapath = myParser.get_value_string("-data");

    printf("=======================\n");
    printf("Createing [%d] threads each loop [%d] times\n", thread_num, num_loops_each_thread);
    printf("=======================\n");
    for(int thread_id = 0; thread_id < thread_num; thread_id++)
        create_thread_for_yolo_task(thread_id, taskids[thread_id] , datapath, num_loops_each_thread, use_track);

    pthread_exit(NULL);
    return 0;
};