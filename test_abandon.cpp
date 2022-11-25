/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "libai_core.hpp"
#include "config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace ucloud;

const int fps = 25;//视频实际帧率
const int interval_ms = 100;//采样间隔ms
int interval_fps = ((float)interval_ms)/1000*fps + 1;//mlu220下fps间隔


vector<string> readFile(string &datapath){
    ifstream fd;
    string line_s,filename;
    vector<string> filelst;
    fd.open(datapath.c_str());
    if (!fd)
    {
        cout<<"file "<<datapath<<" open failed !"<<endl;
    }
    while(getline(fd,line_s)){
        stringstream  item(line_s);
        while(item>>filename){
            filelst.push_back(filename);
            // cout<<"filename is "<<filename<<endl;
        }
    }
    fd.close();
    return filelst;

}

void create_thread_for_yolo_task(int thread_id, TASKNAME taskid ,vector<string> datapath){
    cout<<"step into create yolo task"<<endl;
    int frame_limit = 10000;

    unsigned char* imgBuf=nullptr;
    unsigned char* foreImgbuf = nullptr;
    int fmt_w,fmt_h,width,height,stride,datasize;
  
   
    RET_CODE retcode = RET_CODE::FAILED;
    float threshold, nms_threshold;
    AlgoAPIName apiName, apiSubName;
    std::map<InitParam, std::string> init_param;
    int use_batch = 0;

    bool flag_parser = task_parser(taskid, threshold, nms_threshold, apiName, init_param, use_batch);
    AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
    retcode = ptrMainHandle->init(init_param);
    if( retcode != RET_CODE::SUCCESS )
    { 
        std::cout << "algo initial failed" << endl; 
        return; 
    }
    for (int k=0;k<1000;k++){
    for (int i=1;i<datapath.size();i++)
    {   
        cout<<"i"<<i<<" image file name "<<datapath[i]<<endl;
        cv::Mat bgMat = cv::imread(datapath[0]);
        cv::Mat foreMat = cv::imread(datapath[i]);
        std::string savename= datapath[i].replace(datapath[i].find("."),4,"_"+to_string(i)+"result.jpg");
        std::cout<< "savename is "<<savename<<std::endl;
        foreImgbuf = (unsigned char * )malloc(foreMat.total()*3);
        memcpy(foreImgbuf,foreMat.data,foreMat.total()*3);
        // cout<< "input mat type"<<foreMat.type()<<endl;
        cv::Mat matDiff;
        cv::absdiff(foreMat,bgMat,matDiff);
        // cout<<"mat Diff type"<< matDiff.type()<<endl;
        imgBuf = (unsigned char*)malloc(matDiff.total()*3);
        memcpy(imgBuf,matDiff.data,matDiff.total()*3);
        width = bgMat.cols;
        height = bgMat.rows;
        stride = width;
        datasize = width*height*3;
        TvaiImage tvimage{TVAI_IMAGE_FORMAT_BGR,width,height,stride,imgBuf, datasize,1};
       
        VecObjBBox bboxes; 
        retcode = ptrMainHandle->run(tvimage, bboxes,threshold, nms_threshold);

        // VecObjBBox _bboxes;
        if (!bboxes.empty()){
            for(auto &&box :bboxes){      
                printf("[%d]%f,%f,%d,%d,%d,%d \n",box.objtype, box.confidence, box.objectness, box.rect.x, box.rect.y, box.rect.width, box.rect.height);
                // if(box.objtype == 800){
                //     _bboxes.push_back(box);
                //     printf("[%d]%f,%f,%d,%d,%d,%d \n",box.objtype, box.confidence, box.objectness, box.rect.x, box.rect.y, box.rect.width, box.rect.height);
                // }   
            }
        }
        std::cout<<"step into draw image!"<<" height "<< height <<" width "<<width <<std::endl; 
        cout<<endl;
        if (!bboxes.empty()){
           
            drawImg(foreImgbuf, width, height, bboxes, true, true, false, 1);
            writeImg(savename, foreImgbuf, width , height,false);
        }
       
        if (foreImgbuf) free(foreImgbuf);
        if(imgBuf) free(imgBuf);
        bboxes.clear();
    }
}
}

/*-------------------------------------------
                  Main Function
./test_case_vid {mlu220/mlu270} {datapath} {taskid} {use tracking or not}
-------------------------------------------*/
int main(int argc, char **argv)
{  
    bool use_track = false;
    bool simulate_mlu220 = false;
    bool dont_infer = false;
    string filename = "./abandon01_same.txt";
    vector<string> datapath = readFile(filename);
    TASKNAME taskid = TASKNAME::ABANDON_OBJECT;
    std::cout << "==========FPS===========" << std::endl;
    // string _tmp(argv[1]);
    // if(_tmp=="mlu220") simulate_mlu220 = true;
    // if(simulate_mlu220)
    //     std::cout << "simulate_mlu220: interval fps = " << interval_fps << std::endl;
    // else
    //     std::cout << "normal mlu270" << std::endl;
    // std::cout << "=====================" << std::endl;        

    // if(argc>=3){
    //     string _tmp(argv[2]);
    //     datapath.push_back(_tmp);
    // }
    // if(argc >= 4){
    //     int _taskid = atoi(argv[3]);
    //     taskid = TASKNAME(_taskid);
    // }
    // if (argc >= 5){
    //     use_track = true;
    //     std::cout << "use tracking" << endl;
    // } else {
    //     use_track = false;
    //     std::cout << "no tracking" << endl;
    // }
    // if(argc>=6){
    //     dont_infer = true;
    //     std::cout << "video will be saved only." << endl;
    // }

    create_thread_for_yolo_task(0, taskid , datapath);

    // pthread_exit(NULL);
    return 0;
};
