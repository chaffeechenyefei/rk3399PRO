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
    
    int fmt_w,fmt_h,width,height,stride,datasize;
    VecObjBBox bboxes; 
   
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
    for (int i=1;i<datapath.size();i++)
    {   
        cout<<"i"<<i<<" image file name "<<datapath[i]<<endl;
        cv::Mat bgMat = cv::imread(datapath[0]);
        cv::Mat foreMat = cv::imread(datapath[i]);
        cout<< "input mat type"<<foreMat.type()<<endl;
        cv::Mat matDiff;
        cv::absdiff(foreMat,bgMat,matDiff);
        cout<<"mat Diff type"<< matDiff.type()<<endl;
        imgBuf = (unsigned char*)malloc(matDiff.total()*3);
        memcpy(imgBuf,matDiff.data,matDiff.total()*3);
        width = bgMat.cols;
        height = bgMat.rows;
        stride = width;
        datasize = width*height*3;
        TvaiImage tvimage{TVAI_IMAGE_FORMAT_BGR,width,height,stride,imgBuf, datasize};
       
        ptrMainHandle->run(tvimage, bboxes);

        VecObjBBox _bboxes;
        for(auto &&box :bboxes){
            printf("[%d]%f,%f,%f,%f,%f,%f \n",box.objtype, box.confidence, box.objectness, box.x0, box.y0, box.x1, box.y1);
            if(box.objtype == CLS_TYPE::ABADNON_STATIC)
                _bboxes.push_back(box);
        }     
        
        if (imgBuf){
            free(imgBuf);
        }
    }
    // interval_fps = simulate_mlu220 ? interval_fps:1;
    // // thread thread_source([=](){
    //     int flag_for_trackid_or_cls = 0;
    //     if(taskid == TASKNAME::GKPW || taskid == TASKNAME::GKPW2 || use_track == false) flag_for_trackid_or_cls = 1;
    //     RET_CODE retcode = RET_CODE::FAILED;
    //     float threshold, nms_threshold;
    //     AlgoAPIName apiName, apiSubName;
    //     std::map<InitParam, std::string> init_param;
    //     int use_batch = 0;

    //     bool flag_parser = task_parser(taskid, threshold, nms_threshold, apiName, init_param, use_batch);
    //     if(!flag_parser) {
    //         std::cout << "parser failed" << std::endl;
    //         return;
    //     }

    //     double tm_cost = 0;
    //     int num_result = 0;
    //     std::cout << "yolo::loading model for thread #" << thread_id << endl;
    //     //Get Algo API
    //     AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
    //     std::cout << "AICoreFactory done!" << endl;
    //     //Initial model with loading weights
    //     retcode = ptrMainHandle->init(init_param);
    //     if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo initial failed" << endl; return; }
    //     //Set model parameters
    //     // ptrMainHandle->set_param(threshold, nms_threshold);

    //     ifstream infile;
    //     string filename = datapath;
    //     int real_infer_num = 0;

    //     vidReader handle_t;
    //     bool ret = handle_t.init(filename);
    //     if(!ret) { std::cout << "vid read handle init failed" << endl; return;}
    //     else{ 
    //         std::cout << "vid read handle init success" << endl;
    //         std::cout << "fps = " << handle_t.fps() << ", width = " << handle_t.width() << ", height = " << handle_t.height() << endl;
    //     }
    //     vidWriter w_handle_t;
    //     std::string savefilename = "x";
    //     savefilename = filename +".avi";//.mkv for h264
        
    //     int ratio = (handle_t.width() > 1920) ? 2:1; // resize video if video is too large to save
    //     ret = w_handle_t.init( savefilename, handle_t.width()/ratio, handle_t.height()/ratio, handle_t.fps() );
    //     if(!ret) { std::cout << "vid write handle init failed" << endl; return;}
    //     else{ 
    //         std::cout << "vid write handle init success" << endl;
    //     }
    //     int frame_cnt = 0;
    //     std::vector<VIDOUT*> frameBuf;
    //     unsigned char* static_frame = nullptr;
    //     while(1){
    //         if(frame_cnt%100==0){ std::cout << ((float)frame_cnt)/handle_t.len() << ": " << num_result << " detected" << endl; }
    //         if(frame_cnt >= handle_t.len() || frame_cnt > frame_limit) break;
    //         VecObjBBox bboxes; 
    //         int width, height, stride;
    //         VIDOUT* vidtmp = nullptr;
    //         vidtmp = handle_t.getImg();  //bgr img for illustration; yuv img for infer
    //         if(vidtmp==nullptr) {
    //             std::cout << "vid handle get EOF" << endl;
    //             break;
    //         }
    //         frameBuf.push_back(vidtmp);
    //         frame_cnt++;

    //         if(use_batch <= 1 || taskid==TASKNAME::GKPW){//single frame
    //             bool flag_do_infer = true;
    //             //simulate mlu220情况下，只有interval_fps的需要推理，其他都不需要；mlu270都需要推理
    //             if(simulate_mlu220 && (frame_cnt-1)%interval_fps != 0 ) flag_do_infer = false;
    //             if( flag_do_infer ){
    //                 width = frameBuf[0]->w; height = frameBuf[0]->h; stride = frameBuf[0]->s;
    //                 int inputdata_sz = 3*width*height/2*sizeof(unsigned char);
    //                 TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,frameBuf[0]->yuvbuf, inputdata_sz};
    //                 auto start = chrono::system_clock::now();
    //                 ptrMainHandle->run(tvimage, bboxes);
    //                 auto end = chrono::system_clock::now();
    //                 auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    //                 tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    //                 num_result += bboxes.size();
    //                 real_infer_num++;
    //                 if(!bboxes.empty()){
    //                     bool flag_disp_label = true;
    //                     if(taskid==TASKNAME::GKPW2) flag_disp_label = false;
    //                     bool use_rand_color = false;
                        
    //                     if(taskid==TASKNAME::GKPW){
    //                         VecObjBBox _bboxes;
    //                         for(auto &&box :bboxes){
    //                             if(box.objtype == CLS_TYPE::FALLING_OBJ)
    //                                 _bboxes.push_back(box);
    //                         }
    //                         if(!dont_infer)
    //                             drawImg( frameBuf[0]->bgrbuf, width, height, _bboxes, false, false, false, 1);
    //                     }
    //                     else{
    //                         if(!dont_infer)
    //                             drawImg( frameBuf[0]->bgrbuf, width, height, bboxes, true, flag_disp_label, use_rand_color, flag_for_trackid_or_cls);
    //                     }
                            

    //                 }
    //             }
    //             w_handle_t.writeImg(frameBuf[0]->bgrbuf, width, height);
    //             frameBuf[0]->release();
    //             frameBuf.erase(frameBuf.begin());
    //         }else{// multi frame (2 or 8 frames)
    //             if(frameBuf.size() < use_batch*interval_fps ) continue;
    //             else{
    //                 width = frameBuf[0]->w; height = frameBuf[0]->h; stride = frameBuf[0]->s;
    //                 int inputdata_sz = 3*width*height/2*sizeof(unsigned char);
    //                 BatchImageIN tvimages;

    //                 int _cnt = 0;
    //                 for(auto iterBuf = frameBuf.begin(); iterBuf!=frameBuf.end(); iterBuf++,_cnt++){
    //                     if(_cnt%interval_fps == 0){
    //                         TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,(*iterBuf)->yuvbuf, inputdata_sz};
    //                         tvimages.push_back(tvimage);
    //                     }
    //                     // if(static_frame!=nullptr && iterBuf == frameBuf.begin() && use_static_frame){
    //                     //     if(frame_cnt==10) std::cout << "infer using static frame" << endl;
    //                     //     TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,static_frame, inputdata_sz};
    //                     //     tvimages.push_back(tvimage);
    //                     // } else {
    //                     //     TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,(*iterBuf)->yuvbuf, inputdata_sz};
    //                     //     tvimages.push_back(tvimage);
    //                     // }
    //                 }
    //                 // assert(tvimages.size() == use_batch );
    //                 auto start = chrono::system_clock::now();
    //                 ptrMainHandle->run(tvimages, bboxes);
    //                 auto end = chrono::system_clock::now();
    //                 auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    //                 tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    //                 num_result += bboxes.size();
    //                 real_infer_num++;
    //                 if(!bboxes.empty()){
    //                     drawImg( frameBuf[frameBuf.size()-1]->bgrbuf, width, height, bboxes, false, false, false, 1);
    //                 }
    //                 w_handle_t.writeImg(frameBuf[frameBuf.size()-1]->bgrbuf, width, height);

    //                 if(simulate_mlu220){//mlu220采用非连续方式150ms的两个图像输入
    //                     // frameBuf.erase(frameBuf.begin(),frameBuf.begin()+interval_fps);
    //                     // if(frameBuf.size() != interval_fps*(use_batch-1)) std::cout << "ERROR: " << frameBuf.size() << std::endl;
    //                     frameBuf.clear();
    //                 }else{//mlu270采用连续的方式进行
    //                     frameBuf[0]->release();
    //                     frameBuf.erase(frameBuf.begin());
    //                 }

    //             } // multi process
    //         } // single or multi frames
    //     } // end while

    //     for(auto iter=frameBuf.begin(); iter!=frameBuf.end(); iter++){
    //         (*iter)->release();
    //     }
    //     frameBuf.clear();
    //     if(static_frame!=nullptr) free(static_frame);
    //     std::cout << "yolo::thread #" << thread_id << " : [" << num_result << "]" << endl;
    //     std::cout << "avg tm cost = " << tm_cost/real_infer_num << std::endl;
    // // });//end of thread
    // thread_source.detach();
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
    string filename = "./abandon01.txt";
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
