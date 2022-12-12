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
int max_cnt = 0;

static float calcSimilarity(float* fA, float* fB, int dims){
    float val = 0;
    for(int i=0 ; i < dims ; i++ )
        val += (*fA++) * (*fB++);
    return val;
};

void create_thread_for_yolo_task(int thread_id, TASKNAME taskid ,string datapath){
    RET_CODE retcode = RET_CODE::FAILED;
    float threshold, nms_threshold;
    AlgoAPIName apiNameDetector, apiNameExtractor;
    std::map<InitParam, std::string> init_param;
    int use_batch = 0;
    bool flag_parser = task_parser(TASKNAME::FACE, threshold, nms_threshold, apiNameDetector, init_param, use_batch);
    if(!flag_parser) {
        std::cout << "parser failed" << std::endl;
        return;
    }

    double tm_cost = 0;
    int num_result = 0;
    int max_track_id = -1;
    std::cout << "loading model for thread #" << thread_id << endl;
    //Get Algo API
    AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiNameDetector);
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

    float threshold2, nms_threshold2;
    std::map<InitParam, std::string> init_param2;
    flag_parser = task_parser(TASKNAME::FACE_EXT, threshold, nms_threshold, apiNameExtractor, init_param2, use_batch);
    if(!flag_parser) {
        std::cout << "parser2 failed" << std::endl;
        return;
    }
     AlgoAPISPtr ptrSubHandle = ucloud::AICoreFactory::getAlgoAPI(apiNameExtractor);
     retcode = ptrSubHandle->init(init_param2);
     if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo2 initial failed" << endl; return; }


    ifstream infile;
    string filename = datapath + "/list.txt";
    infile.open(filename, std::ios::in);
    string imgname;
    vector<string> vec_imgnames;
    while(infile >> imgname){
        std::string imgname_not_full = imgname;
        vec_imgnames.push_back(imgname_not_full);
    }
    infile.close();
    printf("total [%d] images listed...\n", vec_imgnames.size());

    int width, height, stride;
    std::map<std::string,float*> features;
    int cnt = 0;
    for(auto &&imgname: vec_imgnames){
        if(cnt++>=max_cnt) break;
        VecObjBBox bboxes; 
        printf("loading %s\n", imgname.c_str());
        std::string imgname_full = datapath + "/" + imgname;
        unsigned char* imgBuf = nullptr;
        int inputdata_sz = 0;
        TvaiImage tvimage;
        // imgBuf = readImg_to_NV21(imgname_full, 256, 256, width, height, stride);
        imgBuf = readImg_to_NV21(imgname_full, width, height, stride);
        inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
        tvimage = {TVAI_IMAGE_FORMAT_NV21,width,height,stride,imgBuf, inputdata_sz};
        printf("image with size(whs) %d, %d, %d\n", width, height, stride);
        // BBox box;
        // box.rect = {0,0,width,height};
        // bboxes.push_back(box);
        //将图像resize到1280x720, 模拟摄像头输入
        auto start = chrono::system_clock::now();
        RET_CODE _ret_ = ptrMainHandle->run(tvimage, bboxes, threshold , nms_threshold);
        if(_ret_!=RET_CODE::SUCCESS) {
            printf("%s err in main handle\n", imgname.c_str());
            free(imgBuf);
            continue;
        }
        if(bboxes.empty()){
            printf("no face detected in %s\n", imgname.c_str());
            BBox box;
            box.rect = {0,0,width, height};
            bboxes.push_back(box);
        } else{
            for(auto &&_b_: bboxes){
                printf("%d,%d,%d,%d in %d,%d\n", _b_.rect.x , _b_.rect.y, _b_.rect.width, _b_.rect.height, width, height);
            }
        }
        _ret_ = ptrSubHandle->run(tvimage, bboxes, 0 , 0 );
        if(_ret_!=RET_CODE::SUCCESS) {
            printf("%s err in sub handle\n", imgname.c_str());
            free(imgBuf);
            continue;
        }        
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

        float *feat = (float*)malloc(512*sizeof(float));
        memcpy(feat, bboxes[0].feat.pFeature, 512*sizeof(float));
        features.insert(std::pair<std::string,float*>(imgname,feat));
        ucloud::AICoreFactory::releaseVecObjBBox(bboxes);
        free(imgBuf);
    }

    printf("average feature extraction cost for per 112x112 image = %1.4fs\n", tm_cost/features.size());

    //calc similarity
    std::cout << "total images: " << vec_imgnames.size() << std::endl;
    std::cout << "total features: " << features.size() << std::endl;
    for(int i = 0; i < features.size(); i++ ){
        if(i%2==1) continue;//奇数进行计算
        // std::cout << i << std::endl;
        for(int j = i+1; j < features.size(); j++){
            if(j%2==0) continue;//偶数进行计算
            // std::cout << j << std::endl;
            float* fA = features[vec_imgnames[i]];
            float* fB = features[vec_imgnames[j]];
            float similarity = calcSimilarity(fA, fB, 512);
            if(j==i+1)
                printf("\033[31m%s\t%s\t%1.3f\n\033[0m",vec_imgnames[i].c_str(),vec_imgnames[j].c_str(),similarity);
            else
                printf("%s\t%s\t%1.3f\n",vec_imgnames[i].c_str(),vec_imgnames[j].c_str(),similarity);
        }
    }


    for(auto &&pf: features){
        free(pf.second);
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
    //register image 与 probe image的顺序是奇偶顺序交叉排列
    myParser.add_argument("-data","data/image","input image path");
    myParser.add_argument("-w", 112, "input image width");
    myParser.add_argument("-h", 112, "input image height");
    myParser.add_argument("-n",10,"max input image");
    if(!myParser.parser(argc, argv)) return -1;

    string datapath = myParser.get_value_string("-data");

    use_string_to_init = false;
    W = myParser.get_value_int("-w");
    H = myParser.get_value_int("-h");
    max_cnt = myParser.get_value_int("-n");

    printf("=======================\n");
    printf("=======================\n");
    create_thread_for_yolo_task(0, TASKNAME(0) , datapath);

    // pthread_exit(NULL);
    return 0;
};