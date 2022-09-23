// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

using namespace std;
using namespace ucloud;
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{   
    Clocker Tk;
    printf("test_yolo execution\n");
    string baseModelPath = argv[1];
    string imagePath = argv[2];
    int img_mode = 0;
    if(argc>=4){
        //0:RGB 1:NV21 2:NV12 3:NV21 binary file 4:NV12 binary file
        img_mode = std::atoi(argv[3]);
    }
    // std::cout << baseModelPath << ", " << imagePath << std::endl;
    printf("model = %s, image = %s\n", baseModelPath.c_str(), imagePath.c_str());

    printf("reading image\n");
    TvaiImage tvInp;
    unsigned char* imgBuf = nullptr;
    int height,width,stride;
    switch (img_mode)
    {
    case 0:
        printf("readImg_to_RGB\n");
        imgBuf = ucloud::readImg_to_RGB(imagePath,width,height);
        tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
        tvInp.dataSize = width*height*3;
        break;
    case 1:
        printf("readImg_to_NV21\n");
        imgBuf = ucloud::readImg_to_NV21(imagePath,width,height,stride);
        tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
        tvInp.dataSize = 3*width*height/2;
        break;
    case 2:
        printf("readImg_to_NV12\n");
        imgBuf = ucloud::readImg_to_NV12(imagePath,width,height,stride);
        tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV12;
        tvInp.dataSize = 3*width*height/2;
        break;        
    case 3:
        printf("yuv_reader nv21\n");
        width = 1280;
        height = 720;
        imgBuf = ucloud::yuv_reader(imagePath,width,height);
        tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
        tvInp.dataSize = 3*width*height/2;
        break;
    case 4:
        printf("yuv_reader nv12\n");
        width = 1280;
        height = 720;
        imgBuf = ucloud::yuv_reader(imagePath,width,height);
        tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV12;
        tvInp.dataSize = 3*width*height/2;  
        break;      
    default:
        break;
    }

    if(imgBuf==nullptr){
        printf("no image is read\n");
        return -3;
    }
    
    tvInp.pData = imgBuf;
    tvInp.height = height;
    tvInp.width = width;
    tvInp.stride = width;
    

    printf("get algo api\n");
    ucloud::AlgoAPISPtr ptrHandle = nullptr;
    ptrHandle = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::GENERAL_DETECTOR);
    // ptrHandle->set_param(0.6,0.6,) 这里省略了阈值的设定, 使用默认阈值
    printf("init model\n");
    std::map<ucloud::InitParam,std::string> modelpathes = { {ucloud::InitParam::BASE_MODEL, baseModelPath},};
    RET_CODE ret = ptrHandle->init(modelpathes);
    if(ret!=RET_CODE::SUCCESS){
        printf("err in RET_CODE ret = ptrHandle->init(modelpathes) \n");
        return -1;
    }

    printf("infer\n");
    auto avg_time = 0.f;
    VecObjBBox bboxes;
    int loop_times = 10000;
    for(int i = 0; i < loop_times; i++){
        bboxes.clear();
        Tk.start();
        ret = ptrHandle->run(tvInp, bboxes);
        auto tm_cost = Tk.end("ptrHandle->run", false);
        avg_time += tm_cost;
        if(ret!=RET_CODE::SUCCESS){
            if(imgBuf) free(imgBuf);
            printf("err [%d] in ptrHandle->run(tvInp, bboxes) \n", int(ret));
            return -2;
        }
        if(i%100 == 0)
            printf("avg exec ptrHandle->run time = %f\n", avg_time/i);
        // int cnt = 0;
        // printf("total [%d] detected\n", bboxes.size());
    }
    printf("avg exec ptrHandle->run time = %f\n", avg_time/loop_times);


    if(imgBuf) free(imgBuf);
    return 0;
}
