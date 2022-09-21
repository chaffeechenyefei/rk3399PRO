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
    printf("test_rk execution\n");
    string baseModelPath = argv[1];
    string imagePath = argv[2];
    // std::cout << baseModelPath << ", " << imagePath << std::endl;
    printf("model = %s, image = %s\n", baseModelPath.c_str(), imagePath.c_str());

    printf("reading image\n");
    int height,width;
    unsigned char* imgBuf = ucloud::readImg_to_RGB(imagePath,width,height);
    if(imgBuf==nullptr){
        printf("no image is read\n");
        return -3;
    }
    TvaiImage tvInp;
    tvInp.pData = imgBuf;
    tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
    tvInp.height = height;
    tvInp.width = width;
    tvInp.stride = width;
    tvInp.dataSize = width*height*3;

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
    int loop_times = 3;
    for(int i = 0; i < loop_times; i++){
        bboxes.clear();
        Tk.start();
        ret = ptrHandle->run(tvInp, bboxes);
        auto tm_cost = Tk.end("ptrHandle->run");
        avg_time += tm_cost;
        if(ret!=RET_CODE::SUCCESS){
            if(imgBuf) free(imgBuf);
            printf("err [%d] in ptrHandle->run(tvInp, bboxes) \n", int(ret));
            return -2;
        }
        int cnt = 0;
        for(auto &&box: bboxes){
            if(cnt++ > 1) break;
            printf("[%d]%f,%f,%f,%f,%f,%f \n",box.objtype, box.confidence, box.objectness, box.x0, box.y0, box.x1, box.y1);
        }
        printf("total [%d] detected\n", bboxes.size());
    }
    printf("avg exec ptrHandle->run time = %f\n", avg_time/loop_times);


    drawImg(imgBuf, width, height, bboxes, true, true, false, 1);
    writeImg("result.jpg", imgBuf, width , height);

    if(imgBuf) free(imgBuf);
    return 0;
}
