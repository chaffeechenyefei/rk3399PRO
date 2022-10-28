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
#include "config.hpp"
using namespace std;
using namespace ucloud;
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{   
    TASKNAME taskid = TASKNAME::PED_CAR_NONCAR;
    float threshold, nms_threshold;
    AlgoAPIName algoname;
    std::map<InitParam, std::string> init_param;
    int use_batch = 0;
    bool flag_parser = task_parser(taskid, threshold, nms_threshold, algoname, init_param, use_batch);

    AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(algoname);
    std::map<InitParam, WeightData> weightConfig;
    for(auto &&param: init_param){
        int tmpSz = 0;
        unsigned char* tmpPtr = readfile(param.second.c_str(), &tmpSz);
        printf("data addr:%d, bufsize:%d\n", tmpPtr , tmpSz);
        WeightData tmp{tmpPtr, tmpSz};
        weightConfig[param.first] = tmp;
    }
    RET_CODE retcode = ptrMainHandle->init(weightConfig);
    for(auto &&param: weightConfig){
        free(param.second.pData);
    }
    if(retcode==RET_CODE::SUCCESS)
        printf("**stage 1 initialed\n");
    else
        printf("**stage 1 failed\n");

    taskid = TASKNAME::PHONING;
    flag_parser = task_parser(taskid, threshold, nms_threshold, algoname, init_param, use_batch);

    AlgoAPISPtr ptrMainHandle2 = ucloud::AICoreFactory::getAlgoAPI(algoname);

    for(auto &&param: init_param){
        int tmpSz = 0;
        unsigned char* tmpPtr = readfile(param.second.c_str(), &tmpSz);
        printf("data addr:%d, bufsize:%d\n", tmpPtr , tmpSz);
        WeightData tmp{tmpPtr, tmpSz};
        weightConfig[param.first] = tmp;
    }
    retcode = ptrMainHandle2->init(weightConfig);
    for(auto &&param: weightConfig){
        free(param.second.pData);
    }
    if(retcode==RET_CODE::SUCCESS)
        printf("**stage 2 initialed\n");
    else
        printf("**stage 2 failed\n");


    return 0;
}
