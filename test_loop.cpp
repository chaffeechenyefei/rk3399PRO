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
    ArgParser myParser;
    myParser.add_argument("-t1", 1, "task1");
    myParser.add_argument("-t2", -1, "task2");
    myParser.add_argument("-t3", -1, "task3");
    myParser.add_argument("-t4", -1, "task4");
    myParser.add_argument("-t5", -1, "task4");
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
    if(myParser.get_value_int("-t5") >= 0){
        taskids.push_back(TASKNAME(myParser.get_value_int("-t5")));
    }    

    int cnt = 0;
    std::map<AlgoAPIName, AlgoAPISPtr> m_handles;
    for(auto &&taskid: taskids){
        printf("== creating taskid %d\n", taskid);
        float threshold, nms_threshold;
        AlgoAPIName algoname;
        std::map<InitParam, std::string> init_param;
        int use_batch = 0;
        if( task_parser(taskid, threshold, nms_threshold, algoname, init_param, use_batch) ){}
        else{
            printf("[%s][%d]ERR task_parser error\n", __FILE__, __LINE__);
            return -1;
        }
        RET_CODE retcode;
        AlgoAPISPtr ptrHandle = ucloud::AICoreFactory::getAlgoAPI(algoname);
        if(ptrHandle!=nullptr) m_handles.insert(std::pair<AlgoAPIName, AlgoAPISPtr>(algoname, ptrHandle));
        else {
            printf("[%s][%d]ERR getAlgoAPI error\n", __FILE__, __LINE__);
            return -1;
        }
        std::map<InitParam, WeightData> weightConfig;
        for(auto &&param: init_param){
            int tmpSz = 0;
            unsigned char* tmpPtr = readfile(param.second.c_str(), &tmpSz);
            printf("data addr:%d, bufsize:%d\n", tmpPtr , tmpSz);
            WeightData tmp{tmpPtr, tmpSz};
            weightConfig[param.first] = tmp;
        }
        {
            retcode = ptrHandle->init(weightConfig);
        }

        for(auto &&param: weightConfig){
            free(param.second.pData);
        }        
        if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo initial failed" << endl; return -1; }
    }

    printf("total [%d] handles created\n", m_handles.size());

    return 0;
}
