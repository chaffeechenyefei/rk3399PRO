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
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "libai_core.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace ucloud;

static unsigned char *load_image(const char *image_path)
{
    //image load RGB
    int req_height = 416;
    int req_width = 736;
    int req_channel = 3;
    printf("w=%d,h=%d,c=%d\n", req_width, req_height, req_channel);

    int height = 0;
    int width = 0;
    int channel = 0;

    unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed!\n");
        return NULL;
    }

    if (width != req_width || height != req_height)
    {
        unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        STBI_FREE(image_data);
        image_data = image_resized;

    }
    // string savename = image_path;
    // savename = savename.replace(savename.find("."),4,"_result.jpg");
    // cv::Mat image(cv::Size(req_width,req_height),CV_8UC3,image_data);
    // cv::cvtColor(image,image,cv::COLOR_RGB2BGR);
    // cv::imwrite(savename,image);


    return image_data;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
//./test_yolo {model path} {data path} {taskid} {img_mode: 0:RGB 1:RGB 2:NV21 3:NV12 4:NV21 binary file 5:NV12 binary file}
int main(int argc, char **argv)
{   
    Clocker Tk;
    printf("test_yolo execution\n");
    string baseModelPath = argv[1];
    string imfile = argv[2];
    vector<string> imlst;
    if (imfile.find("jpg")!=imfile.npos){
        imlst.push_back(imfile);
    }
    else{
        fstream fp;
        fp.open(imfile);
        if (!fp){
        cout<<"imfile cnt open!"<<endl;
        }
        // vector<string> imlst;
        string line,filename;
    
        while (getline(fp,line)){
            stringstream sp(line);
            while(sp>>filename){
                imlst.push_back(filename);
                // printf("image is %s",filename.c_str());
            }
        }
        fp.close();
    } 
    // printf("imlst size %d",imlst.size());

    int img_mode = 0;
    float threshold = 0.2;
    float nms = 0.6;
    int fmt_w = 736; int fmt_h = 416;
    ucloud::TvaiRect roi = {0,0,fmt_w/2, fmt_h};
    bool use_roi = false;
    ucloud::AlgoAPIName algName = ucloud::AlgoAPIName::GENERAL_DETECTOR;
    
    if(argc>=4){
        algName = AlgoAPIName(std::atoi(argv[3]));
    }
    printf("** algName %d\n", algName);    
    if(argc>=5){
        //0:RGB 1:NV21 2:NV12 3:NV21 binary file 4:NV12 binary file
        img_mode = std::atoi(argv[4]);
    }
    if(argc>=6){
        threshold = std::atof(argv[5]);
    }
    if(argc>=7){
        int _use_roi = std::atoi(argv[6]);
        if(_use_roi<=0) use_roi = false;
        else {
            use_roi = true;
            float crop_ratio = ((float)_use_roi)/fmt_w;
            roi = {0,0,_use_roi, int(crop_ratio*fmt_h)};
        }
    }
    printf("** use roi %d\n", use_roi);
    printf("** threshold %0.3f\n",threshold);
    // std::cout << baseModelPath << ", " << imagePath << std::endl;
    printf("model = %s, image = %s\n", baseModelPath.c_str(), imfile.c_str());


    printf("get algo api\n");
    ucloud::AlgoAPISPtr ptrHandle = nullptr;
    ptrHandle = ucloud::AICoreFactory::getAlgoAPI(algName);
    // ptrHandle->set_param(0.6,0.6,) 这里省略了阈值的设定, 使用默认阈值
    // ptrHandle->set_param(0.4,0.4);
    printf("** main init model\n");
    std::map<ucloud::InitParam,std::string> modelpathes = { {ucloud::InitParam::BASE_MODEL, baseModelPath},};
    RET_CODE ret = ptrHandle->init(modelpathes);
    if(ret!=RET_CODE::SUCCESS){
        printf("err in RET_CODE ret = ptrHandle->init(modelpathes) \n");
        return -1;
    }

    printf("** main infer\n");
    auto avg_time = 0.f;
    VecObjBBox bboxes;
    TvaiImage tvInp;
    unsigned char* imgBuf = nullptr;
    int height,width,stride;
    int loop_times = 1;
    // printf("imlst size %d",imlst.size());
    for(int i = 0; i < loop_times; i++){
        for (int j=0;j<imlst.size();j++){
            string imagePath = imlst[j];
            printf("image path is %s\n !",imagePath.c_str());
            // printf("reading image\n");
            switch (img_mode)
            {
            case 0:
                printf("readImg_to_RGB\n");
                
                imgBuf = load_image(imagePath.c_str());
                // imgBuf = ucloud::readImg_to_RGB(imagePath,fmt_w, fmt_h, width,height);
                stride = width;
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
                tvInp.dataSize = width*height*3;
                stride = width;
                break;
            case 1:
                printf("readImg_to_BGR\n");
                imgBuf = ucloud::readImg_to_BGR(imagePath, fmt_w, fmt_h, width,height);
                stride = width;
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_BGR;
                tvInp.dataSize = width*height*3;
                stride = width;
                break;        
            case 2:
                printf("readImg_to_NV21\n");
                imgBuf = ucloud::readImg_to_NV21(imagePath,fmt_w, fmt_h, width,height,stride);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
                tvInp.dataSize = 3*stride*height/2;
                break;
            case 3:
                printf("readImg_to_NV12\n");
                imgBuf = ucloud::readImg_to_NV12(imagePath,fmt_w, fmt_h, width,height,stride);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV12;
                tvInp.dataSize = 3*stride*height/2;
                break;        
            case 4:
                printf("yuv_reader nv21\n");
                width = 1080;
                height = 720;
                imgBuf = ucloud::yuv_reader(imagePath,width,height);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
                tvInp.dataSize = 3*width*height/2;
                break;
            case 5:
                printf("yuv_reader nv12\n");
                width = 1080;
                height = 720;
                imgBuf = ucloud::yuv_reader(imagePath,width,height);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV12;
                tvInp.dataSize = 3*width*height/2;  
                break;   
            case 6:
                printf("readImg_to_NV21_origin_size\n");
                imgBuf = ucloud::readImg_to_NV21(imagePath,width,height,stride);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
                tvInp.dataSize = 3*stride*height/2;
                break;         
            case 7:
                printf("readImg_to_RGB_origin_size\n");
                imgBuf = ucloud::readImg_to_RGB(imagePath,width,height);
                tvInp.format = TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
                tvInp.dataSize = width*height*3;
                stride = width;
                break;                     
            default:
                break;
            }

            if(imgBuf==nullptr){
                printf("no image is read\n");
                return -3;
            }

            // printf("image info: width %d, heights %d, stride %d\n", width, height, stride);
             
            tvInp.pData = imgBuf;
            tvInp.height = height;
            tvInp.width = width;
            tvInp.stride = stride;
            
            bboxes.clear();
            Tk.start();
            // if(use_roi)
            //     ret = ptrHandle->run(tvInp, roi ,bboxes, threshold);
            // else if(!use_roi && imlst.size()==1)
            //     ret = ptrHandle->run(tvInp, bboxes, threshold);
            // else
            ret = ptrHandle->run(tvInp,bboxes,imagePath,threshold,nms);
            
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
            if (imgBuf) free(imgBuf);
        }
    }
    printf("avg exec ptrHandle->run time = %f\n", avg_time/loop_times);

    if(img_mode <= 3 && imlst.size()==1){
        if(tvInp.format!=TVAI_IMAGE_FORMAT_BGR){
            // free(imgBuf);
            imgBuf = readImg_to_BGR(imlst[0],fmt_w, fmt_h, width,height);
        }
        drawImg(imgBuf, width, height, bboxes, true, true, false, 1);
        writeImg("result.jpg", imgBuf, width , height);
    }

    // if(imgBuf) free(imgBuf);
    return 0;
}
