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
// #include "utils/module_base.hpp"
#include "libai_core.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include <opencv2/opencv.hpp>

// #include "rknn_api.h"
#include <rknn_api.h>
#include <sstream>

using namespace std;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

static int rknn_GetTop(
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum)
{
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) ||
                (i == *(pMaxClass + 3)) || (i == *(pMaxClass + 4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
{
    //image load RGB
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)//rknn_tensor_attr ?????? dims ???????????????rknn_toolkit????????????numpy???????????????
    {
    case RKNN_TENSOR_NHWC://nhwc -> (c,w,h,n)
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[1];
        req_channel = input_attr->dims[0];
        break;
    case RKNN_TENSOR_NCHW://nchw -> (w,h,c,n)
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[0];
        req_channel = input_attr->dims[2];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

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



const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};
const int PROP_BOX_SIZE = 6;

static int process_fp(float *input, int *anchor, int grid_h, int grid_w, std::string &filename)
{   
    int grid_len = grid_h * grid_w;
    std::ofstream file;
    file.open(filename,std::ios::trunc);

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                file<<input[PROP_BOX_SIZE*a*grid_len+i*grid_w+j] << " ";
            }
            file<<"\n";
        }
    }
    file.close();
    return 1;
}




/*-------------------------------------------
                  Main Function
-------------------------------------------*/
// ./test_resnet {model_path} {image_path}
int main(int argc, char **argv)
{
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    ifstream file;
    file.open(img_path);
    if (!file) {
        std::cout<<"file "<<img_path<<" cant open!\n"<<std::endl;
    }
    string line,filename;
    vector<string> filelist;
    while(getline(file,line)){
        stringstream is(line);
        while(is>>filename){
            filelist.push_back(filename);
            // std::cout<<"filename is "<< filename << "\n";
        }
        // std::cout<<std::endl;
    }
    // printf("filelist has %d images\n",filelist.size());

    // Load RKNN Model
    model = load_model(model_path, &model_len);

    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    // Load image
    // std::shared_ptr<PreProcess_CPU_DRM_Model> im_utils = std::make_shared<PreProcess_CPU_DRM_Model>();
    // PRE_PARAM param_img2tensor;
    // param_img2tensor.keep_aspect_ratio = true;//???????????????, opencv??????, drm??????
    // param_img2tensor.pad_both_side = false;//???????????????(??????)??????, drm??????
    // param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//?????????RGB??????
    // param_img2tensor.model_input_shape.w = 256;
    // param_img2tensor.model_input_shape.h = 256;
    for (int m = 0;m<filelist.size();m++){
        string impath = filelist[m];
        // printf("imagename %s",impath.c_str());
        string respath = "./cls_layer2.txt";
        // respath = respath.replace(respath.find("."),4,".txt");
        // ofstream out;
        // out.open(respath,ios::trunc);
        printf("imagename %s",impath.c_str());
        unsigned char *input_data = NULL;
        input_data = load_image(impath.c_str(), &input_attrs[0]);

        if (!input_data)
        {
            return -1;
        }

        // Set Input Data
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = input_attrs[0].size;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input_data;

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }

        // Run
        printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        for (int i=0;i<3;i++){
            outputs[i].want_float=1;
        }
        // outputs[0].want_float = 1;
         printf("rknn_outputs num %d\n", io_num.n_output);
        ret = rknn_outputs_get(ctx, 3, outputs, NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        
        
        string savepath;
        int grid_h,grid_w;
        int channels =18;
        for (int i=0;i<3;i++){
            cout<<"step into record!"<<endl;
            switch  (i){
                case 0:
                    grid_h = 92;
                    grid_w = 52;
                    savepath = "./layer0_unshape.txt";
                    break;
                case 1:
                    grid_h = 46;
                    grid_w = 26;
                    savepath = "./layer1_unshape.txt";
                    break;
                case 2:
                    grid_h = 23;
                    grid_w = 13;
                    savepath ="./layer2_unshape.txt"; 
                    break;
            }
            std::ofstream file;
            file.open(savepath,ios::trunc);
            int grid_len = grid_h*grid_w;
            float *input = (float *)outputs[i].buf;
            for (int c=0;c<channels;c++){
                for(int h = 0;h<grid_h;h++){
                    for (int w=0;w<grid_w;w++){
                        int idx = c*grid_len + h*grid_w + w;
                        file<<input[idx]<<" ";
                    }
                    file<<"\n";
                }
            }
            file.close();
        }

        int stride = 32;
        grid_h = 13;
        grid_w = 23;
        int ret = process_fp((float *)outputs[2].buf,(int *)anchor2,grid_h,grid_w,respath);





        // Post Process
        // for (int i = 0; i < io_num.n_output; i++)
        // {
        //     // uint32_t MaxClass[5];
        //     // float fMaxProb[5];
        //     float *buffer = (float *)outputs[i].buf;
        //     uint32_t sz = outputs[i].size / 4;
        //     out<<buffer[0]<<" "<<buffer[1]<<" "<<buffer[2];

        //     // printf("output size = %d\n", sz);
        //     // printf("[ %f, %f ] \n", buffer[0], buffer[1]);
        // }
        // out.close();
        // Release rknn_outputs
        rknn_outputs_release(ctx, 3, outputs);
        if (input_data)
        {
            stbi_image_free(input_data);
        }
    }

    // Release
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
    if (model)
    {
        free(model);
    }

    // if (input_data)
    // {
    //     stbi_image_free(input_data);
    // }


    return 0;
}
