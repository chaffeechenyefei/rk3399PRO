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

#include "rga_func.h"

int RGA_init(rga_context *rga_ctx, const char* dlpath)
{   
    //"/usr/lib/aarch64-linux-gnu/librga.so"
    rga_ctx->rga_handle = dlopen( dlpath , RTLD_LAZY);
    if (!rga_ctx->rga_handle)
    {
        printf("dlopen %s failed\n",dlpath);
        return -1;
    }
    rga_ctx->init_func = (FUNC_RGA_INIT)dlsym(rga_ctx->rga_handle, "c_RkRgaInit");
    rga_ctx->deinit_func = (FUNC_RGA_DEINIT)dlsym(rga_ctx->rga_handle, "c_RkRgaDeInit");
    rga_ctx->blit_func = (FUNC_RGA_BLIT)dlsym(rga_ctx->rga_handle, "c_RkRgaBlit");
    rga_ctx->init_func();
    return 0;
}

void img_resize_fast(rga_context *rga_ctx, int src_fd, int src_w, int src_h, uint64_t dst_phys, int dst_w, int dst_h)
{
    // printf("rga use fd, src(%dx%d) -> dst(%dx%d)\n", src_w, src_h, dst_w, dst_h);

    if (rga_ctx->rga_handle)
    {
        int ret = 0;
        rga_info_t src, dst;

        memset(&src, 0, sizeof(rga_info_t));
        src.fd = src_fd;
        src.mmuFlag = 1;
        // src.virAddr = (void *)psrc;

        memset(&dst, 0, sizeof(rga_info_t));
        dst.fd = -1;
        dst.mmuFlag = 0;

#if defined(__arm__)
        dst.phyAddr = (void *)((uint32_t)dst_phys);
#else
        dst.phyAddr = (void *)dst_phys;
#endif

        dst.nn.nn_flag = 0;

        rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_RGB_888);
        rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);

        ret = rga_ctx->blit_func(&src, &dst, NULL);
        if (ret)
        {
            printf("c_RkRgaBlit error : %s\n", strerror(errno));
        }

        return;
    }
    return;
}

int img_resize_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h, void *dst_virt, int dst_w, int dst_h)
{
    // printf("rga use virtual, src(%dx%d) -> dst(%dx%d)\n", src_w, src_h, dst_w, dst_h);

    if (rga_ctx->rga_handle)
    {
        int ret = 0;
        rga_info_t src, dst;

        memset(&src, 0, sizeof(rga_info_t));
        src.fd = -1;
        src.mmuFlag = 1;
        src.virAddr = (void *)src_virt;

        memset(&dst, 0, sizeof(rga_info_t));
        dst.fd = -1;
        dst.mmuFlag = 1;
        dst.virAddr = dst_virt;

        dst.nn.nn_flag = 0;

        rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_RGB_888);
        rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);

        ret = rga_ctx->blit_func(&src, &dst, NULL);
        if (ret)
        {
            printf("c_RkRgaBlit error : %s\n", strerror(errno));
            return -1;
        }

        return 1;
    }
    return -2;
}

int RGA_deinit(rga_context *rga_ctx)
{
    if(rga_ctx->rga_handle)
    {
        dlclose(rga_ctx->rga_handle);
        rga_ctx->rga_handle = NULL;
    }
}


int img_resize_to_dst_format_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h, 
    void *dst_virt, int dst_w, int dst_h, RGA_MODE mode)
{
    if (rga_ctx->rga_handle)
    {
        int ret = 0;
        rga_info_t src, dst;

        memset(&src, 0, sizeof(rga_info_t));
        src.fd = -1;
        src.mmuFlag = 1;
        src.virAddr = (void *)src_virt;

        memset(&dst, 0, sizeof(rga_info_t));
        dst.fd = -1;
        dst.mmuFlag = 1;
        dst.virAddr = dst_virt;

        dst.nn.nn_flag = 0;

        switch (mode)
        {
        case RGBtoRGB: //rgb2rgb or bgr2bgr
        case BGRtoBGR:
            // printf("RGBtoRGB/BGRtoBGR\n");
            rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_RGB_888);
            rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
            break;
        // case RGBtoBGR: //rgb2bgr or bgr2rgb
        // case BGRtoRGB:
        //     // printf("RGBtoBGR/BGRtoRGB\n");
        //     rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_BGR_888);
        //     rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
        //     break;
        // case NV21toBGR: //bgr2nv21
        //     // printf("NV21toBGR\n");
        //     rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_YCrCb_420_SP);
        //     rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_BGR_888);
        //     break;
        // case NV12toBGR: //bgr2nv12
        //     // printf("NV12toBGR\n");
        //     rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_YCbCr_420_SP);
        //     rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_BGR_888);
        //     break;
        case NV21toRGB: //rgb2nv21
            // printf("NV21toRGB\n");
            rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_YCrCb_420_SP);
            rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
            break;
        case NV12toRGB: //rgb2nv12
            // printf("NV12toRGB\n");
            rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, RK_FORMAT_YCbCr_420_SP);
            rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
            break;    
        default:
            printf("mode unsupported\n");
            ret = -1;
            break;
        }
        if(ret==-1){
            return ret;
        }



        ret = rga_ctx->blit_func(&src, &dst, NULL);
        if (ret)
        {
            printf("c_RkRgaBlit error : %s\n", strerror(errno));
            return -1;
        }

        return 1;
    }
    return -2;
}