#ifndef __RGA_FUNC_H__
#define __RGA_FUNC_H__

#include <dlfcn.h> 
#include <RgaApi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int(* FUNC_RGA_INIT)();
typedef void(* FUNC_RGA_DEINIT)();
typedef int(* FUNC_RGA_BLIT)(rga_info_t *, rga_info_t *, rga_info_t *);

typedef struct _rga_context{
    void *rga_handle;
    FUNC_RGA_INIT init_func;
    FUNC_RGA_DEINIT deinit_func;
    FUNC_RGA_BLIT blit_func;
} rga_context;

typedef enum{
    RGBtoRGB,
    BGRtoBGR,
    RGBtoBGR,
    BGRtoRGB,
    NV21toRGB,
    NV21toBGR,
    NV12toRGB,
    NV12toBGR,
} RGA_MODE;

int RGA_init(rga_context* rga_ctx, const char* dlpath);

void img_resize_fast(rga_context *rga_ctx, int src_fd, int src_w, int src_h, uint64_t dst_phys, int dst_w, int dst_h);

int img_resize_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h, void *dst_virt, int dst_w, int dst_h);
/**
 * img_resize_to_dst_format_slow
 */
int img_resize_to_dst_format_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h, void *dst_virt, int dst_w, int dst_h, RGA_MODE mode);

int RGA_deinit(rga_context* rga_ctx);

#ifdef __cplusplus
}
#endif
#endif/*__RGA_FUNC_H__*/
