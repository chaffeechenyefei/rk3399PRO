#ifndef _MODULE_BASE_HPP_
#define _MODULE_BASE_HPP_
#include <rknn_api.h>
#include "../libai_core.hpp"

#include <vector>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "drm/drm_func.h"
#include "drm/rga_func.h"


#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif

#define NMS_UNION 0
#define NMS_MIN 1
#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

/*******************************************************************************
 * BaseModel NPU(RK)基础推理模块
 * DESC: 主要负责各种模型推理的方式, 所有算法功能的基础模块
*******************************************************************************/
class BaseModel;

/*******************************************************************************
 * PreProcess_CPU_DRM_Model
 * DESC: CPU/DRM模式下图像前处理
*******************************************************************************/
class PreProcess_CPU_DRM_Model;

/*******************************************************************************
 * ImageUtil
 * DESC: DRM模式下图像前处理
*******************************************************************************/
class ImageUtil;

/*******************************************************************************
 * PreProcessModel 用类包了下各种函数, 可以换成namespace
 * DESC: CPU模式下图像前处理函数
*******************************************************************************/
class PreProcessModel;





typedef struct _DATA_SHAPE{
    int n;
    int c;
    int h;
    int w;
} DATA_SHAPE;

enum class MODEL_INPUT_FORMAT{
    //MLU
    BGRA = 0,
    RGBA = 1,
    ABGR = 2,
    ARGB = 3,
    //RKNN
    BGR,
    RGB,
};

//MLU
enum class MODEL_OUTPUT_ORDER{
    NCHW = 0,
    NHWC = 1,
};


/*******************************************************************************
 * BaseModel NPU(RK)基础推理模块
 * DESC: 主要负责各种模型推理的方式, 所有算法功能的基础模块
*******************************************************************************/
class BaseModel: public ucloud::AlgoAPI{
public:
    BaseModel(){};
    /*******************************************************************************
     * base_init
     * 读取模型权重文件, 并设置模型输入输出格式
    *******************************************************************************/
    ucloud::RET_CODE base_init(const std::string &modelpath, bool useDRM = false);
    ucloud::RET_CODE base_init(const unsigned char* modelBuf, int sizeBuf, bool useDRM = false);
    ucloud::RET_CODE base_init(ucloud::WeightData weightConfig, bool useDRM = false);
    virtual ~BaseModel();
    /*******************************************************************************
     * release
     * 析构函数的具体实现
    *******************************************************************************/    
    virtual void release();
    /*******************************************************************************
     * general_infer_uint8_nhwc_to_float 
     * 多输入多输出推理接口, 多输入多输出指单个模型输入/输出多个Tensor, 每个Tensor [N C H W]
     * 不是多输入不是BatchSize的意思
     * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free; MIMO
     * PARAM:
     *  input_datas: NHWC, UINT8
     *  input_shapes: DATA_SHAPE, 只有m_isMap模式下有效, 图像resize通过drm实现, 需要获取输入数据的形状
    *******************************************************************************/  
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float( 
        std::vector<unsigned char*> &input_datas,
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理
    
    /*******************************************************************************
     * WARNING: 内部测试 测试内存池管理(已测试通过,但需要进一步优化)
     * general_infer_uint8_nhwc_to_float_mem [tested]
     * DESC: 输出数据的指针内存由外部预先开辟; MIMO
     * PARAM:
     *  input_datas: NHWC, UINT8
     *  output_datas: float, 指针空间由外部负责
    *******************************************************************************/  
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float_mem( 
        std::vector<unsigned char*> &input_datas, 
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理

    /*******************************************************************************
     * WARNING: 内部测试 测试内基于drm的fast resize功能, 由于嵌入式动态库不支持input map, 所以没有测试
     * general_infer_uint8_nhwc_to_float []
     * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free; SIMO
     * PARAM:
     *  input_img: NHWC, UINT8
     *  input_shape: DATA_SHAPE, 只有m_isMap模式下有效, 图像resize通过drm实现, 需要获取输入数据的形状
    *******************************************************************************/      
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float( 
        cv::Mat &input_img,
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理    

    /*******************************************************************************
     * 输入输出变量大小的获取
    *******************************************************************************/
    std::vector<DATA_SHAPE> get_output_shape();//不建议使用, 存在缺陷
    DATA_SHAPE get_output_shape(int index);//不建议使用, 存在缺陷
    std::vector<DATA_SHAPE> get_input_shape(); //可以使用
    DATA_SHAPE get_input_shape(int index); //可以使用
    /*******************************************************************************
     * 获得输入输出Tensor的total_size
     * get_output_elem_num
     * get_input_elem_num
     * RETURN: [ N*C*H*W , N*C*H*W, ...] 单输入/输出下, 仅[ N*C*H*W ]
    *******************************************************************************/    
    std::vector<int> get_output_elem_num();//return std::vector<rknn_tensor_attr> m_outputAttr.elem
    std::vector<int> get_input_elem_num();//return std::vector<rknn_tensor_attr> m_inputAttr.elem
    /*******************************************************************************
     * 获得输出Tensor的具体维度, 但是返回的shape是逆序排列(RK特有的C++/Python内存排布)
     * get_output_dims
     * RETURN: [ tensor0.shape, tensor1.shape, ... ]
     * 例如 [ N C H W ]的Tensor, 实际返回是 [W H C N]的shape
    *******************************************************************************/   
    std::vector<std::vector<int>> get_output_dims();//std::vector<rknn_tensor_attr> m_outputAttr.dims;


protected:
    unsigned char *load_model(const char *filename, int *model_size);
    DATA_SHAPE get_shape( rknn_tensor_attr& t );

    /*******************************************************************************
     * 模型加载后, 模型输入输出信息打印
    *******************************************************************************/ 
    void print_shape(DATA_SHAPE &t);
    void print_input_shape();
    void print_output_shape();
    void print_input_attr();
    void print_output_attr();

protected:
    rknn_context m_ctx = 0;
    rknn_tensor_mem m_inMem[1];//mem信息反馈
    bool m_isMap = false; //是否通过使用map加速, 需要配合drm
    std::vector<DATA_SHAPE> m_inputShape;
    std::vector<DATA_SHAPE> m_outputShape;//有可能存在C=0的情况, 需要具体分析, 是否可以在模型侧解决. 所有只有elem_num是可靠的计数方式
    std::vector<rknn_tensor_attr> m_inputAttr;
    std::vector<rknn_tensor_attr> m_outputAttr;

    

private://DRM drm模式需要mutex保护
    void *drm_buf = nullptr;
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int drm_handle;
    size_t drm_actual_size = 0;
    rga_context rga_ctx;
    drm_context drm_ctx;
    DATA_SHAPE drm_Shape={0,0,0,0};//当前Shape
};

/*******************************************************************************
 * ImageUtil
 * DESC: DRM模式下图像前处理
*******************************************************************************/
/**
 * drm for preprocess
 * https://blog.csdn.net/u014644466/article/details/124568216
 */
typedef struct _PRE_PARAM{
    bool keep_aspect_ratio;
    bool pad_both_side;
    DATA_SHAPE model_input_shape;
    MODEL_INPUT_FORMAT model_input_format;
} PRE_PARAM;
class ImageUtil{
public:
    ImageUtil(){};
    virtual ~ImageUtil() { release(); };
    // ucloud::RET_CODE init(int w, int h, int channels);
    ucloud::RET_CODE init(ucloud::TvaiImage &tvimage);
    /**
     * dstPtr需要在外部实现开辟
     * tvimage 支持 RGB/BGR/nv21/nv12
     * dst_fmt 支持 RGB/BGR
     */
    ucloud::RET_CODE resize(ucloud::TvaiImage &tvimage, PRE_PARAM pre_param,void *dstPtr);
    ucloud::RET_CODE resize(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,PRE_PARAM pre_param,void *dstPtr);
    /** 弃用
     * dstPtr需要在外部实现开辟
     * src必须是RGB或者BGR 输出同样format
     */
    // ucloud::RET_CODE resize(const cv::Mat &src, const cv::Size &size, void *dstPtr);
    // ucloud::RET_CODE resize(ucloud::TvaiImage &tvimage, DATA_SHAPE dst_size,void *dstPtr);
protected:
    void *drm_buf = NULL;
    int drm_fd = -1;
    int buf_fd = -1;  // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    rga_context rga_ctx;
    drm_context drm_ctx;
    int W = -1;
    int H = -1;
    int C = -1;
    bool initialed = false;
    std::string dl_drm_path = "/usr/lib/aarch64-linux-gnu/libdrm.so";
    std::string dl_rga_path = "/usr/lib/aarch64-linux-gnu/librga.so";
    const int wstep = 4;
    const int hstep = 1;

    void release(void);
    //仅返回支持的nv21/nv12/rgb->rgb
    RGA_MODE get_rga_mode(ucloud::TvaiImageFormat inputFMT, MODEL_INPUT_FORMAT outputFMT, bool &channel_reorder );

};


/*******************************************************************************
 * PreProcess_CPU_DRM_Model
 * DESC: CPU/DRM模式下图像前处理
*******************************************************************************/
class PreProcess_CPU_DRM_Model{
public:
    PreProcess_CPU_DRM_Model(){
        m_drm = std::make_shared<ImageUtil>();
    }
    ~PreProcess_CPU_DRM_Model(){}

    /***whole image preprocess with drm**/
    virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage& tvimage, PRE_PARAM pre_param ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    /***whole image preprocess with opencv**/
    virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage& tvimage, PRE_PARAM pre_param ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    /***image with preprocess with roi+drm**/
    virtual ucloud::RET_CODE preprocess_drm(ucloud::TvaiImage& tvimage , ucloud::TvaiRect roi, PRE_PARAM pre_param ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);
    /***image preprocess with roi+opencv**/
    virtual ucloud::RET_CODE preprocess_opencv(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi, PRE_PARAM pre_param ,std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY);

protected:
    std::shared_ptr<ImageUtil> m_drm = nullptr;

};

/*******************************************************************************
 * PreProcessModel 用类包了下各种函数, 可以换成namespace
 * DESC: CPU模式下图像前处理函数
*******************************************************************************/
class PreProcessModel{
public:
    PreProcessModel(){}
    // static void set(DATA_SHAPE inputSp, MODEL_INPUT_FORMAT inputFmt,bool keep_aspect_ratio, bool pad_both_side);
    ~PreProcessModel(){}

    /*- image with roi and basic atomic param -----------------------------------*/
    static ucloud::RET_CODE preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst, 
        DATA_SHAPE dstSp, MODEL_INPUT_FORMAT dstFmt, float& aspect_ratio_x, float& aspect_ratio_y,
        bool keep_aspect_ratio=false, bool pad_both_side = false);
    /*- image with roi and PRE_PARAM -----------------------------------*/
    static ucloud::RET_CODE preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst, PRE_PARAM& config,
        float& aspect_ratio_x, float& aspect_ratio_y);


    /*- tvimage should be rgb only!!! -----------------------------------*/
    static ucloud::RET_CODE preprocess_rgb_subpixel(cv::Mat &InputRGB, std::vector<cv::Rect>& rois, std::vector<cv::Mat> &dst, 
        DATA_SHAPE dstSp, MODEL_INPUT_FORMAT dstFmt, std::vector<float>& aspect_ratio_x, std::vector<float>& aspect_ratio_y,
        bool keep_aspect_ratio=false, bool pad_both_side = false, bool use_subpixel=true);
    /*- do color changes only!!! -----------------------------------*/    
    static ucloud::RET_CODE preprocess_all_to_rgb(ucloud::TvaiImage &tvimage, cv::Mat &dst);
    /*- combine  preprocess_all_to_rgb + preprocess_rgb_subpixel -----------------------------------*/    
    static ucloud::RET_CODE preprocess_subpixel(ucloud::TvaiImage &tvimage, std::vector<cv::Rect> rois, std::vector<cv::Mat> &dst, PRE_PARAM& config,
        std::vector<float>& aspect_ratio_x, std::vector<float>& aspect_ratio_y, bool use_subpixel=true);

// private:
//     static bool m_keep_aspect_ratio;
//     static bool m_pad_both_side;
//     static DATA_SHAPE m_model_input_shape;
//     static MODEL_INPUT_FORMAT m_model_input_format;
};

class PostProcessModel{
public:
    PostProcessModel(){}
    ~PostProcessModel(){}
    static void base_output2ObjBox_multiCls(float* output ,std::vector<ucloud::VecObjBBox> &vecbox, 
        ucloud::CLS_TYPE* cls_map, std::map<ucloud::CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8);
};

template<typename T>
void base_nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output);
void base_nmsBBox(std::vector<ucloud::VecObjBBox> &input, float threshold, int type, ucloud::VecObjBBox &output);
/*******************************************************************************
 * base_transform_xyxy_xyhw 将模型输出尺度下输出的xyxy坐标 通过比例还原到原图尺寸, 同时根据expand_ratio保持中心的情况下放大框
*******************************************************************************/
template<typename T>
void base_transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio ,float aspect_ratio){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aspect_ratio);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aspect_ratio);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aspect_ratio;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aspect_ratio;
        float _x0 = cx - w/2;
        float _y0 = cy - h/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(w);
        vecbox[i].rect.height = int(h);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aspect_ratio;
            vecbox[i].Pts.pts[j].y /= aspect_ratio;
        }
    }
};
/*******************************************************************************
 * base_transform_xyxy_xyhw 将模型输出尺度下输出的xyxy坐标 通过比例还原到原图尺寸, 同时根据expand_ratio保持中心的情况下放大框
*******************************************************************************/
template<typename T>
void base_transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio ,float aX, float aY){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aX);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aY);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aX;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aY;
        float _x0 = cx - w/2;
        float _y0 = cy - h/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(w);
        vecbox[i].rect.height = int(h);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aX;
            vecbox[i].Pts.pts[j].y /= aY;
        }
    }
};

/*******************************************************************************
 * get_valid_rect 将rect限制在W,H的图像尺寸内
*******************************************************************************/
template<typename T>
inline T get_valid_rect(T rect, int W, int H){
    if(rect.x<0)
        rect.x = 0;
    if(rect.y<0)
        rect.y = 0;
    if(rect.width<0)
        rect.width = 0;
    if(rect.height<0)
        rect.height = 0;
    if(rect.x+rect.width>=W)
        rect.width = W - rect.x - 1;
    if(rect.y+rect.height>=H)
        rect.height = H - rect.y - 1;
    return rect;
};


/*******************************************************************************
 * globalscaleTvaiRect 将rect根据scale放大, 同时不超过W,H的图像尺寸
*******************************************************************************/
ucloud::TvaiRect globalscaleTvaiRect(ucloud::TvaiRect &rect, float scale, int W, int H);

/*******************************************************************************
 * shift_box_from_roi_to_org 将roi坐标下的bbox, 还原成原图坐标下的bbox
*******************************************************************************/
void shift_box_from_roi_to_org(ucloud::VecObjBBox &bboxes, ucloud::TvaiRect &roirect);


#endif