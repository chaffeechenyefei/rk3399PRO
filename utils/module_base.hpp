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

/**
 * BaseModel
 * DESC: 主要负责各种模型推理的方式
 */
class BaseModel: public ucloud::AlgoAPI{
public:
    BaseModel(){};
    /**
     * 读取模型权重文件, 并设置模型输入输出格式
     * */
    ucloud::RET_CODE base_init(const std::string &modelpath, bool useDRM = false);
    virtual ~BaseModel();
    virtual void release();
    /**
     * general_infer_uint8_nhwc_to_float [tested]
     * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free; MIMO
     * PARAM:
     *  input_datas: NHWC, UINT8
     *  input_shapes: DATA_SHAPE, 只有m_isMap模式下有效, 图像resize通过drm实现, 需要获取输入数据的形状
     */
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float( 
        std::vector<unsigned char*> &input_datas,
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理
    /**
     * general_infer_uint8_nhwc_to_float_mem [tested]
     * DESC: 输出数据的指针内存由外部预先开辟; MIMO
     * PARAM:
     *  input_datas: NHWC, UINT8
     *  output_datas: float, 指针空间由外部负责
     */
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float_mem( 
        std::vector<unsigned char*> &input_datas, 
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理

    /**
     * general_infer_uint8_nhwc_to_float []
     * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free; SIMO
     * PARAM:
     *  input_img: NHWC, UINT8
     *  input_shape: DATA_SHAPE, 只有m_isMap模式下有效, 图像resize通过drm实现, 需要获取输入数据的形状
     */
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float( 
        cv::Mat &input_img,
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理    


    

    std::vector<DATA_SHAPE> get_output_shape();
    DATA_SHAPE get_output_shape(int index);
    std::vector<DATA_SHAPE> get_input_shape();
    DATA_SHAPE get_input_shape(int index);
    std::vector<int> get_output_elem_num();
    std::vector<int> get_input_elem_num();


protected:
    unsigned char *load_model(const char *filename, int *model_size);
    DATA_SHAPE get_shape( rknn_tensor_attr& t );
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


/**
 * PreProcessModel
 * DESC: 主要负责各种输入转换的方式
 */
class PreProcessModel{
public:
    PreProcessModel(){}
    static void set(DATA_SHAPE inputSp, MODEL_INPUT_FORMAT inputFmt,bool keep_aspect_ratio, bool pad_both_side);
    ~PreProcessModel(){}
    static ucloud::RET_CODE preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst, 
        DATA_SHAPE dstSp, MODEL_INPUT_FORMAT dstFmt, float& aspect_ratio_x, float& aspect_ratio_y,
        bool keep_aspect_ratio=false, bool pad_both_side = false);
    static ucloud::RET_CODE preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst,
        float& aspect_ratio_x, float& aspect_ratio_y);

private:
    static bool m_keep_aspect_ratio;
    static bool m_pad_both_side;
    static DATA_SHAPE m_model_input_shape;
    static MODEL_INPUT_FORMAT m_model_input_format;
};


#endif