#ifndef _MODULE_BASE_HPP_
#define _MODULE_BASE_HPP_
#include <rknn_api.h>
#include "../libai_core.hpp"

#include <vector>
#include <glog/logging.h>
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
    BGRA = 0,
    RGBA = 1,
    ABGR = 2,
    ARGB = 3,
};

enum class MODEL_OUTPUT_ORDER{
    NCHW = 0,
    NHWC = 1,
};

class BaseModel: public ucloud::AlgoAPI{
public:
    BaseModel(){};
    /**
     * 读取模型权重文件, 并设置模型输入输出格式
     * */
    ucloud::RET_CODE base_init(const std::string &modelpath);
    virtual ~BaseModel();
    /**
     * general_infer_uint8_nhwc_to_float [tested]
     * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free
     * PARAM:
     *  input_datas: NHWC, UINT8
     */
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float( 
        std::vector<unsigned char*> &input_datas, 
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理
    /**
     * general_infer_uint8_nhwc_to_float_mem [tested]
     * DESC: 输出数据的指针内存由外部预先开辟
     * PARAM:
     *  input_datas: NHWC, UINT8
     */
    virtual ucloud::RET_CODE general_infer_uint8_nhwc_to_float_mem( 
        std::vector<unsigned char*> &input_datas, 
        std::vector<float*> &output_datas);//设置virtual, 兼容其他方式的rknn处理
    virtual void release();

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
    std::vector<DATA_SHAPE> m_inputShape;
    std::vector<DATA_SHAPE> m_outputShape;//有可能存在C=0的情况, 需要具体分析, 是否可以在模型侧解决. 所有只有elem_num是可靠的计数方式
    std::vector<rknn_tensor_attr> m_inputAttr;
    std::vector<rknn_tensor_attr> m_outputAttr;

};

#endif