#include "module_base.hpp"
#include <assert.h>

using namespace ucloud;
using namespace std;


void BaseModel::release(){
    LOGI << "-> BaseModel::release";
    if (m_ctx > 0){
        rknn_destroy(m_ctx);
        m_ctx = 0;
    }
    m_inputShape.clear();
    m_outputShape.clear();
    m_inputAttr.clear();
    m_outputAttr.clear();
    LOGI << "<- BaseModel::release";
}

BaseModel::~BaseModel(){
    this->release();
}

std::vector<DATA_SHAPE> BaseModel::get_output_shape(){
    return m_outputShape;
}

DATA_SHAPE BaseModel::get_output_shape(int index){
    index = std::min( (int)m_outputShape.size()-1, std::max(0,index) );
    return m_outputShape[index];
}

std::vector<DATA_SHAPE> BaseModel::get_input_shape(){
    return m_inputShape;
}

DATA_SHAPE BaseModel::get_input_shape(int index){
    index = std::min( (int)m_inputShape.size()-1, std::max(0,index) );
    return m_inputShape[index]; 
}

std::vector<int> BaseModel::get_output_elem_num(){
    vector<int> ele_vec;
    for(auto &&t: m_outputAttr){
        ele_vec.push_back(t.n_elems);
    }
    return ele_vec;
}
std::vector<int> BaseModel::get_input_elem_num(){
    vector<int> ele_vec;
    for(auto &&t: m_inputAttr){
        ele_vec.push_back(t.n_elems);
    }
    return ele_vec;
}

DATA_SHAPE BaseModel::get_shape( rknn_tensor_attr& t ){
    DATA_SHAPE tSp;
    switch (t.fmt)//rknn_tensor_attr 中的 dims 数组顺序与rknn_toolkit的获取的numpy的顺序相反
    {
    case RKNN_TENSOR_NHWC://nhwc -> (c,w,h,n)
        tSp.h = t.dims[2];
        tSp.w = t.dims[1];
        tSp.c = t.dims[0];
        tSp.n = 1;
        break;
    case RKNN_TENSOR_NCHW://nchw -> (w,h,c,n)
        tSp.h = t.dims[1];
        tSp.w = t.dims[0];
        tSp.c = t.dims[2];
        tSp.n = 1;
        break;
    default:
        break;
    }
    return tSp;
}

static const char* get_type_string(rknn_tensor_type type)
{
    switch(type) {
    case RKNN_TENSOR_FLOAT32: return "FP32";
    case RKNN_TENSOR_FLOAT16: return "FP16";
    case RKNN_TENSOR_INT8: return "INT8";
    case RKNN_TENSOR_UINT8: return "UINT8";
    case RKNN_TENSOR_INT16: return "INT16";
    default: return "UNKNOW";
    }
}

static const char* get_qnt_type_string(rknn_tensor_qnt_type type)
{
    switch(type) {
    case RKNN_TENSOR_QNT_NONE: return "NONE";
    case RKNN_TENSOR_QNT_DFP: return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "AFFINE";
    default: return "UNKNOW";
    }
}

static const char* get_format_string(rknn_tensor_format fmt)
{
    switch(fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "UNKNOW";
    }
}
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
void BaseModel::print_input_attr(){
    printf("===print_input_attr===\n");
    for(auto &&t: m_inputAttr){
        dump_tensor_attr(&t);
    }
    printf("=======================\n");
}
void BaseModel::print_output_attr(){
    printf("===print_output_attr===\n");
    for(auto &&t: m_outputAttr){
        dump_tensor_attr(&t);
    }
    printf("=======================\n");
}

void BaseModel::print_shape(DATA_SHAPE &t){
    printf("(n,c,h,w) = [%d,%d,%d,%d] \n", t.n, t.c, t.h, t.w );
}

void BaseModel::print_input_shape(){
    printf("===print_input_shape===\n");
    for(auto &&t: m_inputShape){
        print_shape(t);
    }
    printf("=======================\n");
}

void BaseModel::print_output_shape(){
    printf("===print_output_shape===\n");
    for(auto &&t: m_outputShape){
        print_shape(t);
    }
    printf("=======================\n");
}

unsigned char * BaseModel::load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return nullptr;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

ucloud::RET_CODE BaseModel::base_init(const std::string &modelpath){
    LOGI << "-> BaseModel::base_init";
    release();
    unsigned char *model = nullptr;
    int model_len = 0;
    int ret;
    model = load_model(modelpath.c_str(), &model_len);
    if(model == nullptr )
        return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
    ret = rknn_init(&m_ctx, model, model_len, 0);
    if(model) free(model);
    if(ret < 0){
        LOGI << "npu initial failed code: " << ret;
        return RET_CODE::ERR_NPU_INIT_FAILED;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query( m_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        LOGI << "rknn_query fail! ret=" << ret;
        return RET_CODE::ERR_NPU_QUERY_FAILED;
    } else{
        LOGI << "model input num = " << io_num.n_input << ", output num = " << io_num.n_output;
    }   
    // input tensor
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset( input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query( m_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOGI << "rknn_query fail! ret= " << ret;
            return RET_CODE::ERR_NPU_QUERY_FAILED;
        }
        m_inputShape.push_back(get_shape(input_attrs[i]));
        m_inputAttr.push_back(input_attrs[i]);
    }
    print_input_shape();
    //output tensor::此时的tensor output是非float的, want_float后size会变,ele_num不变
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query( m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOGI << "rknn_query fail! ret= " <<  ret;
            return RET_CODE::ERR_NPU_QUERY_FAILED;
        }
        m_outputShape.push_back(get_shape(output_attrs[i]));
        m_outputAttr.push_back(output_attrs[i]);
    }
    print_output_shape();
    print_input_attr();
    print_output_attr();
    LOGI << "<- BaseModel::base_init";
    return RET_CODE::SUCCESS;
}

RET_CODE BaseModel::general_infer_uint8_nhwc_to_float(std::vector<unsigned char*> &input_datas, std::vector<float*> &output_datas){
    LOGI << "-> BaseModel::general_infer_uint8_nhwc_to_float";
    // Set Input Data
    int ret = -1;
    assert( input_datas.size() == m_inputAttr.size() );
    // rknn_input* inputs = (rknn_input*)malloc(input_datas.size()*sizeof(rknn_input));
    // memset(inputs, 0, input_datas.size()*sizeof(rknn_input));
    //采用数组,自动free,且内部指针不需要负责释放
    rknn_input inputs[input_datas.size()];
    memset(inputs, 0 , sizeof(inputs)); //初始化结构体, 0 = False

    for(int i=0; i < input_datas.size(); i++ ){
        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_UINT8;
        inputs[i].size = m_inputAttr[i].size;
        inputs[i].fmt = RKNN_TENSOR_NHWC;//模型内部都是使用的NCHW, 输入设置NHWC是为了方便图片输入
        inputs[i].buf = input_datas[i];
    }
    // rknn_tensor_mem
    ret = rknn_inputs_set(m_ctx, m_inputAttr.size(), inputs);
    if (ret < 0)
    {
        LOGI << "rknn_input_set fail! ret = " << ret;
        // free(inputs);
        return RET_CODE::ERR_NPU_IOSET_FAILED;
    }

    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret < 0)
    {
        LOGI << "rknn_run fail! ret = " << ret;
        // free(inputs);
        return RET_CODE::ERR_NPU_RUN_FAILED;
    }
    // if(inputs!=nullptr) free(inputs);

    // Get Output 有is_prealloc参数,可以通过内存池减少开辟, 通过memset 0, 默认关闭该功能, 即buf ptr的使用权在rknn系统, 需要copy data
    // rknn_output* outputs = (rknn_output*)malloc(m_outputAttr.size()*sizeof(rknn_output));
    // memset(outputs, 0, m_outputAttr.size()*sizeof(rknn_output));
    rknn_output outputs[m_outputAttr.size()];
    memset(outputs, 0 , sizeof(outputs));//初始化结构体, 0 = False
    for(int i = 0; i < m_outputAttr.size(); i++ ){
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(m_ctx, m_outputAttr.size(), outputs, NULL);
    if (ret < 0)
    {
        LOGI << "rknn_outputs_get fail! ret = " << ret;
        // free(outputs);
        return RET_CODE::ERR_NPU_GET_OUTPUT_FAILED;
    }

    // Trans output result to output_datas
    for(int i=0; i < m_outputAttr.size(); i++){
        // int hwc = m_outputShape[i].h* m_outputShape[i].w * m_outputShape[i].c;
        // printf("outputs size = %d, hwc = %d \n", outputs[i].size ,hwc);//assert 4*hwc == output size
        float* tmp = (float*)malloc(outputs[i].size);//移交出去的指针
        memcpy(tmp, outputs[i].buf, outputs[i].size);
        output_datas.push_back(tmp);
    }

    // Release
    rknn_outputs_release(m_ctx, m_outputAttr.size(), outputs);
    // if(outputs!=nullptr) free(outputs);
    LOGI << "<- BaseModel::general_infer_uint8_nhwc_to_float";
    return RET_CODE::SUCCESS;
}


ucloud::RET_CODE BaseModel::general_infer_uint8_nhwc_to_float_mem( 
    std::vector<unsigned char*> &input_datas, std::vector<float*> &output_datas)
{
    LOGI << "-> BaseModel::general_infer_uint8_nhwc_to_float_mem";
    // Set Input Data
    int ret = -1;
    assert( input_datas.size() == m_inputAttr.size() );
    // rknn_input* inputs = (rknn_input*)malloc(input_datas.size()*sizeof(rknn_input));
    // memset(inputs, 0, input_datas.size()*sizeof(rknn_input));
    //采用数组,自动free,且内部指针不需要负责释放
    rknn_input inputs[input_datas.size()];
    memset(inputs, 0 , sizeof(inputs)); //初始化结构体, 0 = False

    for(int i=0; i < input_datas.size(); i++ ){
        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_UINT8;
        inputs[i].size = m_inputAttr[i].size;
        inputs[i].fmt = RKNN_TENSOR_NHWC;//模型内部都是使用的NCHW, 输入设置NHWC是为了方便图片输入
        inputs[i].buf = input_datas[i];
    }
    // rknn_tensor_mem
    ret = rknn_inputs_set(m_ctx, m_inputAttr.size(), inputs);
    if (ret < 0)
    {
        LOGI << "rknn_input_set fail! ret = " << ret;
        // free(inputs);
        return RET_CODE::ERR_NPU_IOSET_FAILED;
    }

    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret < 0)
    {
        LOGI << "rknn_run fail! ret = " << ret;
        // free(inputs);
        return RET_CODE::ERR_NPU_RUN_FAILED;
    }
    // if(inputs!=nullptr) free(inputs);

    // Get Output 有is_prealloc参数,可以通过内存池减少开辟, 通过memset 0, 默认关闭该功能, 即buf ptr的使用权在rknn系统, 需要copy data
    // rknn_output* outputs = (rknn_output*)malloc(m_outputAttr.size()*sizeof(rknn_output));
    // memset(outputs, 0, m_outputAttr.size()*sizeof(rknn_output));
    rknn_output outputs[m_outputAttr.size()];
    // memset(outputs, 0 , sizeof(outputs));//初始化结构体, 0 = False
    for(int i = 0; i < m_outputAttr.size(); i++ ){
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = true;
        outputs[i].index = i;
        outputs[i].buf = reinterpret_cast<void*>(output_datas[i]);
        outputs[i].size = m_outputAttr[i].n_elems*sizeof(float);
    }
    ret = rknn_outputs_get(m_ctx, m_outputAttr.size(), outputs, NULL);
    if (ret < 0)
    {
        LOGI << "rknn_outputs_get fail! ret = " << ret;
        // free(outputs);
        return RET_CODE::ERR_NPU_GET_OUTPUT_FAILED;
    }

    // // Trans output result to output_datas
    // for(int i=0; i < m_outputAttr.size(); i++){
    //     // int hwc = m_outputShape[i].h* m_outputShape[i].w * m_outputShape[i].c;
    //     // printf("outputs size = %d, hwc = %d \n", outputs[i].size ,hwc);//assert 4*hwc == output size
    //     float* tmp = (float*)malloc(outputs[i].size);//移交出去的指针
    //     memcpy(tmp, outputs[i].buf, outputs[i].size);
    //     output_datas.push_back(tmp);
    // }

    // Release
    rknn_outputs_release(m_ctx, m_outputAttr.size(), outputs);
    // if(outputs!=nullptr) free(outputs);
    LOGI << "<- BaseModel::general_infer_uint8_nhwc_to_float_mem";
    return RET_CODE::SUCCESS;
}