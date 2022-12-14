#include "module_base.hpp"
#include "basic.hpp"
#include <assert.h>
#include <fstream>

using namespace ucloud;
using namespace std;


void BaseModel::release(){
    LOGI << "-> BaseModel::release";
    if (m_ctx > 0){
#ifdef RKNN_1_7_X        
        if(m_isMap && !m_inputAttr.empty()){
            rknn_inputs_unmap(m_ctx, m_inputAttr.size(), m_inMem);
        }
#endif        
        rknn_destroy(m_ctx); m_ctx = 0;
#ifdef RKNN_1_7_X        
        if(m_isMap){
            // drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, drm_handle, drm_buf, drm_actual_size);
            if(drm_buf) {
                drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, drm_handle, drm_buf, drm_actual_size);
                drm_buf = nullptr;
            }
            drm_deinit(&drm_ctx, drm_fd);
            RGA_deinit(&rga_ctx);
        }
#endif        
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

std::vector<std::vector<int>> BaseModel::get_output_dims(){
    std::vector<std::vector<int>> output_dims;
    for(auto &&m: m_outputAttr){
        int nd = m.n_dims;
        std::vector<int> tmp;
        for(int i = 0; i < nd; i++ ){
            tmp.push_back(m.dims[i]);
        }
        output_dims.push_back(tmp);
    }
    return output_dims;
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
//rknn_tensor_attr 中的 dims 数组顺序与 rknn_toolkit 的获取的 numpy 的顺序 相反
static void dump_tensor_attr(rknn_tensor_attr *attr)
{   
    printf("index=%d, name=%s, n_dims=%d , n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims,
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    printf("dims[0->n] = [");
    for(int i=0; i<attr->n_dims; i++){
        printf("%d,", attr->dims[i]);
    }
    printf("]\n");
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
        fflush(stdout);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        fflush(stdout);
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


ucloud::RET_CODE BaseModel::base_init(ucloud::WeightData weightConfig, bool useDRM){
    return base_init(weightConfig.pData, weightConfig.size, useDRM);
}


ucloud::RET_CODE BaseModel::base_init(unsigned char* modelBuf, int sizeBuf, bool useDRM){
    LOGI << "-> BaseModel::base_init";
    release();
    // unsigned char *model = nullptr;
    int model_len = 0;
    int ret;
    // model = load_model(modelpath.c_str(), &model_len);
    if(modelBuf == nullptr || sizeBuf <= 0){
        printf("**[%s][%d] model buf size should not be %d\n", __FILE__, __LINE__, sizeBuf);
        return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
    }
    model_len = sizeBuf;
    ret = rknn_init(&m_ctx, modelBuf, model_len, 0);
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
    #ifdef PRINT_ATTR
    print_input_shape();
    #endif
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
    #ifdef PRINT_ATTR
    print_output_shape();
    print_input_attr();
    print_output_attr();
    #endif

    //是否采用map+drm的方式进行高效推理
    m_isMap = useDRM;
    if(m_isMap){
#ifdef RKNN_1_7_X        
        rknn_inputs_map(m_ctx, io_num.n_input , m_inMem );
        printf("input virt_addr = %p, phys_addr = 0x%llx, fd = %d, size = %d\n",
               m_inMem[0].logical_addr, m_inMem[0].physical_addr, m_inMem[0].fd,
               m_inMem[0].size);
        if (m_inMem[0].physical_addr == 0xffffffffffffffff){
            printf("get unvalid input physical address, please extend in/out memory space\n");
            return RET_CODE::ERR_NPU_MEM_ERR;
        }
        //DRM
        memset(&rga_ctx, 0, sizeof(rga_context));
        memset(&drm_ctx, 0, sizeof(drm_context));
        RGA_init(&rga_ctx, "/usr/lib/aarch64-linux-gnu/librga.so");
        drm_fd = drm_init(&drm_ctx, "/usr/lib/aarch64-linux-gnu/libdrm.so");
#endif        
    }
    LOGI << "<- BaseModel::base_init";
    // printf("BaseModel::base init success");
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE BaseModel::base_init(const std::string &modelpath, bool useDRM){
    LOGI << "-> BaseModel::base_init";
    release();
    unsigned char *model = nullptr;
    int model_len = 0;
    int ret;
    model = load_model(modelpath.c_str(), &model_len);
    if(model == nullptr )
        return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
    RET_CODE RET = BaseModel::base_init(model, model_len, useDRM);
    free(model);
    return RET;
}


/*******************************************************************************
 * general_infer_uint8_nhwc_to_uint8, 
 * 与general_infer_uint8_nhwc_to_float相对应, 输出变为uint8
*******************************************************************************/ 
ucloud::RET_CODE BaseModel::general_infer_uint8_nhwc_to_uint8( 
    std::vector<unsigned char*> &input_datas,
    std::vector<unsigned char*> &output_datas,   
    std::vector<float> &out_scales,
    std::vector<uint32_t> &out_zps)
{
    // return RET_CODE::FAILED;
    LOGI << "-> BaseModel::general_infer_uint8_nhwc_to_uint8";
    if(m_isMap) return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
    // Set Input Data
    int ret = -1;
    assert( input_datas.size() == m_inputAttr.size() );

    if(!m_isMap){
        /*正常情况下的使用*/
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
        if (ret != RKNN_SUCC)
        {
            printf("rknn_input_set fail! ret = %d", ret);
            fflush(stdout);
            // free(inputs);
            return RET_CODE::ERR_NPU_IOSET_FAILED;
        }
    } else {
        return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
    }
    
    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret != RKNN_SUCC )
    {
        printf("rknn_run fail! ret = %d",ret);
        fflush(stdout);
        // free(inputs);
        return RET_CODE::ERR_NPU_RUN_FAILED;
    }
    // if(inputs!=nullptr) free(inputs);
    rknn_output outputs[m_outputAttr.size()];
    memset(outputs, 0 , sizeof(outputs));//初始化结构体, 0 = False
    for (int i = 0; i < m_outputAttr.size(); i++)
    {
        if(m_outputAttr[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && m_outputAttr[i].type == RKNN_TENSOR_UINT8) continue;
        else outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(m_ctx, m_outputAttr.size(), outputs, NULL);
    if (ret != RKNN_SUCC )
    {
        printf("rknn_outputs_get fail! ret = %d",ret);
        fflush(stdout);
        // free(outputs);
        return RET_CODE::ERR_NPU_GET_OUTPUT_FAILED;
    }

    // Trans output result to output_datas
    for(int i=0; i < m_outputAttr.size(); i++){
        // int hwc = m_outputShape[i].h* m_outputShape[i].w * m_outputShape[i].c;
        // printf("outputs size = %d, hwc = %d \n", outputs[i].size ,hwc);//assert 4*hwc == output size
        uint8_t* tmp = (uint8_t*)malloc(outputs[i].size);//移交出去的指针
        memcpy(tmp, outputs[i].buf, outputs[i].size);
        output_datas.push_back(tmp);
        out_scales.push_back(m_outputAttr[i].scale);
        out_zps.push_back(m_outputAttr[i].zp);
    }

    // Release
    rknn_outputs_release(m_ctx, m_outputAttr.size(), outputs);
    // if(outputs!=nullptr) free(outputs);
    LOGI << "<- BaseModel::general_infer_uint8_nhwc_to_uint8";
    return RET_CODE::SUCCESS;
}

/*******************************************************************************
 * general_infer_uint8_nhwc_to_float 
 * 多输入多输出推理接口, 多输入多输出指单个模型输入/输出多个Tensor, 每个Tensor [N C H W]
 * 不是多输入不是BatchSize的意思
 * DESC: 输出数据的指针内存由函数内部malloc, 需要使用完毕后在外部free; MIMO
 * PARAM:
 *  input_datas: NHWC, UINT8
 *  input_shapes: DATA_SHAPE, 只有m_isMap模式下有效, 图像resize通过drm实现, 需要获取输入数据的形状
*******************************************************************************/  
RET_CODE BaseModel::general_infer_uint8_nhwc_to_float(
    std::vector<unsigned char*> &input_datas, 
    std::vector<float*> &output_datas)
{
    // return RET_CODE::FAILED;
    LOGI << "-> BaseModel::general_infer_uint8_nhwc_to_float";
    if(m_isMap) return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
    // Set Input Data
    int ret = -1;
    assert( input_datas.size() == m_inputAttr.size() );

    if(!m_isMap){
        /*正常情况下的使用*/
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
        if (ret != RKNN_SUCC)
        {
            printf("rknn_input_set fail! ret = %d", ret);
            fflush(stdout);
            // free(inputs);
            return RET_CODE::ERR_NPU_IOSET_FAILED;
        }
    } else {
        return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
    }
    
    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret != RKNN_SUCC )
    {
        printf("rknn_run fail! ret = %d",ret);
        fflush(stdout);
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
    if (ret != RKNN_SUCC )
    {
        printf("rknn_outputs_get fail! ret = %d",ret);
        fflush(stdout);
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
    if(m_isMap) return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
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
        // inputs[i].pass_through = 0; //add by lihui 2022-10-27
    }
    // rknn_tensor_mem
    ret = rknn_inputs_set(m_ctx, m_inputAttr.size(), inputs);
    if (ret != RKNN_SUCC )
    {
        LOGI << "rknn_input_set fail! ret = " << ret;
        // free(inputs);
        return RET_CODE::ERR_NPU_IOSET_FAILED;
    }

    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret != RKNN_SUCC )
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
    if (ret != RKNN_SUCC )
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

//SIMO
RET_CODE BaseModel::general_infer_uint8_nhwc_to_float(
    cv::Mat &input_img,
    std::vector<float*> &output_datas)
{
    LOGI << "-> BaseModel::general_infer_uint8_nhwc_to_float[drm]";
    if(!m_isMap) return RET_CODE::ERR_NPU_SYNC_NOT_MATCH;
    // Set Input Data
    int ret = -1;
    assert( 1 == m_inputAttr.size() );
    // size_t actual_size = 0;
    if(drm_buf==nullptr){
        drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, input_img.cols, input_img.rows, input_img.channels() * 8,
                            &buf_fd, &drm_handle, &drm_actual_size);
    } else if(drm_buf!=nullptr && (drm_Shape.w!=input_img.cols || drm_Shape.h != input_img.rows ) ){
        //第一次初始化时跳过, 之后如果size不对则需要重新开辟空间
        drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, drm_handle, drm_buf, drm_actual_size);
        drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, input_img.cols, input_img.rows, input_img.channels() * 8,
                            &buf_fd, &drm_handle, &drm_actual_size);
    }
    
    memcpy(drm_buf, input_img.data , input_img.total() * input_img.channels());
    
    img_resize_fast(&rga_ctx, buf_fd, input_img.cols, input_img.rows, m_inMem[0].physical_addr, m_inputShape[0].w, m_inputShape[0].h);    
#ifdef RKNN_1_7_X    
    rknn_inputs_sync(m_ctx, 1, m_inMem);
#else
    printf("rknn_inputs_sync not supported\n");
    return RET_CODE::ERR_NPU_GET_OUTPUT_FAILED;   
#endif    

    LOGI << "-> rknn_run";
    ret = rknn_run(m_ctx, nullptr);
    if (ret != RKNN_SUCC )
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
    if (ret != RKNN_SUCC )
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
    LOGI << "<- BaseModel::general_infer_uint8_nhwc_to_float[drm]";
    return RET_CODE::SUCCESS;
}

//static成员变量类外初始化
// bool PreProcessModel::m_keep_aspect_ratio = false;
// bool PreProcessModel::m_pad_both_side = false;
// DATA_SHAPE PreProcessModel::m_model_input_shape = {0,0,0,0};
// MODEL_INPUT_FORMAT PreProcessModel::m_model_input_format = MODEL_INPUT_FORMAT::RGB;

// void PreProcessModel::set(DATA_SHAPE inputSp, MODEL_INPUT_FORMAT inputFmt,bool keep_aspect_ratio, bool pad_both_side){
//     m_keep_aspect_ratio = keep_aspect_ratio;
//     m_pad_both_side = pad_both_side;
//     m_model_input_format = inputFmt;
//     m_model_input_shape = inputSp;
// }

RET_CODE PreProcessModel::preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst, PRE_PARAM& config,
        float& aspect_ratio_x, float& aspect_ratio_y)
{
    return preprocess(tvimage, roi, dst, 
            config.model_input_shape, config.model_input_format, aspect_ratio_x, aspect_ratio_y,
            config.keep_aspect_ratio, config.pad_both_side);
}

RET_CODE PreProcessModel::preprocess(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi,cv::Mat &dst, 
        DATA_SHAPE dstSp, MODEL_INPUT_FORMAT dstFmt, float& aspect_ratio_x, float& aspect_ratio_y, 
        bool keep_aspect_ratio, bool pad_both_side)
{
    RET_CODE ret = RET_CODE::SUCCESS;
    cv::Mat src, src_roi_rgb, src_roi, tmp, resized_roi;
    cv::Rect _roi = {roi.x, roi.y, roi.width, roi.height};
    /*--------------tvimage to src------------------*/
    switch (tvimage.format)
    {//TVAI_IMAGE_FORMAT_NV12,TVAI_IMAGE_FORMAT_NV21都先转RGB
    case TVAI_IMAGE_FORMAT_NV12:
        tmp = cv::Mat(cv::Size(tvimage.width,3*tvimage.height/2),CV_8UC1, tvimage.pData);
        cv::cvtColor(tmp, src, cv::COLOR_YUV2RGB_NV12);
        break;
    case TVAI_IMAGE_FORMAT_NV21:
        tmp = cv::Mat(cv::Size(tvimage.width,3*tvimage.height/2),CV_8UC1, tvimage.pData);
        cv::cvtColor(tmp, src, cv::COLOR_YUV2RGB_NV21);
        break;
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
        src = cv::Mat(cv::Size(tvimage.width,tvimage.height),CV_8UC3, tvimage.pData);
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        printf("tvimage.format enum[%d] is not supported\n", tvimage.format);
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    
    /*--------------src to src_roi------------------*/
    src_roi = src(_roi);
    /*--------------src_roi to src_roi_rgb------------------*/
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_BGR:
        cv::cvtColor(src_roi, src_roi_rgb, cv::COLOR_BGR2RGB);
        break;  
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        src_roi_rgb = src_roi;
        break;                               
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        printf("tvimage.format enum[%d] is not supported\n", tvimage.format);
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    /*--------------src_roi_rgb to resized_roi------------------*/
    if(keep_aspect_ratio){
        resized_roi = resize(src_roi_rgb, cv::Size(dstSp.w,dstSp.h), pad_both_side, aspect_ratio_x);
        aspect_ratio_y = aspect_ratio_x;
    } else {
        resized_roi = resize_no_aspect(src_roi_rgb, cv::Size(dstSp.w,dstSp.h), aspect_ratio_x, aspect_ratio_y);
    }

    /*--------------resized_roi(rgb) to dst------------------*/
    switch (dstFmt)
    {
    case MODEL_INPUT_FORMAT::RGB :
        dst = resized_roi;
        break;
    case MODEL_INPUT_FORMAT::BGR :
        cv::cvtColor(resized_roi, dst, cv::COLOR_RGB2BGR);
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        printf("model input format enum[%d] is not supported\n", dstFmt);
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;

    return ret;
}

ucloud::RET_CODE PreProcessModel::preprocess_rgb_subpixel(cv::Mat &InputRGB, std::vector<cv::Rect>& rois, std::vector<cv::Mat> &dst, 
    DATA_SHAPE dstSp, MODEL_INPUT_FORMAT dstFmt, std::vector<float>& aspect_ratio_x, std::vector<float>& aspect_ratio_y,
    bool keep_aspect_ratio, bool pad_both_side, bool use_subpixel)
{
    LOGI << "-> PreProcessModel::preprocess_rgb_subpixel";
    RET_CODE ret = RET_CODE::SUCCESS;
    cv::Mat sub_cvimage, resized_roi ,target_cvimage;
    for(auto &&roi: rois){
        if(roi.x == 0 && roi.y == 0 && roi.width == InputRGB.cols && roi.height == InputRGB.rows ){
            //roi即原图大小
            sub_cvimage = InputRGB;
        } else {
            if (use_subpixel){
                getRectSubPix(InputRGB, cv::Size(roi.width, roi.height), 
                    cv::Point2f(float(roi.x + (1.0*roi.width)/2), float(roi.y + (1.0*roi.height)/2)) , sub_cvimage);
            }else{
                int x = clip<int>(roi.x, 0, InputRGB.cols-10);
                int y = clip<int>(roi.y, 0, InputRGB.rows-10);
                int w = clip<int>(roi.width,1, InputRGB.cols-x);
                int h = clip<int>(roi.height,1, InputRGB.rows-y);
                InputRGB(cv::Rect(x,y,w,h)).copyTo(sub_cvimage);
            }
        }

        float aX=1.0; float aY=1.0;
        if(keep_aspect_ratio){
            resized_roi = resize(sub_cvimage, cv::Size(dstSp.w,dstSp.h), pad_both_side, aX);
            aspect_ratio_x.push_back(aX);
            aspect_ratio_y.push_back(aX);
        } else {
            resized_roi = resize_no_aspect(sub_cvimage, cv::Size(dstSp.w,dstSp.h), aX, aY);
            aspect_ratio_x.push_back(aX);
            aspect_ratio_y.push_back(aY);
        }
        switch (dstFmt)
        {
        case MODEL_INPUT_FORMAT::RGB :
            target_cvimage = resized_roi;
            break;
        case MODEL_INPUT_FORMAT::BGR :
            cv::cvtColor(resized_roi, target_cvimage, cv::COLOR_RGB2BGR);
            break;
        default:
            ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
            printf("model input format enum[%d] is not supported\n", dstFmt);
            break;
        }
        if(ret!=RET_CODE::SUCCESS) return ret;   
        dst.push_back(target_cvimage);
//         cv::imwrite("./patch.jpg",target_cvimage);
    }

    LOGI << "<- PreProcessModel::preprocess_rgb_subpixel";
    return ret;
}

/**
 * 输入TVAI_IMAGE_FORMAT_RGB时,不会额外开辟空间, 返回的dst使用的是tvimage的pData
 */
ucloud::RET_CODE PreProcessModel::preprocess_all_to_rgb(ucloud::TvaiImage &tvimage, cv::Mat &dst){
    LOGI << "-> PreProcessModel::preprocess_all_to_rgb";
    RET_CODE ret = RET_CODE::SUCCESS;
    cv::Mat tmp;
    /*--------------tvimage to src------------------*/
    switch (tvimage.format)
    {//TVAI_IMAGE_FORMAT_NV12,TVAI_IMAGE_FORMAT_NV21都先转RGB
    case TVAI_IMAGE_FORMAT_NV12:
        tmp = cv::Mat(cv::Size(tvimage.width,3*tvimage.height/2),CV_8UC1, tvimage.pData);
        cv::cvtColor(tmp, dst, cv::COLOR_YUV2RGB_NV12);
        break;
    case TVAI_IMAGE_FORMAT_NV21:
        tmp = cv::Mat(cv::Size(tvimage.width,3*tvimage.height/2),CV_8UC1, tvimage.pData);
        cv::cvtColor(tmp, dst, cv::COLOR_YUV2RGB_NV21);
        break;
    case TVAI_IMAGE_FORMAT_RGB:
        dst = cv::Mat(cv::Size(tvimage.width,tvimage.height),CV_8UC3, tvimage.pData);
        // tmp.copyTo(dst);
        // dst = tmp;
        break;
    case TVAI_IMAGE_FORMAT_BGR:
        tmp = cv::Mat(cv::Size(tvimage.width,tvimage.height),CV_8UC3, tvimage.pData);
        cv::cvtColor(tmp, dst, cv::COLOR_BGR2RGB);
        break;
    default:
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        printf("tvimage.format enum[%d] is not supported\n", tvimage.format);
        break;
    }
    LOGI << "<- PreProcessModel::preprocess_all_to_rgb";
    // if(ret!=RET_CODE::SUCCESS) return ret;   
    return ret; 
}

ucloud::RET_CODE PreProcessModel::preprocess_subpixel(ucloud::TvaiImage &tvimage, std::vector<cv::Rect> rois, std::vector<cv::Mat> &dst, PRE_PARAM& config,
        std::vector<float>& aspect_ratio_x, std::vector<float>& aspect_ratio_y, bool use_subpixel)
{
    LOGI << "-> PreProcessModel::preprocess_subpixel";
    cv::Mat cvimage;
    ucloud::RET_CODE ret = preprocess_all_to_rgb(tvimage, cvimage);
    if(ret!=RET_CODE::SUCCESS) return ret;
    ret = preprocess_rgb_subpixel(cvimage, rois, dst, 
            config.model_input_shape, config.model_input_format, 
            aspect_ratio_x, aspect_ratio_y,config.keep_aspect_ratio, config.pad_both_side, use_subpixel);
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- PreProcessModel::preprocess_subpixel";
    return ret;
}


void PostProcessModel::base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, 
    CLS_TYPE* cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - 5;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = *_output++;
            float cy = *_output++;
            float w = *_output++;
            float h = *_output++;
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            _output++;//skip objectness
            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map == nullptr)
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            if(unique_cls_map.find(fbox.objtype)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objtype]].push_back(fbox);
        }
    }
    return;
}

template<typename T>
bool base_sortBox(const T& a, const T& b) {
  return  a.confidence > b.confidence;
}
template<typename T>
void base_nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output) {
  std::sort(input.begin(), input.end(), base_sortBox<T>);
  std::vector<int> bboxStat(input.size(), 0);
  for (size_t i=0; i<input.size(); ++i) {
    if (bboxStat[i] == 1) continue;
    output.push_back(input[i]);
    float area0 = (input[i].y1 - input[i].y0 + 1e-3)*(input[i].x1 - input[i].x0 + 1e-3);
    for (size_t j=i+1; j<input.size(); ++j) {
      if (bboxStat[j] == 1) continue;
      float roiWidth = std::min(input[i].x1, input[j].x1) - std::max(input[i].x0, input[j].x0);
      float roiHeight = std::min(input[i].y1, input[j].y1) - std::max(input[i].y0, input[j].y0);
      if (roiWidth<=0 || roiHeight<=0) continue;
      float area1 = (input[j].y1 - input[j].y0 + 1e-3)*(input[j].x1 - input[j].x0 + 1e-3);
      float ratio = 0.0;
      if (type == NMS_UNION) {
        ratio = roiWidth*roiHeight/(area0 + area1 - roiWidth*roiHeight);
      } else {
        ratio = roiWidth*roiHeight / std::min(area0, area1);
      }

      if (ratio > threshold) bboxStat[j] = 1; 
    }
  }
  return;
}
/**
 * Multi Class
 **/ 
void base_nmsBBox(std::vector<VecObjBBox> &input, float threshold, int type, VecObjBBox &output){
    LOGI << "-> YOLO_DETECTION::base_nmsBBox";
    if (input.empty()){
        VecObjBBox().swap(output);
        return;
    }
    for (int i = 0; i < input.size(); i++ ){
        base_nmsBBox(input[i], threshold, type, output);
    }
    LOGI << "<- YOLO_DETECTION::base_nmsBBox";
    return;
}
/*----------------------------------------------------------------------*/
/*ImageUtil DRM*/
/*----------------------------------------------------------------------*/
ucloud::RET_CODE ImageUtil::init(ucloud::TvaiImage &tvimage) {
    LOGI << "-> ImageUtil::init";
    //w,h需要偶数
    int texW,texH,bpp;
    int channels = 1;
    bool valid_img_fmt = true;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_RGB:
        texW = tvimage.width;
        texH = tvimage.height;
        if(texW%wstep!=0) texW += wstep - texW%wstep;
        if(texH%hstep!=0) texH += hstep - texH%hstep;
        bpp = 3*8;//bit per pixel
        channels = 3;
        break;
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        texW = tvimage.stride;
        texH = 3*tvimage.height/2;
        bpp = 8;
        channels = 1;
        break;
    default:
        valid_img_fmt = false;
        break;
    }
    if(!valid_img_fmt){
        printf("ImageUtil::init failed for unsupported image format\n");
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    // w += (w%2==0)?0:1;
    if(!initialed){
        // LOGI << "memset";
        memset(&rga_ctx, 0, sizeof(rga_context));
        memset(&drm_ctx, 0, sizeof(drm_context));
        LOGI << "drm_init";
        if(exists_file(dl_drm_path))
            drm_fd = drm_init(&drm_ctx, dl_drm_path.c_str());
        else{
            printf("drm_init failed, because %s not found\n", dl_drm_path.c_str());
            return RET_CODE::FAILED;
        }
        if(drm_fd < 0){
            printf("drm_init failed\n");
            return RET_CODE::FAILED;
        }
        LOGI << "drm_buf_alloc";
        if(exists_file(dl_drm_path)){
            drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, texW, texH, bpp, &buf_fd, &handle,
                                &actual_size);
            W = texW; H = texH;                            
        }
        else{
            printf("rga_init failed, because %s not found\n", dl_rga_path.c_str());
            return RET_CODE::FAILED;
        }
        LOGI << "RGA_init";
        int ret = RGA_init(&rga_ctx, dl_rga_path.c_str());
        if(ret<0){
            printf("rga_init failed\n");
            return RET_CODE::FAILED;
        }
        initialed = true;
    } else {
        if(texW!=W || texH!=H || channels!=C){
            LOGI << "reinitial";
            drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
            drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, texW, texH, bpp, &buf_fd, &handle,
                            &actual_size);
            W = texW; H = texH; C = channels;
        }
    }    
    LOGI << "<- ImageUtil::init";
    return RET_CODE::SUCCESS;
}

// ucloud::RET_CODE ImageUtil::init(int w, int h, int channels) {
//     LOGI << "-> ImageUtil::init";
//     //w,h需要偶数
//     if(channels == 3){
//         if( w%2 != 0)
//             w += 2 - w%2;
//     }
//     if(!initialed){
//         // LOGI << "memset";
//         memset(&rga_ctx, 0, sizeof(rga_context));
//         memset(&drm_ctx, 0, sizeof(drm_context));
//         LOGI << "drm_init";
//         if(exists_file(dl_drm_path))
//             drm_fd = drm_init(&drm_ctx, dl_drm_path.c_str());
//         else{
//             printf("drm_init failed, because %s not found\n", dl_drm_path.c_str());
//             return RET_CODE::FAILED;
//         }
//         if(drm_fd < 0){
//             printf("drm_init failed\n");
//             return RET_CODE::FAILED;
//         }
//         LOGI << "drm_buf_alloc";
//         if(exists_file(dl_drm_path)){
//             if(channels==3)
//                 drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, w, h, channels*8, &buf_fd, &handle,
//                                 &actual_size);
//             else
//                 drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, w, 3*h/2, channels*8, &buf_fd, &handle,
//                                 &actual_size);
//             W = w; H = h; C = channels;                            
//         }
//         else{
//             printf("rga_init failed, because %s not found\n", dl_rga_path.c_str());
//             return RET_CODE::FAILED;
//         }
//         LOGI << "RGA_init";
//         int ret = RGA_init(&rga_ctx, dl_rga_path.c_str());
//         if(ret<0){
//             printf("rga_init failed\n");
//             return RET_CODE::FAILED;
//         }
//         initialed = true;
//     } else {
//         if(w!=W || h!=H || channels!=C){
//             LOGI << "reinitial";
//             drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
//             drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, w, h, channels*8, &buf_fd, &handle,
//                             &actual_size);
//             W = w; H = h; C = channels;
//         }
//     }    
//     LOGI << "<- ImageUtil::init";
//     return RET_CODE::SUCCESS;
// }

void ImageUtil::release(void) {
    if(initialed){
        drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
        drm_deinit(&drm_ctx, drm_fd);
        RGA_deinit(&rga_ctx);
    }
    initialed = false;
}

/**
 * 因为RK_BGR模式存在问题, 所以遇到需要输出BGR则统一转RGB后再转BGR;
 * 如果输入是BGR, 则假装是RGB.
 */
RGA_MODE ImageUtil::get_rga_mode(TvaiImageFormat inputFMT, MODEL_INPUT_FORMAT outputFMT, bool &channel_reorder ){
    RGA_MODE ret = RGBtoRGB;
    channel_reorder = false;
    switch (inputFMT)
    {
    case TVAI_IMAGE_FORMAT_BGR :
        if(outputFMT == MODEL_INPUT_FORMAT::BGR )
            ret = BGRtoBGR;
        else if(outputFMT == MODEL_INPUT_FORMAT::RGB){
            ret = BGRtoBGR;
            channel_reorder = true;
        }
        break;
    case TVAI_IMAGE_FORMAT_RGB :
        if(outputFMT == MODEL_INPUT_FORMAT::BGR ){
            ret = RGBtoRGB;
            channel_reorder = true;
        }
        else if(outputFMT == MODEL_INPUT_FORMAT::RGB)
            ret = RGBtoRGB;
        break;
    case TVAI_IMAGE_FORMAT_NV21:
        if(outputFMT == MODEL_INPUT_FORMAT::BGR ){
            ret = NV21toRGB;
            channel_reorder = true;
        }
        else if(outputFMT == MODEL_INPUT_FORMAT::RGB)
            ret = NV21toRGB;    
        break;
    case TVAI_IMAGE_FORMAT_NV12:
        if(outputFMT == MODEL_INPUT_FORMAT::BGR ){
            ret = NV12toRGB;
            channel_reorder = true;
        }
        else if(outputFMT == MODEL_INPUT_FORMAT::RGB)
            ret = NV12toRGB;
        break;
    default:
        break;
    }
    return ret;
};

RET_CODE ImageUtil::resize(ucloud::TvaiImage &tvimage, PRE_PARAM pre_param,void *dstPtr){
    LOGI << "-> ImageUtil::resize";
    bool channel_reorder = false;
    int img_width = tvimage.width;
    int img_height = tvimage.height;
    int img_width_pad = img_width;
    int img_height_pad = img_height;
    bool valid_img_format = true;
    int ret = -1;
    RGA_MODE mode = get_rga_mode(tvimage.format, pre_param.model_input_format, channel_reorder);
    LOGI << "RGA_MODE "<< mode << ", channel_reorder: "<< channel_reorder;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_RGB:
        {
        LOGI << "TVAI_IMAGE_FORMAT_BGR/TVAI_IMAGE_FORMAT_RGB";
        //图像的宽必须是偶数才能drm resize
        cv::Mat cvimage(img_height,img_width,CV_8UC3,tvimage.pData);
        // cv::imwrite("x.jpg", cvimage); //check正常
        if(img_width%wstep!=0) img_width_pad = img_width + wstep - img_width%wstep;
        if(img_height%hstep!=0) img_height_pad = img_height + hstep - img_height%hstep;
        if( img_width==img_width_pad && img_height == img_height_pad ){
            LOGI << "no padding";
            memcpy(drm_buf, cvimage.data, img_width_pad * img_height_pad * 3);
        } else{
            LOGI << "padding";
            cv::Mat cvimage_padded = cv::Mat::zeros(cv::Size(img_width_pad, img_height_pad), CV_8UC3);
            cv::Mat tmp = cvimage_padded(cv::Rect(0,0,img_width, img_height));
            cvimage.copyTo(tmp);
            memcpy(drm_buf, cvimage_padded.data, img_width_pad * img_height_pad * 3);
            // cv::imwrite("y.jpg", cvimage_padded); //check正常
        }
        ret = img_resize_to_dst_format_slow(&rga_ctx, drm_buf, img_width_pad, img_height_pad, dstPtr, 
            pre_param.model_input_shape.w, pre_param.model_input_shape.h, mode);
        }    
        break;
    case TVAI_IMAGE_FORMAT_NV21:
    case TVAI_IMAGE_FORMAT_NV12:
        {
        LOGI << "TVAI_IMAGE_FORMAT_NV21/TVAI_IMAGE_FORMAT_NV12";
        memcpy(drm_buf, tvimage.pData, 3*tvimage.stride * img_height/2);
        ret = img_resize_to_dst_format_slow(&rga_ctx, drm_buf, img_width, img_height, dstPtr, 
            pre_param.model_input_shape.w, pre_param.model_input_shape.h, mode);
        }
        break;
    default:
        valid_img_format = false;
        break;
    }
    if(!valid_img_format){
        printf("invalid image format for ImageUtil::resize\n");
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    //人为反转
    if(channel_reorder){
        int _w = pre_param.model_input_shape.w;
        int _h = pre_param.model_input_shape.h;
        cv::Mat tmp1(cv::Size(_w,_h),CV_8UC3, dstPtr);
        cv::Mat tmp2;
        cv::cvtColor(tmp1,tmp2,cv::COLOR_RGB2BGR);
        memcpy(dstPtr, tmp2.data, _w*_h*3);
    }    
    LOGI << "<- ImageUtil::resize";
    // cv::Mat cvimage_show(size.h, size.w, CV_8UC3, dstPtr);
    // cv::imwrite("resized.jpg", cvimage_show);
    if(ret >= 0) return RET_CODE::SUCCESS;
    else return RET_CODE::FAILED;      
}


RET_CODE ImageUtil::resize(ucloud::TvaiImage &tvimage, ucloud::TvaiRect roi, PRE_PARAM pre_param, void *dstPtr){
    LOGI << "-> ImageUtil::resize";
    bool channel_reorder = false;
    int img_width = tvimage.width;
    int img_height = tvimage.height;
    int img_width_pad = img_width;
    int img_height_pad = img_height;
    bool valid_img_format = true;
    int ret = -1;
    RGA_MODE mode = get_rga_mode(tvimage.format, pre_param.model_input_format, channel_reorder);
    LOGI << "RGA_MODE "<< mode << ", channel_reorder: "<< channel_reorder;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_RGB:
        {
        LOGI << "TVAI_IMAGE_FORMAT_BGR/TVAI_IMAGE_FORMAT_RGB";
        //图像的宽必须是偶数才能drm resize
        cv::Mat cvimage(img_height,img_width,CV_8UC3,tvimage.pData);
        // cv::imwrite("x.jpg", cvimage); //check正常
        if(img_width%wstep!=0) img_width_pad = img_width + wstep - img_width%wstep;
        if(img_height%hstep!=0) img_height_pad = img_height + hstep - img_height%hstep;
        if( img_width==img_width_pad && img_height == img_height_pad ){
            LOGI << "no padding";
            memcpy(drm_buf, cvimage.data, img_width_pad * img_height_pad * 3);
        } else{
            LOGI << "padding";
            cv::Mat cvimage_padded = cv::Mat::zeros(cv::Size(img_width_pad, img_height_pad), CV_8UC3);
            cv::Mat tmp = cvimage_padded(cv::Rect(0,0,img_width, img_height));
            cvimage.copyTo(tmp);
            memcpy(drm_buf, cvimage_padded.data, img_width_pad * img_height_pad * 3);
            // cv::imwrite("y.jpg", cvimage_padded); //check正常
        }
        LOGI << "roi: " << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height;
        ret = img_roi_resize_to_dst_format_slow(&rga_ctx, drm_buf, roi.x, roi.y, roi.width, roi.height, img_width_pad, img_height_pad ,dstPtr, 
            pre_param.model_input_shape.w, pre_param.model_input_shape.h, mode);
        }    
        break;
    case TVAI_IMAGE_FORMAT_NV21:
    case TVAI_IMAGE_FORMAT_NV12:
        {
        LOGI << "TVAI_IMAGE_FORMAT_NV21/TVAI_IMAGE_FORMAT_NV12";
        memcpy(drm_buf, tvimage.pData, 3*tvimage.stride * tvimage.height/2);
        ret = img_roi_resize_to_dst_format_slow(&rga_ctx, drm_buf, roi.x, roi.y, roi.width, roi.height, tvimage.stride , tvimage.height,
        dstPtr, pre_param.model_input_shape.w, pre_param.model_input_shape.h, mode);
        }
        break;
    default:
        valid_img_format = false;
        break;
    }
    if(!valid_img_format){
        printf("invalid image format for ImageUtil::resize\n");
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    //人为反转
    if(channel_reorder){
        int _w = pre_param.model_input_shape.w;
        int _h = pre_param.model_input_shape.h;
        cv::Mat tmp1(cv::Size(_w,_h),CV_8UC3, dstPtr);
        cv::Mat tmp2;
        cv::cvtColor(tmp1,tmp2,cv::COLOR_RGB2BGR);
        memcpy(dstPtr, tmp2.data, _w*_h*3);
    }    
    LOGI << "<- ImageUtil::resize";
    // cv::Mat cvimage_show(size.h, size.w, CV_8UC3, dstPtr);
    // cv::imwrite("resized.jpg", cvimage_show);
    if(ret >= 0) return RET_CODE::SUCCESS;
    else return RET_CODE::FAILED;      
}


/*******************************************************************************
 * globalscaleTvaiRect 将rect根据scale放大, 同时不超过W,H的图像尺寸
*******************************************************************************/
TvaiRect globalscaleTvaiRect(TvaiRect &rect, float scale, int W, int H){
    /**
     * H,W is the border of image 
     */
    TvaiRect output;
    float cx = rect.x + rect.width/2;
    float cy = rect.y + rect.height/2;
    output.width = rect.width*scale;
    output.height = rect.height*scale;
    output.x = std::max(cx - output.width/2,  0.f);
    output.y = std::max(cy - output.height/2, 0.f);
    output.width = std::min(W - output.x, output.width);
    output.height = std::min(H - output.y, output.height);
    return output;
}

/*******************************************************************************
 * shift_box_from_roi_to_org 将roi坐标下的bbox, 还原成原图坐标下的bbox
*******************************************************************************/
void shift_box_from_roi_to_org(ucloud::VecObjBBox &bboxes, ucloud::TvaiRect &roirect){
    for(auto &&bbox: bboxes){
        bbox.rect.x += roirect.x;
        bbox.rect.y += roirect.y;
        for(auto &&pt: bbox.Pts.pts){
            pt.x += roirect.x;
            pt.y += roirect.y;
        }
    }
}

/*******************************************************************************
 * PreProcess_CPU_DRM_Model
 * DESC: CPU/DRM模式下图像前处理
*******************************************************************************/
/***whole image preprocess with drm**/
ucloud::RET_CODE PreProcess_CPU_DRM_Model::preprocess_drm(ucloud::TvaiImage& tvimage, PRE_PARAM pre_param , 
    std::vector<unsigned char*> &input_datas, 
    std::vector<float> &aX, std::vector<float> &aY)
{
    TvaiRect roi = {0,0,tvimage.width,tvimage.height};
    return preprocess_drm(tvimage, roi, pre_param, input_datas, aX, aY);
}

/***whole image preprocess with opencv**/
RET_CODE PreProcess_CPU_DRM_Model::preprocess_opencv(ucloud::TvaiImage& tvimage, PRE_PARAM pre_param ,
    std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY)
{
    TvaiRect roi = {0,0,tvimage.width,tvimage.height};
    return preprocess_opencv(tvimage, roi, pre_param, input_datas, aX, aY);
}

/***image with preprocess with roi+drm**/
RET_CODE PreProcess_CPU_DRM_Model::preprocess_drm(ucloud::TvaiImage& tvimage , ucloud::TvaiRect roi, PRE_PARAM pre_param ,
    std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> PreProcess_CPU_DRM_Model::preprocess_drm";
    bool valid_input_format = true;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
    case TVAI_IMAGE_FORMAT_NV12:
    case TVAI_IMAGE_FORMAT_NV21:
        break;
    default:
        valid_input_format = false;
        break;
    }
    if(!valid_input_format) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    int dst_w = pre_param.model_input_shape.w;
    int dst_h = pre_param.model_input_shape.h;
    unsigned char* data = (unsigned char*)std::malloc(3*dst_w*dst_h);
    RET_CODE uret = m_drm->init(tvimage);
    if(uret!=RET_CODE::SUCCESS) { free(data); data=nullptr; return uret;}
    // int ret = m_drm->resize(tvimage,m_InpSp, data);
    int ret = m_drm->resize(tvimage, roi, pre_param, data);
    input_datas.push_back(data);
    aX.push_back( (float(dst_w))/roi.width );
    aY.push_back( (float(dst_h))/roi.height );

#ifdef VISUAL
    cv::Mat cvimage_show( cv::Size(dst_w, dst_h), CV_8UC3, data);
    cv::imwrite("preprocess_drm.jpg", cvimage_show);
#endif

    LOGI << "<- PreProcess_CPU_DRM_Model::preprocess_drm";
    return RET_CODE::SUCCESS;    
}

/***image preprocess with roi+opencv**/
RET_CODE PreProcess_CPU_DRM_Model::preprocess_opencv(ucloud::TvaiImage& tvimage, ucloud::TvaiRect roi, PRE_PARAM pre_param ,
    std::vector<unsigned char*> &input_datas, std::vector<float> &aX, std::vector<float> &aY)
{
    LOGI << "-> PreProcess_CPU_DRM_Model::preprocess_opencv";
    bool use_subpixel = false;
    std::vector<cv::Mat> dst;
    std::vector<cv::Rect> _roi_ = {cv::Rect(roi.x,roi.y,roi.width,roi.height)};
    RET_CODE ret = PreProcessModel::preprocess_subpixel(tvimage, _roi_, 
        dst, pre_param, aX, aY, use_subpixel);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&ele: dst){
        #ifdef VISUAL
        cv::imwrite("preprocess_opencv.jpg", ele);
        #endif
        // cv::imwrite("preprocess_opencv.jpg", ele);
        unsigned char* data = (unsigned char*)std::malloc(ele.total()*3);
        memcpy(data, ele.data, ele.total()*3);
        input_datas.push_back(data);
    }
    LOGI << "<- PreProcess_CPU_DRM_Model::preprocess_opencv";
    return ret;    
}





unsigned char * readfile(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        fflush(stdout);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        fflush(stdout);
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
