#ifndef _MODULE_SISO_HPP_
#define _MODULE_SISO_HPP_

#include "module_base.hpp"
#include "basic.hpp"
/**
 * NaiveModel实现了单图像输入单head的输出, 可以用来做参考.
 * chaffee.chen@ucloud.cn 2022-09-09
 */
class NaiveModel: public ucloud::AlgoAPI {
public:
/**
 * public API
 */
    NaiveModel();
    virtual ~NaiveModel();
    virtual ucloud::RET_CODE init(std::map<ucloud::InitParam, std::string> &modelpath);
    virtual ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes);

/**
 * non-public API
 */
    /**
     * run_mem 
     * DESC: 内部使用general_infer_uint8_nhwc_to_float_mem 减少内存开辟和复制
     */
    virtual ucloud::RET_CODE run_mem(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes);
    virtual ucloud::RET_CODE preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas);

private:
    std::shared_ptr<BaseModel> m_net = nullptr;//推理模型的主干部分
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;//It will contain zero!!!!
    int m_OutEleNum;//输出Tensor元素总数
    int m_InpNum = 1;//输入Tensor数量
    int m_OtpNum = 1;//输出Tensor数量

    MemPool m_OtpMemPool;//输出的内存池, 减少反复开辟和释放内存空间

#ifdef TIMING
    Timer m_Tk;
#endif
};

#endif