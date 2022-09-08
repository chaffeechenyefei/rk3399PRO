#ifndef _MODULE_SISO_HPP_
#define _MODULE_SISO_HPP_

#include "module_base.hpp"

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
    virtual ucloud::RET_CODE preprocess(ucloud::TvaiImage& tvimage, std::vector<unsigned char*> &input_datas);
    virtual ucloud::RET_CODE postprocess(std::vector<float*> &output_datas);

private:
    std::shared_ptr<BaseModel> m_net = nullptr;
    DATA_SHAPE m_InpSp;
    DATA_SHAPE m_OutSp;//It will contain zero!!!!
    int m_OutEleNum;
};

#endif