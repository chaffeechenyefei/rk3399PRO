#include "imp_abandon_object.hpp"
#include <opencv2/opencv.hpp>

ucloud::RET_CODE IMP_ABANDON_DETECTOR::init(std::map<ucloud::InitParam, ucloud::WeightData> &weightConfig){
    LOGI << "-> IMP_ABANDON_DETECTION::init";
    ucloud::WeightData abandon_modelpath;
    if(weightConfig.find(ucloud::InitParam::BASE_MODEL)==weightConfig.end()) {
        std::cout << weightConfig.size() << endl;
        for(auto param: weightConfig){
            printf( "[%d]:[%s], ", param.first, param.second);
        }
        printf("ERR:: IMP_ABANDON_DETECTION->init() still missing models\n");
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    RET_CODE ret = RET_CODE::FAILED;
    abandon_modelpath = weightConfig[ucloud::InitParam::BASE_MODEL];
    //ped detection
    ret = m_abandon_ptr->init(abandon_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    LOGI << "<- IMP_ABANDON_DETECTION::init";
    return RET_CODE::SUCCESS;
}

ucloud::RET_CODE IMP_ABANDON_DETECTOR::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI << "-> IMP_ABANDON_DETECTION::init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    std::map<ucloud::InitParam, ucloud::WeightData> weightConfig;
    for(auto &&modelp: modelpath){
        int szBuf = 0;
        unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
        weightConfig[modelp.first] = ucloud::WeightData{tmpBuf,szBuf};
    }
    ret = init(weightConfig);
    for(auto &&wC: weightConfig){
        free(wC.second.pData);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    return ucloud::RET_CODE::SUCCESS;
}

void IMP_ABANDON_DETECTOR::updateBG(ucloud::BatchImageIN &batch_tvimages){
        Mat mat0(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[0].pData);
        Mat mat1(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[1].pData);
        Mat mat2(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[2].pData);
        m_BGmat = m_Bg_weight[0]*mat0 + m_Bg_weight[1]*mat1+m_Bg_weight[2]*mat2; 
}

ucloud::RET_CODE IMP_ABANDON_DETECTOR::run(ucloud::BatchImageIN &batch_tvimages, ucloud::VecObjBBox &bboxes){
    LOGI << "-> IMP_ABANDON_DETECTOR::run";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ucloud::TvaiImageFormat format;
    int width,height,stride,datasize,uucam_id;
    ucloud::VecObjBBox detBboxes;

    if (m_upbg){
        //如果选择动态更新背景，那么batch_tvimages的大小要满足3，
        //每次输入的batch_tvimages应将vector头上的tvimage剔除，新增的tvimage在vector的尾部
        if (m_BGmat.empty()){
            updateBG(batch_tvimages);
            return ret;
            // Mat mat0(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[0].pData);
            // Mat mat1(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[1].pData);
            // Mat mat2(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[2].pData);
            // m_BGmat = m_Bg_weight[0]*mat0 + m_Bg_weight[1]*mat1+m_Bg_weight[2]*mat2;
        }
        Mat diff;
        //batch_tvimages的尾部是最新的帧
        Mat cur(batch_tvimages[0].height,batch_tvimages[0].width,CV_8UC3,batch_tvimages[2].pData); 
        absdiff(m_BGmat,cur,diff);
        unsigned char *imgBuf = nullptr;
        imgBuf = (unsigned char *)malloc(diff.total()*3);
        memcpy(imgBuf,diff.data,diff.total()*3);
        ucloud::TvaiImage tvImg{batch_tvimages[0].format,batch_tvimages[0].width,batch_tvimages[0].height,batch_tvimages[0].stride,imgBuf,batch_tvimages[0].dataSize}; 
        ret = m_abandon_ptr->run(tvImg,detBboxes);
        if (imgBuf){
            free(imgBuf);
        }
        updateBG(batch_tvimages);
    }
    else {
        if (m_BackGround.pData==nullptr || m_BGmat.empty()){
            m_BackGround = batch_tvimages[0];
            m_BGmat = Mat(m_BackGround.height,m_BackGround.height,CV_8UC3,m_BackGround.pData);
            return ret;
        }
        Mat diff;
        Mat cur(m_BackGround.height,m_BackGround.width,CV_8UC3,batch_tvimages[0].pData); 
        absdiff(m_BGmat,cur,diff);
        unsigned char *imgBuf = nullptr;
        imgBuf = (unsigned char *)malloc(diff.total()*3);
        memcpy(imgBuf,diff.data,diff.total()*3);
        ucloud::TvaiImage tvImg{batch_tvimages[0].format,batch_tvimages[0].width,batch_tvimages[0].height,batch_tvimages[0].stride,imgBuf,batch_tvimages[0].dataSize};
        ret = m_abandon_ptr->run(tvImg,detBboxes);
        if (imgBuf){
            free(imgBuf);
        }
    }

    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] IMP_ABANDON_DETECTOR classify images failed!\n", __FILE__, __LINE__);
        return ret;
    }
    for(auto &&box:detBboxes){
        if(box.objtype == ucloud::CLS_TYPE::ABADNON_STATIC)
            bboxes.push_back(box);
    }
    LOGI << "<- IMP_ABANDON_DETECTOR::run";
    return ret;
}


ucloud::RET_CODE IMP_ABANDON_DETECTOR::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return m_abandon_ptr->get_class_type(valid_clss);
}

// ucloud::RET_CODE ABANDON_OBJECT_DETECTOR::updateBG(const Mat cur){
//     // Mat cur(tvimage.height,tvimage.width,CV_8UC3,tvimage.pData);
//     Mat first = m_batchBg.front();
//     m_bg -=  first*m_Bg_weight[0];
//     m_batchBg.pop();
//     m_bg += cur*m_Bg_weight[m_Bg_weight.size()-1];
//     m_batchBg.push(cur);
//     return RET_CODE::SUCCESS;
// }
