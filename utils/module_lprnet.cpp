#include "module_lprnet.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>

using namespace std;
using namespace ucloud;

static cv::Mat get_perspective_mat(vector<cv::Point2f> &srcPts, float w=94, float h=24){
    vector<cv::Point2f> dstPts = {
        cv::Point2f(w,h),
        cv::Point2f(0,h),
        cv::Point2f(0,0),
        cv::Point2f(w,0),
    };

    return cv::getPerspectiveTransform(srcPts, dstPts);
}



static void transpose(float *ptrsrc, float *ptrdst, int r = 18, int c = 68){
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            ptrdst[i*c+j] = ptrsrc[j*r+i];
        }
    }
}

static string decode_license(float *output){
    int maxid = -1;
    int pos;
    float max_val;

    float *_output = (float*)malloc(sizeof(float)*18*68);
    transpose(output, _output);

    vector<int> preb_label, no_repeat_blank_label;
    for (int i = 0; i < 18; i++){
        pos = 0;
        max_val = _output[i*68];
        for (int j = 0; j < 68; j++){
            // if(i==0)
            //     printf("%.1f, ", _output[i*68+j]);
            if (_output[i*68 + j] > max_val){
                pos = j;
                max_val = _output[i*68 + j];
            }
        }
        // if(i==0)
        //     printf("\n");
        preb_label.push_back(pos);
    }
    free(_output);

    // printf("preb_label = ");
    // for(auto &&t: preb_label){
    //     printf("%d, ", t);
    // }
    // printf("\n");
    // return "testing";

    int pre_c = preb_label[0];
    if (pre_c != 67){
        no_repeat_blank_label.push_back(pre_c);
    }
    int c;
    for (int i = 0; i < preb_label.size(); i++){
        c = preb_label[i];
        if (pre_c == c || c == 67){
            if (c == 67){
                pre_c = c;
            }
            continue;
        }
        no_repeat_blank_label.push_back(c);
        pre_c = c;
    }

    // printf("no_repeat_blank_label = ");
    // for(auto &&t: no_repeat_blank_label){
    //     printf("%d, ", t);
    // }
    // printf("\n");    
    
    string licplate_str = "";
    for (int i = 0; i < no_repeat_blank_label.size(); i++){
        licplate_str +=  LICPLATE_CHARS[no_repeat_blank_label[i]];
    }
    return licplate_str;
}



LPRNET::LPRNET(){
    LOGI<<"-> LPRNET";
    m_net = std::make_shared<BaseModel>();
    m_cv_preprocess_net = std::make_shared<PreProcess_CPU_DRM_Model>();
    LOGI<<"<- LPRNET";
}

LPRNET::~LPRNET(){
    LOGI<<"->< ~LPRNET";
}

ucloud::RET_CODE LPRNET::init(std::map<ucloud::InitParam,ucloud::WeightData> &weightConfig){
    LOGI<<"-> LPRNET::init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;

    if (weightConfig.find(ucloud::InitParam::BASE_MODEL)== weightConfig.end()){
        printf("**[%s][%d] base model not found in weightConfig\n", __FILE__, __LINE__);
        return ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    bool useDRM = false;
    ret = m_net->base_init(weightConfig[ucloud::InitParam::BASE_MODEL],useDRM);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        return ret;
    }
    if(m_InpNum != m_net->get_input_shape().size()){
        printf("** dims err m_InpuNum[%d] != m_net->get_input_shape().size()[%d]\n", m_InpNum, m_net->get_input_shape().size());
        return RET_CODE::FAILED;
    }
    if(m_OutNum != m_net->get_output_shape().size()){
        printf("** dims err m_OutNum[%d] != m_net->get_output_shape().size()[%d]\n", m_OutNum, m_net->get_output_shape().size());
        return RET_CODE::FAILED;
    }

    m_InpSp = m_net->get_input_shape()[0];
    m_OutEleDims = m_net->get_output_dims();
    m_OutEleNums = m_net->get_output_elem_num();
    m_param_img2tensor.keep_aspect_ratio = false;
    m_param_img2tensor.pad_both_side = false;
    m_param_img2tensor.model_input_format = MODEL_INPUT_FORMAT::RGB;//confirm??
    m_param_img2tensor.model_input_shape = m_InpSp;
    LOGI << "<- LPRNET::init";
    return ret;
}


ucloud::RET_CODE LPRNET::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"-> LPRNET::init";
    std::map<InitParam, WeightData> weightConfig;
    for(auto &&modelp: modelpath){
        int szBuf = 0;
        unsigned char* tmpBuf = readfile(modelp.second.c_str(),&szBuf);
        weightConfig[modelp.first] = WeightData{tmpBuf,szBuf};
    }
    RET_CODE ret = init(weightConfig);
    for(auto &&wC: weightConfig){
        free(wC.second.pData);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- LPRNET::init";
    return ret;
}



ucloud::RET_CODE LPRNET::postprocess(std::vector<float*> &output_datas, float threshold,BBox &bbox){
    LOGI << "-> LPRNET::postprocess";
    RET_CODE ret = RET_CODE::SUCCESS;
    if(output_datas.empty()) return RET_CODE::FAILED;
    float *ptr = output_datas[0];
    // std::string license_str = "testing";
    std::string license_str = decode_license(ptr);
    printf("[%s][%d] car license: %s\n", __FILE__, __LINE__, license_str.c_str());
    LOGI << "<- LPRNET::postprocess";
    return ret;
}

/*******************************************************************************
 * run 对bboxes中的每个区域进行分类, 并将结果更新到bboxes中(objtype, objectness, confidence)
*******************************************************************************/
ucloud::RET_CODE LPRNET::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    LOGI<<"-> LPRNET::run";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    switch (tvimage.format)
    {
    case ucloud::TVAI_IMAGE_FORMAT_BGR:
    case ucloud::TVAI_IMAGE_FORMAT_RGB:
    case ucloud::TVAI_IMAGE_FORMAT_NV12:
    case ucloud::TVAI_IMAGE_FORMAT_NV21:
        ret = ucloud::RET_CODE::SUCCESS;
        break;
    
    default:
        ret = ucloud::RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if (ret!=ucloud::RET_CODE::SUCCESS) return ret;
    
    // std::vector<int> output_nums = m_net->get_output_elem_num();
    
    if(bboxes.empty()) return RET_CODE::SUCCESS;//没有目标直接返回

    for(auto &&box: bboxes){
        std::vector<unsigned char*> input_datas;
        std::vector<float*> output_datas;
        TvaiRect roi = box.rect;
        vector<float> aX,aY;
        
        roi =  get_valid_rect(roi, tvimage.width, tvimage.height);
        ret = m_cv_preprocess_net->preprocess_opencv(tvimage, roi, m_param_img2tensor, input_datas, aX, aY);//drm存在问题

        //perspective transform
        if(!box.Pts.pts.empty() || box.Pts.pts.size() < 4){
            cv::Mat img(cv::Size(94,24), CV_8UC3, input_datas[0]);
            cv::Mat pimg;
            vector<cv::Point2f> srcPts = {
                cv::Point2f( box.Pts.pts[0].x - roi.x , box.Pts.pts[0].y - roi.y ),
                cv::Point2f( box.Pts.pts[1].x - roi.x , box.Pts.pts[1].y - roi.y ),
                cv::Point2f( box.Pts.pts[2].x - roi.x, box.Pts.pts[2].y - roi.y ),
                cv::Point2f( box.Pts.pts[3].x - roi.x, box.Pts.pts[3].y - roi.y ),
            };
            cv::Mat projMat = get_perspective_mat(srcPts);
            cv::warpPerspective(img, pimg, projMat, cv::Size(94,24));
            memcpy(pimg.data, input_datas[0], 94*24*3);
        }

        if(ret!=ucloud::RET_CODE::SUCCESS){
            printf("**[%s][%d] LPRNET preprocess return [%d]\n", __FILE__, __LINE__, ret);
            return ret;
        }

        ret  = m_net->general_infer_uint8_nhwc_to_float(input_datas,output_datas);
        if(ret!=RET_CODE::SUCCESS) {
            for(auto &&t: input_datas) free(t);
            return ret;
        }

        // cv::Mat img_show(cv::Size(94,24), CV_8UC3, input_datas[0]);
        // cv::imwrite("tmp.jpg",img_show);

        ret = postprocess(output_datas, threshold, box);

        if(ret!=RET_CODE::SUCCESS) {
            LOGI << "-> free1";
            for(auto &&t: input_datas) free(t);
            for(auto &&t: output_datas) free(t);
            LOGI << "<- free1";
            return ret;
        }

        for(auto &&t: output_datas){
            LOGI << "-> free2";
            free(t);
            LOGI << "<- free2";
        }
        for(auto &&t: input_datas){
            LOGI << "-> free3";
            free(t);
            LOGI << "<- free3";
        }
    }
    return ret;
}


ucloud::RET_CODE LPRNET::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    return ucloud::RET_CODE::SUCCESS;
}

ucloud::RET_CODE LPRNET::set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss){
    return RET_CODE::SUCCESS;
}
