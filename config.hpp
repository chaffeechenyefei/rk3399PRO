#include "libai_core.hpp"
#include <string.h>
#include <iostream>
#include <map>

using ucloud::AlgoAPIName;
using ucloud::InitParam;
using std::string;

enum class TASKNAME{
    //ID
    FACE = 0,//人脸检测+特征提取
    //yolo based
    PED_CAR_NONCAR  = 1,//人车非检测
    PED             = 2,//行人检测
    FIRE            = 3,//火焰检测
    PED_FALL        = 4,//摔倒检测
    SAFETY_HAT      = 5,//安全帽检测
    TRASH_BAG       = 6,//垃圾袋检测
    //special
    SKELETON        = 7,//骨架检测
    WATER           = 8,//积水检测
    FIRE_X          = 9,//火焰检测, 加强版
    BANNER          = 10,//横幅标语检测
    NONCAR          = 11,//非机动车检测
    FIGHT           = 12,//打斗识别(全图检测)
    FIGHT_DET       = 13,//打斗识别(有人的地方检测打斗)
    GKPW            = 14,//高空抛物
    GKPW2           = 15,//传统方法的高空抛物
    SMOKING         = 16,//抽烟检测
    SMOKING_FACE    = 17,//抽烟检测, 只用人脸嘴部
    PHONING         = 18,//打电话玩手机检测
    HEAD            = 19,//人头检测
    SOS             = 20,//SOS求救
    PED_SK          = 21,//骨架检测
    FACE_ATTR       = 22,//人脸检测+属性
    PED_CAR_NONCARV2  = 23,//人车非检测
    LICPLATE        = 24,//车牌检测+车牌识别
    RESERVED2       = 25,//yolov5 mode3
    PED_CAR_NONCAR_FAST_LOAD  = 26,//人车非检测快速加载
    FACE_EXT        = 27,//单纯人脸特征提取
    PED_BEND        = 28,//行人弯腰检测
    HAND            = 29,//手的检测
    SMOKE_CLOUD     = 30, //烟雾团检测  
    SMOKE_CLOUD_UNET= 31, //烟雾团检测  unet测试
    ABANDON_OBJECT  = 32,
    TASK_END,

    HAND_DET        = 50,//手的检测 224x320
    HAND_L_DET      = 51,//手的检测 736x416
    JSON            = 100,//用户自定义json形式
    TJ_HELMET       = 200,//同济416x416安全帽检测 
} ;

/**
 * 寒武纪模型文件KEY
 */
enum class MODELFILENAME{
    FACE_DET,//人脸检测
    FACE_EXT,//人脸特征提取
    SKELETON_DET_R50,//骨架原始R50
    SKELETON_DET_R18,//骨架R18, coco
    FIRE_DET,//火焰检测
    FIRE_DET_220407,//火焰检测220407
    FIRE_CLS,//火焰分类
    WATER_DET_UNET,//UNet积水分割
    WATER_DET_PSP,//PSPNet积水分割
    SMOKE_CLOUD_DET_R34,//烟雾团分割
    SMOKE_CLOUD_DET_UNET,//烟雾团分割
    GENERAL_DET,//人车非检测
    GENERAL_DET_MODE3,//人车非检测
    GENERAL_DET_MODE4,//人车非检测
    GENERAL_TRK_MLU,//跟踪特征提取器(寒武纪)
    GENERAL_TRK_R18,//跟踪特诊提取器(R18)
    PED_DET,//行人检测
    PED_FALL_DET,//摔倒检测
    SAFETY_HAT_DET,//安全帽
    TJ_HELMET_DET, //同济416x416安全帽检测 

    LICPLATE_DET,//车牌检测
    LICPLATE_RECOG,//车牌识别

    TRASH_BAG_DET,//垃圾袋
    BANNER_DET,//横幅
    MOTOR_DET,//电瓶车、自行车检测
    HAND_DET_736x416,//手的检测736x416
    HAND_DET_224x320,//手的检测224x320
    CIG_DET,//香烟检测
    PHONE_CLS_220215,//打电话分类
    PHONE_CLS_220302,//打电话分类
    PHONE_CLS_220302_INNER_NORM,//打电话分类 模型内-m/std
    HEAD_DET,//人头检测
    MOD_DET_DIF,//DIF移动物体分割
    MOD_DET_UNET,//UNet移动物体分割
    ACTION_CLS,//行为识别
    FACEATTR_CLS,//人脸属性分类器112x112
    
    ABANDON_OBJECT_DET,
};

/**
 * 寒武纪模型文件KEY,ADDR
 */
std::string rknn_model_path = "/home/firefly/yefei/test/data/model/";
std::map<MODELFILENAME,string> cambricon_model_file = {
    {MODELFILENAME::FACE_DET,           rknn_model_path + "retinaface_int8_2022xx_736x416_slow.rknn"},
    {MODELFILENAME::FACE_EXT,           rknn_model_path + "resnet50_irse_mx_int8_2022xx_112x112_fast.rknn"},
    {MODELFILENAME::SKELETON_DET_R50,   "pose_resnet_50_256x192_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::SKELETON_DET_R18,   rknn_model_path + "rknn_int8_posenet-r18_20220225_192x256_slow.rknn"},
    {MODELFILENAME::FIRE_CLS,           rknn_model_path + "rknn_int8_fire-r34_20220302_224x224_fast.rknn"},
    {MODELFILENAME::WATER_DET_UNET,     "unetwater_393_224x224_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::WATER_DET_PSP,      "pspwater_20211119_736x416_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::SMOKE_CLOUD_DET_R34, rknn_model_path + "rknn_int8_smokecloud-r34_20221115_224x224_fast.rknn"},
    {MODELFILENAME::SMOKE_CLOUD_DET_UNET,rknn_model_path + "rknn_int8_smokecloud-unet_20221116_224x224_fast.rknn"},
    {MODELFILENAME::GENERAL_TRK_MLU,    "feature_extract_4c4b_argb_270_v1.5.0.cambricon"},
    {MODELFILENAME::GENERAL_TRK_R18,    "track-r18_20220113_64x128_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::GENERAL_DET,        rknn_model_path + "yolov5s-conv-9-20211104_736x416.rknn"},
    {MODELFILENAME::GENERAL_DET_MODE3,  rknn_model_path + "yolov5s-conv-9-20211104_736x416_mode3_precompiled.rknn"},
    {MODELFILENAME::GENERAL_DET_MODE4,  rknn_model_path + "yolov5s-conv-9-20211104_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::PED_DET,            rknn_model_path + "yolov5s-conv-people-aug-fall_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::PED_FALL_DET,       rknn_model_path + "yolov5s-conv-fall-ped-20220301_736x416_mode4_precompiled.rknn"},//20220222
    {MODELFILENAME::SAFETY_HAT_DET,     rknn_model_path + "yolov5s-conv-safety-hat-20220217_736x416_mode4_precompiled.rknn"},//20220222
    {MODELFILENAME::TJ_HELMET_DET,      "yolov5s-conv-safety-hat-tongji-20220915_416x416_mlu220_bs1c1_fp16.cambricon"},//20220915
    {MODELFILENAME::TRASH_BAG_DET,      rknn_model_path + "yolov5s-conv-trashbag-20211214_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::FIRE_DET,           "yolov5s-conv-fire-21102010_736x416_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::FIRE_DET_220407,    rknn_model_path + "yolov5s-conv-fire-220407_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::BANNER_DET,         rknn_model_path + "yolov5s-conv-banner-20211130_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::MOTOR_DET,          rknn_model_path + "yolov5s-conv-motor-20211217_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::HAND_DET_224x320,   "yolov5s-conv-hand-20220117_224x320_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::HAND_DET_736x416,   rknn_model_path + "yolov5s-conv-hand-20220118_736x416_mode4_precompiled.rknn"},
    {MODELFILENAME::CIG_DET,            rknn_model_path + "yolov5s-conv-cig-20220311_256x256_mode4_precompiled.rknn"},
    {MODELFILENAME::PHONE_CLS_220215,   "phoning-r34_20220215_256x256_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::PHONE_CLS_220302,   rknn_model_path + "phone_resnet34_20220302_256x256_mode0_precompiled.rknn"},
    {MODELFILENAME::PHONE_CLS_220302_INNER_NORM,   rknn_model_path + "phone_resnet34_20220302_256x256_mode0_normal_rgb_precompiled.rknn"},
    {MODELFILENAME::HEAD_DET,           rknn_model_path + "yolov5s-conv-head-20220121_736x416_mode4_precompiled.rknn"},//20220222
    //BATCH IN============================================================================================================================
    {MODELFILENAME::ACTION_CLS,         "tsn_53_224x224_mlu220_bs1c1_fp16.cambricon"},
    {MODELFILENAME::MOD_DET_UNET,       "unetResNet18_bn_110_224x224_mlu220_t2bs1c1_int8.cambricon"},
    {MODELFILENAME::MOD_DET_DIF,        "diffunet_20220106_736x416_mlu220_t2bs1c1_fp16_int8.cambricon"},
    {MODELFILENAME::FACEATTR_CLS,       "faceattr-effnet_20220628_112x112_mlu220_bs1c1_fp16.cambricon"},//20220628
    {MODELFILENAME::LICPLATE_DET,       "yolov5s-face-licplate-20220815_736x416_mlu220_bs1c1_fp16.cambricon"},//20220815
    {MODELFILENAME::LICPLATE_RECOG,     "licplate-recog_20220822_94x24_mlu220_bs1c1_fp16.cambricon"}, //20220822
    {MODELFILENAME::ABANDON_OBJECT_DET,  rknn_model_path+"rknn_int8_abandon-unet_20220825_736x416_fast.rknn"},
};

bool task_parser(TASKNAME taskid, float &threshold, float &nms_threshold, AlgoAPIName &apiName, std::map<InitParam, std::string> &init_param, int &use_batch, bool displayTask=false){
    if(!displayTask) std::cout << "=============parser start=================" << std::endl;
    string taskDesc;
    use_batch = 1;
    threshold = 0.2;
    nms_threshold = 0.6;
    bool retcode = true;
    switch (taskid)
    {
    case TASKNAME::SKELETON:
        threshold = 0.5;
        apiName = AlgoAPIName::SKELETON_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::SKELETON_DET_R18]},
        };
        taskDesc = "skeleton detector";
        break;
    case TASKNAME::FACE:
        threshold = 0.5;
        apiName = AlgoAPIName::FACE_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::FACE_DET]},
        };
        taskDesc = "face";
        break;           
    case TASKNAME::FACE_ATTR://推荐阈值0.8
        threshold = 0.7;
        apiName = AlgoAPIName::FACE_DETECTOR_ATTR;
        nms_threshold = 0.6;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::FACE_DET] },
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::FACEATTR_CLS]},
            {InitParam::TRACK_MODEL,  cambricon_model_file[MODELFILENAME::GENERAL_TRK_MLU]},
        };
        taskDesc = "face detection with attribution";
        break;
    case TASKNAME::FACE_EXT:
        threshold = 0.5;
        apiName = AlgoAPIName::FEATURE_EXTRACTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::FACE_EXT]},
        };
        taskDesc = "face feature extraction on 112x112 images only";
        break;           
    case TASKNAME::SMOKING://推荐阈值0.6
        threshold = 0.6;
        apiName = AlgoAPIName::SMOKING_DETECTOR;
        nms_threshold = 0.2;//trival in this task
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::FACE_DET] },
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::CIG_DET]},
        };
        taskDesc = "smoking";
        break;
    case TASKNAME::PHONING:
        threshold = 0.7;
        apiName = AlgoAPIName::PHONING_DETECTOR;
        nms_threshold = 0.6;
        init_param = { 
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4]},
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::PHONE_CLS_220302_INNER_NORM]},
        };
        taskDesc = "phoning";
        break;
    case TASKNAME::PED_FALL:
        threshold = 0.3;
        apiName = AlgoAPIName::PED_FALL_DETECTOR_X;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::PED_FALL_DET]},
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::SKELETON_DET_R18]},
        };
        taskDesc = "ped falling";
        break;
    case TASKNAME::PED_BEND:
        threshold = 0.3;
        apiName = AlgoAPIName::PED_BEND_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4]},
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::SKELETON_DET_R18]},
        };
        taskDesc = "ped bending";
        break;        
    case TASKNAME::PED_SK:
        threshold = 0.5;
        apiName = AlgoAPIName::PED_SK_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::PED_DET]},
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::SKELETON_DET_R18]},
        };
        taskDesc = "ped (wh)192x256 skeleton";
        break;        
    case TASKNAME::PED:
        threshold = 0.5;
        apiName = AlgoAPIName::PED_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::PED_DET]},
        };
        taskDesc = "ped";
        break;        
    case TASKNAME::HAND:
        threshold = 0.5;
        apiName = AlgoAPIName::HAND_DETECTOR;
        nms_threshold = 0.6;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::HAND_DET_736x416]},
        };
        taskDesc = "hand detection";
        break;            


    case TASKNAME::FIRE: //建议阈值0.7
        threshold = 0.35;
        apiName = AlgoAPIName::FIRE_DETECTOR;
        init_param = { 
            // {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::FIRE_DET] },
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::FIRE_DET_220407] },
            // {InitParam::SUB_MODEL, cambricon_model_file[MODELFILENAME::FIRE_CLS] },
        };
        taskDesc = "FIRE";
        break;
    case TASKNAME::FIRE_X: //建议阈值0.2
        threshold = 0.4;
        apiName = AlgoAPIName::FIRE_DETECTOR_X;
        taskDesc = "FIRE_X(cascaded models)";  
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::FIRE_DET_220407] },
            {InitParam::SUB_MODEL, cambricon_model_file[MODELFILENAME::FIRE_CLS] },
        };        
        break;

    case TASKNAME::PED_CAR_NONCAR: //建议阈值0.6
        threshold = 0.55;
        apiName = AlgoAPIName::GENERAL_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4] },
        };
        taskDesc = "PED CAR NONCAR";
        break;
    case TASKNAME::PED_CAR_NONCARV2: //建议阈值0.6
        threshold = 0.40;
        apiName = AlgoAPIName::GENERAL_DETECTORV2;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4] },
        };
        taskDesc = "PED CAR NONCARV2";
        break;
    case TASKNAME::PED_CAR_NONCAR_FAST_LOAD: //建议阈值0.6
        threshold = 0.55;
        apiName = AlgoAPIName::GENERAL_DETECTOR_FAST_LOAD;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4] },
        };
        taskDesc = "PED CAR NONCAR yolov5 mode4 fast load";
        break;           
    case TASKNAME::LICPLATE: //建议阈值0.55
        threshold = 0.5;
        apiName = AlgoAPIName::LICPLATE_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::LICPLATE_DET] },
            // {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::LICPLATE_RECOG]},
        };
        taskDesc = "LICPLATE detect with recog";
        break;   
    case TASKNAME::SAFETY_HAT: //建议阈值0.55
        threshold = 0.55;
        apiName = AlgoAPIName::SAFETY_HAT_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::SAFETY_HAT_DET] },
        };
        taskDesc = "SAFETY_HAT";
        break;    
    // case TASKNAME::TJ_HELMET: //建议阈值0.55
    //     threshold = 0.55;
    //     apiName = AlgoAPIName::TJ_HELMET_DETECTOR;
    //     init_param = { 
    //         {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::TJ_HELMET_DET] },
    //     };
    //     taskDesc = "TONGJ_SAFETY_HAT";
    //     break;           
    case TASKNAME::TRASH_BAG: //建议阈值0.3
        threshold = 0.3;
        apiName = AlgoAPIName::TRASH_BAG_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::TRASH_BAG_DET] },
        };
        taskDesc = "TRASH_BAG";
        break;    
    case TASKNAME::WATER: //建议阈值0.5
        threshold = 0.5;
        apiName = AlgoAPIName::WATER_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::WATER_DET_PSP] },
        };
        taskDesc = "WATER";
        break;    
    case TASKNAME::SMOKE_CLOUD: //建议阈值0.5
        threshold = 0.5;
        apiName = AlgoAPIName::SMOKE_CLOUD_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::SMOKE_CLOUD_DET_R34] },
        };
        taskDesc = "SMOKE_CLOUD SEG";
        break;      
    case TASKNAME::SMOKE_CLOUD_UNET: //建议阈值0.5
        threshold = 0.5;
        apiName = AlgoAPIName::SMOKE_CLOUD_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::SMOKE_CLOUD_DET_UNET] },
        };
        taskDesc = "SMOKE_CLOUD SEG";
        break;                
    case TASKNAME::BANNER: //建议阈值0.5
        threshold = 0.5;
        apiName = AlgoAPIName::BANNER_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::BANNER_DET] },
        };          
        taskDesc = "BANNER";
        break;
    case TASKNAME::NONCAR: //建议阈值0.6
        threshold = 0.6;
        apiName = AlgoAPIName::NONCAR_DETECTOR;
        taskDesc = "NONCAR";
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::MOTOR_DET] },
        };          
        break;
    case TASKNAME::FIGHT: //建议阈值0.8
        threshold = 0.8;    
        apiName = AlgoAPIName::ACTION_CLASSIFIER;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::ACTION_CLS] },
        };           
        taskDesc = "FIGHT";
        use_batch = 8;
        break;
    case TASKNAME::GKPW: //建议阈值0.4
        threshold = 0.2;
        apiName = AlgoAPIName::MOD_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::MOD_DET_DIF] },
        };          
        taskDesc = "GKPW";
        use_batch = 2;
        break;
    case TASKNAME::GKPW2: //建议阈值0.4
        threshold = 0.6;
        apiName = AlgoAPIName::MOD_MOG2_DETECTOR;    
        taskDesc = "GKPW_MOG";
        break;
    case TASKNAME::HEAD:
        threshold = 0.6;
        apiName = AlgoAPIName::HEAD_DETECTOR;
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::HEAD_DET] },
        };   
        taskDesc = "HEAD DET";
        break;
    case TASKNAME::SOS://推荐阈值0.6
        threshold = 0.5;
        apiName = AlgoAPIName::SOS_DETECTOR;
        nms_threshold = 0.6;//trival in this task
        init_param = { 
            {InitParam::BASE_MODEL, cambricon_model_file[MODELFILENAME::GENERAL_DET_MODE4] },
            {InitParam::SUB_MODEL,  cambricon_model_file[MODELFILENAME::HAND_DET_736x416]},
        };
        taskDesc = "SOS DETECTION";
        break;

    case TASKNAME::ABANDON_OBJECT:
        // cout<<"step into taskname ::abandon obejct"<<endl;
        threshold = 0.2;
        apiName = AlgoAPIName::ABANDON_OBJECT_DETECTOR;
        nms_threshold = 0.2;
        init_param = {
            {InitParam::BASE_MODEL,  cambricon_model_file[MODELFILENAME::ABANDON_OBJECT_DET]},
        };
        taskDesc = "ABANDON OBJECT ";
        break;
    case TASKNAME::JSON:
        taskDesc = "USER_DEFINED_JSON";
        break;
    default:
        retcode = false;
        break;
    }

    if(retcode){
        printf("*taskid %d: %s, threshold(nms)=%1.2f (%1.2f)\n", int(taskid), taskDesc.c_str(), threshold, nms_threshold);
        // std::cout << "TASK: " << taskDesc << ", threshold = " << threshold << ", nms_threshold = " << nms_threshold << std::endl;
        if(!displayTask) {
            for(auto &&param: init_param)
                printf("  |__ (%d) %s\n", param.first, param.second.c_str());
        }
    } else {
        printf("taskid %d not found\n", int(taskid));
    }
        // std::cout << "cambricon files:" << std::endl;
    // for(auto param: init_param){
    //     std::cout << param.first << "," << param.second << std::endl;
    // }
    if(!displayTask) std::cout << "=============parser end=================" << std::endl;
    return retcode;
}

void print_all_task(){
    for(int i = 0; i < int(TASKNAME::TASK_END); i++){
        float threshold, nms_threshold;
        int use_batch;
        std::map<InitParam, std::string> init_param;
        ucloud::AlgoAPIName algoname;
        task_parser(TASKNAME(i),threshold, nms_threshold,algoname, init_param, use_batch, true);
    }
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




