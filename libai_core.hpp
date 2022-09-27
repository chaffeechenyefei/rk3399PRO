/**
 * libai_core_220906 RK3399PRO
 * chaffee.chen@ucloud.cn
 */
#ifndef _LIBAI_CORE_HPP_
#define _LIBAI_CORE_HPP_

#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <map>

#if __GNUC__ >= 4
    #ifdef UCLOUD_EXPORT
        #define UCLOUD_API_PUBLIC __attribute__((visibility ("default")))
        #define UCLOUD_API_LOCAL __attribute__((visibility("hidden")))
    #else
        #define UCLOUD_API_PUBLIC
        #define UCLOUD_API_LOCAL
    #endif
#else
    #error "##### requires gcc version >= 4.0 #####"
#endif

#define TEST 1

namespace ucloud{
//返回值
typedef enum _RET_CODE{
    SUCCESS                     = 0, //成功
    FAILED                      = 1, //未知失败

    ERR_MODEL_FILE_NOT_EXIST    = 3, //模型文件不存在
    ERR_INIT_PARAM_FAILED       = 4, //参数初始化失败
    ERR_UNSUPPORTED_IMG_FORMAT  = 5, //图像输入格式不支持
    ERR_MODEL_NOT_MATCH         = 6, //载入的模型有问题, 无法推理
    ERR_MODEL_NOT_INIT          = 7,  //模型没有被加载, 无法推理, 请先调用成员函数init()
    ERR_OUTPUT_CLS_INIT_FAILED  = 8,  //检测模型输出类型(CLS_TYPE)绑定失败
    ERR_BATCHSIZE_NOT_MATCH     = 9,  //输入数据batchsize大小和模型不一致
    ERR_PHYADDR_EMPTY           = 10, //物理地址空
    ERR_VIRTUAL_FUNCTION        = 11, //虚函数, 该类不支持该接口
    ERR_EMPTY_BOX               = 12, //输入的BOX为空
    ERR_POST_EXE                = 13, //后处理程序错误

    ERR_NPU_INIT_FAILED         = 100, //NPU初始化失败
    ERR_NPU_QUERY_FAILED        = 101, //NPU query调用失败
    ERR_NPU_IOSET_FAILED        = 102, //NPU输入输出设置失败
    ERR_NPU_RUN_FAILED          = 103, //NPU运行推理失败
    ERR_NPU_GET_OUTPUT_FAILED   = 104, //NPU获取推理输出失败
    ERR_NPU_MEM_ERR             = 105, //NPU地址分配失败
    ERR_NPU_SYNC_NOT_MATCH      = 106, //NPU使用了input map但没有使用相应的sync
}RET_CODE;

//目标类别
typedef enum _CLS_TYPE{
    PEDESTRIAN                  = 0     ,   //行人
    FACE                        = 1     ,   //人脸
    PEDESTRIAN_FALL             = 2     ,   //摔倒的行人
    HAND                        = 3     ,   //人手检测

    CAR                         = 10    ,   //车辆
    NONCAR                      = 100   ,   //非机动车
    BYCYCLE                             ,   //自行车
    EBYCYCLE                            ,   //电瓶车

    PET                         = 200   ,   //宠物
    PET_DOG                             ,   //宠物狗
    PET_CAT                             ,   //宠物猫

    WATER_PUDDLE                = 300   ,   //积水
    TRASH_BAG                           ,   //垃圾袋
    BANNER                              ,   //横幅标语
    FIRE                                ,   //火焰
    //行为识别
    FIGHT                       = 400   ,   //打架行为
    SMOKING                     = 410   ,   //抽烟
    PHONING                     = 411   ,   //打电话或玩手机
    //安全帽
    PED_HEAD                    = 500   ,   //头
    PED_SAFETY_HAT                      ,   //安全帽
    //高空抛物
    FALLING_OBJ                 = 600   ,   //高空抛物--轨迹确定
    FALLING_OBJ_UNCERTAIN               ,   //高空抛物--轨迹未确定
    //车牌检测
    LICPLATE_BLUE               = 700   ,   //蓝牌
    LICPLATE_SGREEN                     ,   //小型新能源车（纯绿）
    LICPLATE_BGREEN                     ,   //大型新能源车（黄加绿）
    LICPLATE_YELLOW                     ,   //黄牌

    OTHERS                      = 900   ,   //其它类别, 相当于占位符
    OTHERS_A                    = 901   ,   //自定义占位符
    OTHERS_B                    = 902   ,
    OTHERS_C                    = 903   ,
    OTHERS_D                    = 904   ,

    UNKNOWN                     = 1000  ,   //未定义
}CLS_TYPE;

//目标特征值
typedef struct TvaiFeature_S
{
    unsigned int          featureLen = 0;        /* 特征值长度, 字节数 */
    unsigned char         *pFeature = nullptr;         /* 特征值指针 */
}TvaiFeature;

// 分辨率, 用于设定检测目标的大小范围
typedef struct TvaiResolution_S {
    unsigned int     width;                 /* 宽度 */
    unsigned int     height;                /* 高度 */
}TvaiResolution;

// 通用矩形框
typedef struct TvaiRect_S
{
    int     x;          /* 左上角X坐标 */
    int     y;          /* 左上角Y坐标 */
    int     width;      /* 区域宽度 */
    int     height;     /* 区域高度 */
}TvaiRect;

//骨架关键点
typedef struct _SkeletonLandmark{
    float x[17];
    float y[17];
} SkLandmark;

//人脸五官关键点坐标
typedef struct _FaceLandmark { //Left eye, Right eye, Nose, Left mouth, Right mouth
  float x[5];
  float y[5];
} FaceLandmark;

//人脸检测返回的检测框信息
typedef struct _FaceInfo {
    float x0;
    float y0;
    float x1;
    float y1;
    TvaiRect rect; //tvai
    float confidence; //tvai
    FaceLandmark landmark;
} FaceInfo;

//通用关键点类型
typedef enum _LandMarkType{
    FACE_5PTS            =   0, //人脸五点
    SKELETON             =   1, //人体骨架信息
    UNKNOW_LANDMARK      =   10,//未知预留
}LandMarkType;

//关键点坐标参考系
typedef enum _RefCoord{
    IMAGE_ORIGIN        =   0,//图像坐标系, 以图像左上角为坐标原点
    ROI_ORIGIN          =   1,//以ROI区域原点为坐标原点
    HEATMAP_ORIGIN      =   2,//以模型输出的heatmap图像左上角为坐标原点
}RefCoord;

//关键点xy坐标结构体
typedef struct _uPoint{
    float x;
    float y;
    _uPoint(float _x, float _y):x(_x),y(_y){}
    _uPoint(){x=0;y=0;}
} uPoint;

//关键点信息结构
typedef struct _LandMark{
    std::vector<uPoint> pts;
    LandMarkType type;
    RefCoord refcoord;
} LandMark;

//通用返回信息结构
typedef struct _BBox {
    //模型输出:
    float x0;//top left corner x -> model scale
    float y0;//top left corner y -> model scale
    float x1;//bottom right corner x -> model scale
    float y1;//bottom right corner y -> model scale
    float x,y,w,h;//top left corner + width + height -> model scale
    //最终图像输出(由模型输出经过aspect_scale, feature_map缩放得到)
    TvaiRect rect; //tvai -> image scale 最终输出
    float objectness = 0; //物体概率
    float confidence = 0; //tvai 某类别的概率(由模型输出的objectness*confidence得到)或事件概率
    float quality = 0;//图像质量分0-1, 1:高质量图像
    CLS_TYPE objtype = CLS_TYPE::UNKNOWN;
    std::string objname = "unknown";//objtype的文字描述, 在objtype = OTHERS(_A)的情况下, 可以进行透传. 目的: 支持临时算法改动
    std::string desc = "";//json infomation, 目的: 预留, 用于临时情况下将信息以json字段方式输出
    LandMark Pts;//关键点位信息
    TvaiFeature feat;//特征信息
    int track_id = -1;//跟踪唯一标识
    std::vector<float> trackfeat;//跟踪用特征
}BBox;

// 图像格式
typedef enum _TvaiImageFormat{
    TVAI_IMAGE_FORMAT_NULL    = 0,              /* 格式为空 */
    TVAI_IMAGE_FORMAT_GRAY,                     /* 单通道灰度图像 */
    TVAI_IMAGE_FORMAT_NV12,                     /* YUV420SP_NV12：YYYYYYYY UV UV */
    TVAI_IMAGE_FORMAT_NV21,                     /* YVU420SP_NV21：YYYYYYYY VU VU */
    TVAI_IMAGE_FORMAT_RGB,                      /* 3通道，RGBRGBRGBRGB */
    TVAI_IMAGE_FORMAT_BGR,                      /* 3通道，BGRBGRBGRBGR */
    TVAI_IMAGE_FORMAT_I420,                     /* YUV420p_I420 ：YYYYYYYY UU VV */
}TvaiImageFormat;


// 输入图像结构
typedef struct TvaiImage_S
{
    TvaiImageFormat      format;      /* 图像像素格式 */
    int                  width;       /* 图像宽度 */
    int                  height;      /* 图像高度 */
    int                  stride;      /* 图像水平跨度 */
    unsigned char        *pData = nullptr; /* 图像数据。*/
    int                  dataSize;    /* 图像数据的长度 */
    uint64_t             u64PhyAddr[3]={0}; /* 数据的物理地址 */
    bool                 usePhyAddr=false;  /* 是否使用数据的物理地址 */
    int                  uuid_cam=-1;/*图像设备来源唯一标号, 用于上下文相关的任务*/

    TvaiImage_S(TvaiImageFormat _fmt, int _width, int _height, int _stride, unsigned char* _pData, int _dataSize, int _uuid_cam=-1):\
        format(_fmt), width(_width), height(_height), stride(_stride), pData(_pData), dataSize(_dataSize), uuid_cam(_uuid_cam){}
    TvaiImage_S(){
        format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NULL;
        width = 0;
        height = 0;
        stride = 0;
        dataSize = 0;
        uuid_cam = -1;
    }
}TvaiImage;

//自定义简称
typedef std::vector<TvaiRect> VecRect;
typedef std::vector<FaceInfo> VecFaceBBox;
typedef std::vector<TvaiFeature> VecFeat;
typedef std::vector<BBox> VecObjBBox;
typedef std::vector<SkLandmark> VecSkLandmark;
typedef std::vector<TvaiImage> BatchImageIN;
typedef std::vector<VecObjBBox> BatchBBoxOUT;
typedef std::vector<VecObjBBox> BatchBBoxIN;
////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 方法,工厂类,枚举类型
////////////////////////////////////////////////////////////////////////////////////////////////////////

//算法功能的枚举
typedef enum _AlgoAPIName{
    FACE_DETECTOR       = 0,//人脸检测
    FACE_EXTRACTOR      = 1,//人脸特征提取
    GENERAL_DETECTOR    = 2,//通用物体检测器即yolodetector, 可用于人车非 return PEDESTRIAN, CAR, NONCAR
    ACTION_CLASSIFIER   = 3,//行为识别, 目前支持打斗 [需要数据更新模型] x
    MOD_DETECTOR        = 4,//高空抛物, Moving Object Detection(MOD)[需要改善后处理, 开放做多帧接口测试]
    PED_DETECTOR        = 5,//行人检测加强版, 针对摔倒进行数据增强, mAP高于人车非中的人 
    FIRE_DETECTOR       = 6,//火焰检测
    FIRE_DETECTOR_X     = 7,//火焰检测加强版, 带火焰分类器
    WATER_DETECTOR      = 8,//积水检测 x
    PED_FALL_DETECTOR   = 9,//行人摔倒检测, 只检测摔倒的行人
    SKELETON_DETECTOR   = 10,//人体骨架/关键点检测器--后续对接可用于摔倒检测等业务 x
    SAFETY_HAT_DETECTOR = 11,//安全帽检测 return PED_SAFETY_HAT, PED_HEAD
    TRASH_BAG_DETECTOR  = 12,//垃圾袋检测 x
    BANNER_DETECTOR     = 13,//横幅检测 x
    NONCAR_DETECTOR     = 14,//非机动车检测加强版, 针对非机动车进电梯开发 
    SMOKING_DETECTOR    = 15,//抽烟行为检测 x
    PHONING_DETECTOR    = 16,//打电话/玩手机行为检测 x
    HEAD_DETECTOR       = 17,//人头检测, 检测画面中人头数量, 用于密集场景人数统计
// #ifndef MLU220 //新增内容2022-03-03
    SOS_DETECTOR        = 18,//SOS举手求救
    PED_SK_DETECTOR     = 19,//行人弯腰检测，测试环节[本质是行人检测+骨架检测]
    FACE_DETECTOR_ATTR  = 20,//人脸检测
    GENERAL_DETECTORV2  = 21,//跟踪器替代
    LICPLATE_DETECTOR   = 22, //车牌检测
    LICPLATE_RECOGNIZER = 23, //车牌识别
// #endif
    RESERVED1           = 41,//yingxun保留
    RESERVED2           = 42,
    RESERVED3           = 43,
    UDF_JSON            = 5000, //用户自定义json输入
    //=========内部使用======================================================================
    GENERAL_TRACKOR     = 50,//通用跟踪模块, 不能实例化, 但可以在内部使用
    MOD_MOG2_DETECTOR   = 51,//高空抛物, Moving Object Detection(MOD)[MoG2版本]
    HAND_DETECTOR       = 52,//人手检测 224x320, 一般用于内部, 不单独使用
    HAND_L_DETECTOR     = 53,//人手检测 736x416, 一般用于内部, 不单独使用
    SMOKING_CLASSIFIER  = 54,//抽烟行为分类
    BATCH_GENERAL_DETECTOR    = 100,//测试用
    FIRE_CLASSIFIER                ,//火焰分类, 内部测试用
    WATER_DETECTOR_OLD      = 1008,//积水检测(旧版unet,与新版之间存在后处理的逻辑差异)
}AlgoAPIName;

typedef enum _InitParam{
    BASE_MODEL          = 0, //基础模型(检测模型/特征提取模型/分类模型)
    TRACK_MODEL         = 1, //跟踪模型
    SUB_MODEL           = 2, //模型级联时, 主模型用于初步检测, 次模型用于二次过滤, 提高精度
}InitParam;

typedef enum _APIParam{
    OBJ_THRESHOLD      = 0, //目标检测阈值/分类阈值
    NMS_THRESHOLD      = 1, //NMS检测阈值
    MAX_OBJ_SIZE       = 2, //目标最大尺寸限制
    MIN_OBJ_SIZE       = 3, //目标最小尺寸限制
    VALID_REGION       = 4, //有效检测区域设定
}APIParam;

//算法对外暴露的接口形式
class UCLOUD_API_PUBLIC AlgoAPI{
public:
    AlgoAPI(){};
    virtual ~AlgoAPI(){};
    virtual int get_batchsize(){return 1;}
    /**
     * 带跟踪模块的初始化方式
     */
    virtual RET_CODE init(){return RET_CODE::ERR_VIRTUAL_FUNCTION;}                            
    /**
     * 新版本的初始化方式
     */
    virtual RET_CODE init(std::map<InitParam, std::string> &modelpath){return RET_CODE::ERR_VIRTUAL_FUNCTION;}

    // virtual RET_CODE set_param(float threshold, float nms_threshold){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    /** ALL_IN_ONE 
     * general detection(including face/skeleton/ped_car_non_car detection) and face feature extraction
     * */
    virtual RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    /**
     * 高空抛物已改单帧推理模式, 多帧推理接口仍保留可使用.
     * chaffee@2022-05-17
    */
    virtual RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
        //接口兼容:兼容单帧输入的情况@2022-02-17
        if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
        else return run(batch_tvimages[0], bboxes);
    }
    /**
     * 返回检测的类别, 或返回适用的类别
     */
    virtual RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss){return RET_CODE::ERR_VIRTUAL_FUNCTION;};
};
typedef std::shared_ptr<AlgoAPI> AlgoAPISPtr;

/**
 * AICoreFactory
 * DESC: 获得算法功能句柄的唯一途径
 */
class UCLOUD_API_PUBLIC AICoreFactory{
public:
    AICoreFactory();
    ~AICoreFactory();
    static AlgoAPISPtr getAlgoAPI(AlgoAPIName apiName);
    //用户自定义json配置文件导入初始化算法
    static AlgoAPISPtr getAlgoAPI(const std::string &configpath);
};

UCLOUD_API_PUBLIC unsigned char* readImg_to_RGB(std::string filepath, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_BGR(std::string filepath, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV21(std::string filepath, int &width, int &height, int &stride);
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV12(std::string filepath, int &width, int &height, int &stride);
/**
 * 读入图像, 并转为yuv数据, 同时进行缩放
 * PARA:
 * w :将输入图像resize到w/h尺寸
 * h :将输入图像resize到w/h尺寸
 * width :yuv能适配的尺寸
 * height :yuv能适配的尺寸
 */
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV21(std::string filepath, int w, int h,int &width, int &height, int &stride);
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV12(std::string filepath, int w, int h,int &width, int &height, int &stride);
//写图像, 是否采用覆盖式写入
UCLOUD_API_PUBLIC void writeImg(std::string filepath , unsigned char* imgPtr, int width, int height, bool overwrite=true);
UCLOUD_API_PUBLIC void freeImg(unsigned char** imgPtr);
// use_rand_color: 在没有trackid的时候, 是否使用random color或统一的[0,255,0]
UCLOUD_API_PUBLIC void drawImg(unsigned char* img, int width, int height, VecObjBBox &bboxs, \
        bool disp_landmark=false ,bool disp_label=false, bool use_rand_color=true, int color_for_trackid_or_cls = 0);
//读取yuv和rgb的二进制文件流, 便于测试
UCLOUD_API_PUBLIC unsigned char* yuv_reader(std::string filename, int w=1920, int h=1080);
UCLOUD_API_PUBLIC unsigned char* rgb_reader(std::string filename, int w=1920, int h=1080);        
//视频读取基于opencv
class UCLOUD_API_PUBLIC VIDOUT{
public:
    VIDOUT(){}
    ~VIDOUT(){release();}
    VIDOUT(const VIDOUT &obj)=delete;
    VIDOUT& operator=(const VIDOUT & rhs)=delete;
    unsigned char* bgrbuf=nullptr;
    unsigned char* yuvbuf=nullptr;
    int w,h,s;//yuv
    int _w,_h;//bgr
    void release(){
        if(bgrbuf!=nullptr) free(bgrbuf);
        if(yuvbuf!=nullptr) free(yuvbuf);
    }
};
class UCLOUD_API_PUBLIC vidReader{
public:
    vidReader(){}
    ~vidReader(){release();}
    bool init(std::string filename);
    unsigned char* getbgrImg(int &width, int &height);
    unsigned char* getyuvImg(int &width, int &height, int &stride);
    VIDOUT* getImg();
    int len(){return m_len;}
    int width();
    int height();
    int fps();
private:
    void release();
    void* handle_t=nullptr;
    int m_len = 0;
};
class UCLOUD_API_PUBLIC vidWriter{
public:
    vidWriter(){};
    ~vidWriter(){release();}
    bool init(std::string filename, int width, int height, int fps);
    void writeImg(unsigned char* buf, int bufw, int bufh);
private:
    void release();
    void* handle_t=nullptr;
    int m_width;
    int m_height;
    int m_fps;
};

class UCLOUD_API_PUBLIC Clocker{
public:
    Clocker();
    ~Clocker();
    void start();
    double end(std::string title, bool display=true);//return seconds
private:
    void* ctx;
};
/**
 * 20210917
 * 以下是历史遗留产物, 后期不再更新维护.
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////
//特征比对, 可改用blas加速计算
////////////////////////////////////////////////////////////////////////////////////////////////////////
inline float _calcSimilarity(float* fA, float* fB, int dims){
    float val = 0;
    for(int i=0 ; i < dims ; i++ )
        val += (*fA++) * (*fB++);
    return val;
};


inline void calcSimilarity(VecFeat& featA,VecFeat& featB, std::vector<std::vector<float>>& result){
    std::vector<std::vector<float>>().swap(result);
    result.clear();
    for( int a = 0 ; a < featA.size() ; a++ ){
        float* fA = reinterpret_cast<float*>(featA[a].pFeature);
        int dims = featA[a].featureLen/sizeof(float);
        std::vector<float> inner_reuslt;
        for ( int b = 0; b < featB.size() ; b++ ){
            float* fB = reinterpret_cast<float*>(featB[b].pFeature);
            float val = _calcSimilarity(fA, fB, dims);
            inner_reuslt.push_back(val);
        }
        result.push_back(inner_reuslt);
    }
};

};//namespace ucloud



#endif