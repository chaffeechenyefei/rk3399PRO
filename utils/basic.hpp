#ifndef _BASIC_HPP_
#define _BASIC_HPP_

#include <memory.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <mutex>
#include <sys/time.h>
#include <chrono>
#include <map>
#include <vector>

/**
 * 内存池小测试
 */
typedef struct _MemNode{
    void* ptr = nullptr;
    _MemNode* next = nullptr;
} MemNode;


typedef std::function<void*(size_t)> MallocFuncPtr;
typedef std::function<void(void*)> FreeFuncPtr;
class MemPool{
public:
    MemPool(){};
    MemPool(size_t mem_size, int num_of_nodes){init(mem_size, num_of_nodes);};
    void init(size_t mem_size, int num_of_nodes);
    void bind_malloc(MallocFuncPtr fp){mallocFunc = fp;}
    void bind_free(FreeFuncPtr fp){freeFunc = fp; }
    ~MemPool();
    MemNode* malloc();
    void free(MemNode*);
protected:
    MemNode* create(size_t mem_size, int num_of_nodes);
    MallocFuncPtr mallocFunc;
    FreeFuncPtr freeFunc;

    size_t memSize = 0;
    int numOfNodes = 0;
    MemNode* freeNodeHeader = nullptr;
    std::set<MemNode*> occupiedNodes;

    std::mutex _mlock_;
};

/**
 * 时间计量函数
 */
class Timer{
public:
    Timer(){}
    ~Timer();
    void start();
    double end(std::string title);
private:
    std::vector<std::chrono::time_point<std::chrono::system_clock>> tQue;
};

/**
 * String字符操作
 * 是否以XXX开头(结尾) 
 */
bool hasEnding(std::string const &fullString, std::string const &ending);
bool hasBegining(std::string const &fullString, std::string const & begining );
/**
 * 列出目录下结尾是xxx的文件名.
 */
void ls_files( const std::string &dir_name, std::vector<std::string> &filenames, const std::string& endswith );
/*
 函数说明：对字符串中所有指定的子串进行替换
 参数：
string resource_str            //源字符串
string sub_str                //被替换子串
string new_str                //替换子串
返回值: string
 */
std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str);





// 图像格式
typedef enum {
    BGR = 0, /*default keep BGR*/
    YUV2BGR_NV21,                     /* YUV420SP_NV12：YYYYYYYY UV UV */
    YUV2RGB_NV21,                      /* 3通道，RGBRGBRGBRGB */
    BGR2RGB,
    RGB2RGB,
}TransformFormat;

void swapYUV_I420toNV12(unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height);
void swapYUV_I420toNV21(unsigned char* i420bytes, unsigned char* nv21bytes, int width, int height);
void BGR2YUV_nv21(cv::Mat src, cv::Mat &dst, int &yuvW, int &yuvH);
void BGR2YUV_nv12(cv::Mat src, cv::Mat &dst, int &yuvW, int &yuvH);
unsigned char* BGR2YUV_nv21_with_stride(cv::Mat src, int &yuvW, int &yuvH, int &stride , int align=64);

void YUV2BGR_n21(unsigned char* nv21bytes,int width, int height, int stride, cv::Mat &dst );

class TransformOp{
public:
    TransformOp();
    ~TransformOp();
    //Transform inputData into BGR or RGB fmt
    int Decode(unsigned char* inputData, int width, int height, int stride, TransformFormat fmt, cv::Mat &dst);
    int Process(unsigned char* inputData, int width, int height, int stride, TransformFormat fmt, \
            int dstW, int dstH, cv::Mat &dst, \
            float &aspect_ratio ,bool appAlphaCh=false, bool keep_aspect_ratio=false, bool padding_both_side = false);
};

//new method for head pose restriction
float dist_p_line(float x0,float y0, float A, float B, float C);
std::vector<float> get_mid_line(float x1,float y1, float x2, float y2);
float dist_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2);
float dist_p_p(float x1,float y1,float x2,float y2);
float ratio_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2);



#endif