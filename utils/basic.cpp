#include "basic.hpp"
#include "glog/logging.h"
#include <dirent.h>
#include <sys/stat.h>

#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif

MemPool::~MemPool(){
    MemNode* nodePtr=nullptr;
    while(freeNodeHeader){
        nodePtr = freeNodeHeader->next;
        freeFunc(freeNodeHeader->ptr);
        freeNodeHeader = nodePtr;
    }
    if(!occupiedNodes.empty()){
        LOGI << "Error: nodes in mem pool is still occupied!";
        for(auto &node: occupiedNodes){
            freeFunc(node);
        }
        LOGI << "manually released";
    }
}

void MemPool::init(size_t mem_size, int num_of_nodes){
    LOGI << "-> creating nodes in MemPool::init()";
    mallocFunc = ::malloc;
    freeFunc = ::free;
    memSize = mem_size;
    freeNodeHeader = create(mem_size, num_of_nodes);
    LOGI << "<- creating nodes in MemPool::int()";
}

MemNode* MemPool::create(size_t mem_size, int num_of_nodes){
    MemNode* curFreeNodeHeader = nullptr;
    for(int i=0; i< num_of_nodes; i++){
        MemNode* newNode = new MemNode;
        newNode->ptr = mallocFunc(mem_size);
        if(curFreeNodeHeader){
            MemNode* tmpNode;
            tmpNode = curFreeNodeHeader;
            curFreeNodeHeader = newNode;
            newNode->next = tmpNode;
        }else{
            curFreeNodeHeader = newNode;
        }
    }
    numOfNodes += num_of_nodes;
    return curFreeNodeHeader;
}

MemNode* MemPool::malloc(){
    std::lock_guard<std::mutex> lock(_mlock_);
    if(freeNodeHeader==nullptr){
        //TODO需要加开空间
        LOGI << "creating more nodes";
        freeNodeHeader = create(memSize, numOfNodes);
    }
    MemNode* freeNode = freeNodeHeader;
    freeNodeHeader = freeNodeHeader->next;
    occupiedNodes.insert(freeNode);
    LOGI << "used nodes = " << occupiedNodes.size();
    return freeNode;
}

void MemPool::free(MemNode* nodePtr){
    std::lock_guard<std::mutex> lock(_mlock_);
    if(nodePtr!=nullptr){
        nodePtr->next = freeNodeHeader;
        freeNodeHeader = nodePtr;
        occupiedNodes.erase(nodePtr);
        LOGI << "remaining used nodes = " << occupiedNodes.size();
    }
}

/**
 * Timer
 */
void Timer::start(){
    auto t = std::chrono::system_clock::now();
    tQue.push_back(t);
}

double Timer::end(std::string title){
    auto t = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> start;
    if(!tQue.empty()){
        start = *(tQue.rbegin());
        tQue.pop_back();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t-start);
        double tm_cost = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
        printf("[%s] cost = %fs\n", title.c_str() ,tm_cost);
        return tm_cost;
    } else {
        printf("%s skipped...\n", title.c_str());
        return 0;
    }
}

Timer::~Timer(){
    tQue.clear();
}



/**
 * https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
 */
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}
/**
 * https://stackoverflow.com/questions/1878001/how-do-i-check-if-a-c-stdstring-starts-with-a-certain-string-and-convert-a
 */
bool hasBegining(std::string const &fullString, std::string const & begining ) {
    if (fullString.rfind(begining, 0) == 0) { // pos=0 limits the search to the prefix
    // s starts with prefix
        return true;
    } else return false;
}


/* Show all files under dir_name , do not show directories ! */
void ls_files( const std::string &dir_name, std::vector<std::string> &filenames, const std::string& endswith )
{
	// check the parameter !
	if( dir_name.empty() || dir_name == "" )
		return;
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name.c_str() , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		std::cout<<"dir_name is not a valid directory !"<< std::endl;
		return;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( dir_name.c_str() );
	if( NULL == dir )
	{
		std::cout<<"Can not open dir "<<dir_name<<std::endl;
		return;
	}
	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
        std::string tmp(filename->d_name);
        if(endswith.empty() || endswith == "" )
            filenames.push_back(tmp);
        else{
            if(hasEnding(tmp, endswith)) filenames.push_back(tmp);
        }
		// std::cout<<filename ->d_name <<std::endl;
	}
    closedir(dir);
} 


/*
 函数说明：对字符串中所有指定的子串进行替换
 参数：
string resource_str            //源字符串
string sub_str                //被替换子串
string new_str                //替换子串
返回值: string
 */
std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
{
    std::string dst_str = resource_str;
    std::string::size_type pos = 0;
    while(( pos = dst_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
    {
        dst_str.replace(pos, sub_str.length(), new_str);
    }
    return dst_str;
}

void swapYUV_I420toNV12(unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height)
{
    int nLenY = width * height;
    int nLenU = nLenY / 4;

    memcpy(nv12bytes, i420bytes, width * height);

    for (int i = 0; i < nLenU; i++)
    {
        nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];                    // U
        nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];        // V
    }
}

void swapYUV_I420toNV21(unsigned char* i420bytes, unsigned char* nv21bytes, int width, int height)
{
    int nLenY = width * height;
    int nLenU = nLenY / 4;

    memcpy(nv21bytes, i420bytes, width * height);

    for (int i = 0; i < nLenU; i++)
    {
        nv21bytes[nLenY + 2 * i] = i420bytes[nLenY + nLenU + i];    // V
        nv21bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + i];       // U
    }
}

void BGR2YUV_nv12(cv::Mat src, cv::Mat &dst, int &yuvW, int &yuvH)
{
    int H = src.rows;
    int W = src.cols;
    int padH = H + ( H%2!=0 ? 1: 0 );
    int padW = W + ( W%2!=0 ? 1: 0 );
    cv::Mat padim = cv::Mat::zeros(cv::Size(padW,padH), src.type());
    src.copyTo(padim(cv::Rect(0,0,W,H)));

    dst = cv::Mat(padH*1.5, padW, CV_8UC1, cv::Scalar(0));
    cv::Mat src_YUV_I420(padH*1.5, padW, CV_8UC1, cv::Scalar(0));  //YUV_I420
    cv::cvtColor(padim, src_YUV_I420, CV_BGR2YUV_I420);
    swapYUV_I420toNV12(src_YUV_I420.data, dst.data, padW, padH);

    yuvW = padW;
    yuvH = padH;
}

void BGR2YUV_nv21(cv::Mat src, cv::Mat &dst, int &yuvW, int &yuvH)
{
    int H = src.rows;
    int W = src.cols;
    int padH = H + ( H%2!=0 ? 1: 0 );
    int padW = W + ( W%2!=0 ? 1: 0 );
    cv::Mat padim = cv::Mat::zeros(cv::Size(padW,padH), src.type());
    src.copyTo(padim(cv::Rect(0,0,W,H)));

    dst = cv::Mat(padH*1.5, padW, CV_8UC1, cv::Scalar(0));
    cv::Mat src_YUV_I420(padH*1.5, padW, CV_8UC1, cv::Scalar(0));  //YUV_I420
    cv::cvtColor(padim, src_YUV_I420, CV_BGR2YUV_I420);
    swapYUV_I420toNV21(src_YUV_I420.data, dst.data, padW, padH);

    yuvW = padW;
    yuvH = padH;
}

void YUV2BGR_n21(unsigned char* nv21bytes,int width, int height, int stride, cv::Mat &dst ){
    if(stride==width){
        // memcpy(ptrDst, nv21bytes, 3*width*height/2*sizeof(uchar));
        cv::Mat yuvn21Mat(3*height/2, width, CV_8UC1, nv21bytes );
        cv::cvtColor(yuvn21Mat, dst, cv::COLOR_YUV2BGR_NV21);
    } else {
        cv::Mat yuvn21Mat(3*height/2, width, CV_8UC1, cv::Scalar(0));
        uchar* ptrDst = yuvn21Mat.data;
        for( int i = 0 ; i < 3*height/2 ; i++ ){
            uchar* _ptrSrc = nv21bytes + stride*i;
            uchar* _ptrDst = ptrDst + width*i;
            memcpy(_ptrDst, _ptrSrc, width );
        }
        cv::cvtColor(yuvn21Mat, dst, cv::COLOR_YUV2BGR_NV21);
    }
}


unsigned char* BGR2YUV_nv21_with_stride(cv::Mat src, int &yuvW, int &yuvH, int &stride , int align){
    int H = src.rows;
    int W = src.cols;
    int padH = H + ( H%2!=0 ? 1: 0 );
    int padW = W + ( W%2!=0 ? 1: 0 );
    cv::Mat padim = cv::Mat::zeros(cv::Size(padW,padH), src.type());
    src.copyTo(padim(cv::Rect(0,0,W,H)));

    cv::Mat _dst = cv::Mat(padH*1.5, padW, CV_8UC1, cv::Scalar(0));
    cv::Mat src_YUV_I420(padH*1.5, padW, CV_8UC1, cv::Scalar(0));  //YUV_I420
    cv::cvtColor(padim, src_YUV_I420, CV_BGR2YUV_I420);
    swapYUV_I420toNV21(src_YUV_I420.data, _dst.data, padW, padH);

    int dst_stride = padW;
    dst_stride = std::ceil(1.0 * dst_stride / align) * align;  // align stride to 64 by default

    unsigned char* dst = (uchar*)malloc(3*dst_stride*padH/2);

    if(!_dst.isContinuous()){
        std::cout << "[BGR2YUV_nv21_with_stride] _dst not continuous" << std::endl;
    }

    for (int i = 0 ; i < 3*padH/2; i++){
        uchar* ptr_src = _dst.data + i*padW;
        uchar* ptr_dst = dst + i*dst_stride;
        memcpy(ptr_dst, ptr_src, padW);
    }
    yuvW = padW;
    yuvH = padH;
    stride = dst_stride;
    return dst;
}


/////////////////////////////////////////////////////////////////////
// Class TransformOp 
/////////////////////////////////////////////////////////////////////
TransformOp::TransformOp(){}

TransformOp::~TransformOp(){}

int TransformOp::Decode(unsigned char* inputData, int width, int height, int stride, TransformFormat fmt, cv::Mat &dst){
    stride = MAX(width, stride);
    cv::Mat inputMat(height, stride, CV_8UC3, inputData);
    if(fmt == TransformFormat::BGR ){
        inputMat.copyTo(dst);
    } else if( fmt == TransformFormat::BGR2RGB ){
        cv::cvtColor(inputMat, dst, cv::COLOR_BGR2RGB );
    } else if( fmt == TransformFormat::YUV2BGR_NV21 || fmt == TransformFormat::YUV2RGB_NV21 ){
        uchar* databuf_no_stride = nullptr;
        if (width == stride){
            databuf_no_stride = inputData;
        } else {
            databuf_no_stride = (uchar*)malloc(3*width*height/2);
            uchar* tmpSrc = inputData;
            uchar* tmpDst = databuf_no_stride;
            for(int i = 0 ; i < 3*height/2 ; i++ ){
                memcpy(tmpDst, tmpSrc, stride);
                tmpDst += width;
                tmpSrc += stride;
            }
        }
        cv::Mat yuvMat(3*height/2, width, CV_8UC1 , databuf_no_stride);
        if (fmt == TransformFormat::YUV2BGR_NV21)
            cv::cvtColor(yuvMat, dst, cv::COLOR_YUV2BGR_NV21);
        else
            cv::cvtColor(yuvMat, dst, cv::COLOR_YUV2RGB_NV21);
            
        if (width != stride)
            free(databuf_no_stride);
    } else {
        return -1;
    }
    return 1;
}

int TransformOp::Process(unsigned char* inputData, int width, int height, int stride, TransformFormat fmt, \
        int dstW, int dstH, cv::Mat &dst, \
        float &aspect_ratio ,bool appAlphaCh, bool keep_aspect_ratio, bool padding_both_side){
    cv::Mat decodeMat, resizedMat;
    cv::Mat dst_no_alpha = cv::Mat::zeros(dstH,dstW,CV_8UC3);
    int decode_ret = this->Decode(inputData, width, height, stride, fmt, decodeMat);
    // cv::imwrite("2.png", decodeMat);
    if (decode_ret <= 0 )
        return -1;
    // for the sake of stride in RGB/BGR situation
    width = decodeMat.cols;
    height = decodeMat.rows;

    aspect_ratio = 1.0;
    if (keep_aspect_ratio){
        float w_aspect_ratio = (1.0*dstW)/width;
        float h_aspect_ratio = (1.0*dstH)/height;
        aspect_ratio = MIN( w_aspect_ratio, h_aspect_ratio );
        int _H = MIN(aspect_ratio*height, dstH);
        int _W = MIN(aspect_ratio*width, dstW);
        cv::resize(decodeMat, resizedMat, cv::Size(_W,_H));
        if (padding_both_side){
            if ( w_aspect_ratio > h_aspect_ratio ){
                int padSz = dstW - _W;
                resizedMat.copyTo(dst_no_alpha(cv::Rect(padSz/2,0,_W,_H)));
            } else {
                int padSz = dstH - _H;
                resizedMat.copyTo(dst_no_alpha(cv::Rect(0,padSz/2,_W,_H)));
            }
        } else {
            resizedMat.copyTo(dst_no_alpha(cv::Rect(0,0,_W,_H)));
        }
    } else{
        cv::resize(decodeMat, dst_no_alpha, cv::Size(dstW,dstH));
    }


    if( appAlphaCh ){
        switch (fmt)
        {
        case TransformFormat::BGR:
            cv::cvtColor(dst_no_alpha,dst,cv::COLOR_BGR2BGRA);
            break;
        case TransformFormat::BGR2RGB:
            cv::cvtColor(dst_no_alpha,dst,cv::COLOR_RGB2RGBA);
            break;
        case TransformFormat::YUV2BGR_NV21:
            cv::cvtColor(dst_no_alpha,dst,cv::COLOR_BGR2BGRA);
            break;
        case TransformFormat::YUV2RGB_NV21:
            cv::cvtColor(dst_no_alpha,dst,cv::COLOR_RGB2RGBA);
            break;
        default:
            break;
        }
    } else{
        dst_no_alpha.copyTo(dst);
    }

    if (!dst.isContinuous())
        LOG(INFO) << "make continous";
        // dst = dst.clone();

    return 1;
}
/////////////////////////////////////////////////////////////////////
// End of Class TransformOp 
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
// Head Pose Restriction
/////////////////////////////////////////////////////////////////////
//new method for head pose restriction
float dist_p_line(float x0,float y0, float A, float B, float C){
/*
:param p: [x0,y0]
:param line: Ax+By+C=0 [A,B,C]
:return: d = | Ax0+By0+C | / sqrt(A^2+B^2)
*/
    float dm = fabsf(A*x0+B*y0+C);
    float dn = sqrtf(A*A+B*B+1e-3);
    return dm/dn;
}

std::vector<float> get_mid_line(float x1,float y1, float x2, float y2){
    float A = x1 - x2;
    float B = y1 - y2;
    float C = -(y1*y1-y2*y2 + x1*x1 - x2*x2)/2;
    std::vector<float> res;
    res.push_back(A);
    res.push_back(B);
    res.push_back(C);
    return res;
}

float dist_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2){
    std::vector<float> mid_line = get_mid_line(x1,y1,x2,y2);
    float A = mid_line[0];
    float B = mid_line[1];
    float C = mid_line[2];
    float d = dist_p_line(x0,y0,A,B,C);
    return d;
}

float dist_p_p(float x1,float y1,float x2,float y2){
    float d = sqrtf( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
    return d;
}

float ratio_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2){
    float dp1p2 = dist_p_p(x1,y1,x2,y2);
    float dp0line = dist_p_mid_line(x0,y0,x1,y1,x2,y2);
    return dp0line/(dp1p2+1e-3);
}
/////////////////////////////////////////////////////////////////////
// End of Head Pose Restriction
/////////////////////////////////////////////////////////////////////