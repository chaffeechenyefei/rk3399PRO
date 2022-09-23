#ifndef _MODULE_TRACK_HPP_
#define _MODULE_TRACK_HPP_
#include "module_base.hpp"
#include "trackor/bytetrack_no_reid/BYTETracker.h"

#include <mutex>

using ucloud::BBox;
using ucloud::VecObjBBox;
using ucloud::RET_CODE;
using ucloud::TvaiImage;

/**
 * 跟踪器通用接口
 */
template<class PARAM>
class TrackPoolAPI{
public:
    TrackPoolAPI(){}
    virtual ~TrackPoolAPI(){}
    virtual RET_CODE init(std::string &modelpath){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    virtual void update(TvaiImage &tvimage, VecObjBBox &bboxIN, PARAM& params ){}
    virtual void set_fps(int fps){}
protected:
    std::mutex m_mutex;//Lock should be used in those apis only
};


typedef struct _BYTETRACKPARM{
    float track_threshold;
    float high_detect_threshold;
}BYTETRACKPARM;
/**
 * ByteTrack_No_ReID
 */
typedef std::shared_ptr<bytetrack_no_reid::BYTETracker> ByteTrackNoReID_Ptr;
class ByteTrackNoReIDPool: public TrackPoolAPI<BYTETRACKPARM>{
public:
    ByteTrackNoReIDPool(){}
    ~ByteTrackNoReIDPool(){}
    ByteTrackNoReIDPool(int fps, int nn_buf):\
        m_fps(fps),\
        m_nn_buf(nn_buf) {}

    void set_fps(int fps){m_fps=fps;}
    void update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM &params);
    void add_trackor(int cam_uuid, BYTETRACKPARM params);

private:
    std::map<int, ByteTrackNoReID_Ptr> m_trackors;
    std::map<int, BYTETRACKPARM> m_params;
    int m_fps = 25;
    int m_nn_buf = 30;
    // std::mutex m_mutex;
};



#endif