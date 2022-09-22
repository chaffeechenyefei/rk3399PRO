#ifndef _MODULE_TRACK_HPP_
#define _MODULE_TRACK_HPP_
#include "module_base.hpp"
#include "trackor/bytetrack_no_reid/BYTETracker.h"

#include <mutex>

using ucloud::BBox;
using ucloud::VecObjBBox;
using ucloud::RET_CODE;
using ucloud::TvaiImage;

// /**
//  * 跟踪器通用接口
//  */
// class TrackorAPI{
// public:
//     TrackorAPI(){}
//     virtual ~TrackorAPI(){}
//     virtual RET_CODE init(std::string &modelpath){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
//     virtual void update(TvaiImage &tvimage, VecObjBBox &bboxIN, std::vector<float> &thresholds){}
//     virtual void set_fps(int fps){}
//     // virtual void set_threshold(std::vector<float> &thresholds){}
//     // virtual void update(VecObjBBox &bboxIN, int cam_uuid){}
// protected:
//     std::mutex m_mutex;//Lock should be used in those apis only
// };
// typedef std::shared_ptr<TrackorAPI> Trackor_Ptr;

// enum class BYTETRACK_THRESHOLD{
//     TRACK_THRESHOLD,
//     HIGH_DETECT_THRESHOLD,
// };
typedef struct _BYTETRACKPARM{
    float track_threshold;
    float high_detect_threshold;
}BYTETRACKPARM;
/**
 * ByteTrack_No_ReID
 */
typedef std::shared_ptr<bytetrack_no_reid::BYTETracker> ByteTrackNoReID_Ptr;
class ByteTrackNoReIDPool{
public:
    ByteTrackNoReIDPool(){}
    ~ByteTrackNoReIDPool(){}
    ByteTrackNoReIDPool(int fps, int nn_buf):\
        m_fps(fps),\
        m_nn_buf(nn_buf) {}

    void set_fps(int fps){m_fps=fps;}
    // void set_threshold(std::vector<float> &thresholds){
    //     std::lock_guard<std::mutex> lk(m_mutex);
    //     m_track_threshold = thresholds[0];
    //     m_high_detect_threshold = thresholds[1];
    //     for(auto &&_trk: m_trackors){
    //         _trk.second->reset(m_track_threshold, m_high_detect_threshold, m_fps, m_nn_buf);
    //     }
    // }
    void update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM params);
    void add_trackor(int cam_uuid, BYTETRACKPARM params);

private:
    std::map<int, ByteTrackNoReID_Ptr> m_trackors;
    std::map<int, BYTETRACKPARM> m_params;
    int m_fps = 25;
    int m_nn_buf = 30;
    std::mutex m_mutex;
};



#endif