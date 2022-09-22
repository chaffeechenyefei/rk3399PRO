#include "module_track.hpp"

using namespace std;

static bool isBox_tlwh_close(BBox &box, vector<float> tlwh ){
    float k1 = box.rect.x - tlwh[0];
    float k2 = box.rect.y - tlwh[1];
    float k3 = box.rect.width - tlwh[2];
    float k4 = box.rect.height - tlwh[3];
    float dist = k1*k1 + k2*k2 + k3*k3 + k4*k4;
    if(dist < 1e-3)
        return true;
    else
        return false;
}

void ByteTrackNoReIDPool::add_trackor(int cam_uuid, BYTETRACKPARM params){
    std::lock_guard<std::mutex> lk(m_mutex);
    if(m_trackors.find(cam_uuid)==m_trackors.end()){
        //没有找到，新建一个
        ByteTrackNoReID_Ptr m_trackor = std::make_shared<bytetrack_no_reid::BYTETracker>(params.track_threshold, params.high_detect_threshold, m_fps,m_nn_buf);
        m_trackors[cam_uuid] = m_trackor;
        m_params[cam_uuid] = params;
    } else{
        if(m_params[cam_uuid].track_threshold == params.track_threshold \
            && m_params[cam_uuid].high_detect_threshold == params.high_detect_threshold){
            //threshold一致 nothing
        } else {
            m_params[cam_uuid] = params;
            m_trackors[cam_uuid]->reset(params.track_threshold, params.high_detect_threshold, m_fps, m_nn_buf);
        }
    }
}

void ByteTrackNoReIDPool::update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM params){
    int cam_uuid = tvimage.uuid_cam;
    add_trackor(cam_uuid, params);
    vector<bytetrack_no_reid::Object> vecObjIN;
    for(auto &&box: bboxIN){
        bytetrack_no_reid::Object _box;
        _box.rect.x = box.rect.x;
        _box.rect.y = box.rect.y;
        _box.rect.width = box.rect.width;
        _box.rect.height = box.rect.height;
        _box.prob = box.confidence;
        _box.label = box.objtype;
        vecObjIN.push_back(_box);
    }
    
    std::lock_guard<std::mutex> lk(m_mutex);
    vector<STrack> vecTrackOUT = m_trackors[cam_uuid]->update(vecObjIN);
    for(auto &&trackOUT: vecTrackOUT){
        bboxIN[trackOUT.detect_idx].track_id = trackOUT.track_id;
    }

    // for(auto &&trackOUT: vecTrackOUT){
    //     vector<float> tlwh = trackOUT.tlwh;
    //     for(auto &&box: bboxIN){
    //         if(isBox_tlwh_close(box, tlwh))
    //             box.track_id = trackOUT.track_id;
    //     }
    // }
}

