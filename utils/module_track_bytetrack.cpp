#include "module_track.hpp"
#include <stdexcept>

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

cam_class_uuid get_cam_class_uuid(int cam_uuid, ucloud::CLS_TYPE clsType){
    string _cam_uuid = to_string(cam_uuid);
    string _cls_uuid = to_string((int)clsType);
    return _cam_uuid + "_" + _cls_uuid;
}

/**
 * ByteTrackNoReIDPool
 */
void ByteTrackNoReIDPool::clear(){
    for(auto && trackor:m_trackors){
        trackor.second->clear();
    }
}

void ByteTrackNoReIDPool::add_trackor(cam_class_uuid uuid, BYTETRACKPARM params){
    std::lock_guard<std::mutex> lk(m_mutex);
    if(m_trackors.find(uuid)==m_trackors.end()){
        //没有找到，新建一个
        ByteTrackNoReID_Ptr m_trackor = std::make_shared<bytetrack_no_reid::BYTETracker>(params.track_threshold, params.high_detect_threshold, m_fps,m_nn_buf);
        m_trackors[uuid] = m_trackor;
        m_params[uuid] = params;
    } else{
        if(m_params[uuid].track_threshold == params.track_threshold \
            && m_params[uuid].high_detect_threshold == params.high_detect_threshold){
            //threshold一致 nothing
            return;
        } else {
            m_params[uuid] = params;
            m_trackors[uuid]->reset(params.track_threshold, params.high_detect_threshold, m_fps, m_nn_buf);
        }
    }
}

void ByteTrackNoReIDPool::update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM &params){
    int cam_uuid = tvimage.uuid_cam;
    std::map<ucloud::CLS_TYPE, VecObjBBox> clss_bboxes;
    for(auto &&box: bboxIN){
        clss_bboxes[box.objtype].push_back(box);
    }
    VecObjBBox tmp;//merge
    vector<ucloud::CLS_TYPE> triggered;
    for(auto &&clsboxes: clss_bboxes){//loop class
        cam_class_uuid uuid = get_cam_class_uuid(cam_uuid,  clsboxes.first);
        add_trackor(uuid, params);
        triggered.push_back(clsboxes.first);
        vector<bytetrack_no_reid::Object> vecObjIN;
        for(auto &&box: clsboxes.second){//loop boxes for each class
            bytetrack_no_reid::Object _box;
            _box.rect.x = box.rect.x;
            _box.rect.y = box.rect.y;
            _box.rect.width = box.rect.width;
            _box.rect.height = box.rect.height;
            _box.prob = box.confidence;
            _box.label = box.objtype;
            vecObjIN.push_back(_box);
            // std::lock_guard<std::mutex> lk(m_mutex);
            vector<bytetrack_no_reid::STrack> vecTrackOUT = m_trackors[uuid]->update(vecObjIN);
            for(auto &&trackOUT: vecTrackOUT){
                clsboxes.second[trackOUT.detect_idx].track_id = trackOUT.track_id;
            }
        }
        tmp.insert(tmp.end(), clsboxes.second.begin(), clsboxes.second.end() );
    }
    //for untrigger class
    for(auto &&clsboxes: clss_bboxes){
        bool skip = false;
        for(auto &&k: triggered){
            if(k == clsboxes.first){
                skip = true;
                break;
            }
        }
        if(!skip){//即使没有检测到该类别, 只要该类别的trackor存在, 就不能跳帧
            cam_class_uuid uuid = get_cam_class_uuid(cam_uuid,  clsboxes.first);
            vector<bytetrack_no_reid::Object> vecObjIN;
            m_trackors[uuid]->update(vecObjIN);
        }
    }
    //merge
    bboxIN.swap(tmp);

}

//////////////////////////////////////////////
/**
 * ByteTrackOriginPool
 */
void ByteTrackOriginPool::add_trackor(cam_class_uuid uuid, BYTETRACKPARM params){
    std::lock_guard<std::mutex> lk(m_mutex);
    if(m_trackors.find(uuid)==m_trackors.end()){
        //没有找到，新建一个
        ByteTrackOrigin_Ptr m_trackor = std::make_shared<bytetrack_origin::BYTETracker>(params.track_threshold, params.high_detect_threshold, m_fps,m_nn_buf);
        m_trackors[uuid] = m_trackor;
        m_params[uuid] = params;
    } else{
        if(m_params[uuid].track_threshold == params.track_threshold \
            && m_params[uuid].high_detect_threshold == params.high_detect_threshold){
            //threshold一致 nothing
            return;
        } else {
            m_params[uuid] = params;
            m_trackors[uuid]->reset(params.track_threshold, params.high_detect_threshold, m_fps, m_nn_buf);
        }
    }
}

void ByteTrackOriginPool::update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM &params){
    int cam_uuid = tvimage.uuid_cam;
    std::map<ucloud::CLS_TYPE, VecObjBBox> clss_bboxes;
    for(auto &&box: bboxIN){
        clss_bboxes[box.objtype].push_back(box);
    }
    VecObjBBox tmp;//merge
    vector<ucloud::CLS_TYPE> triggered;
    for(auto &&clsboxes: clss_bboxes){//loop class
        cam_class_uuid uuid = get_cam_class_uuid(cam_uuid,  clsboxes.first);
        LOGI << "cam uuid: " << uuid;
        add_trackor(uuid, params);
        triggered.push_back(clsboxes.first);
        vector<bytetrack_origin::Object> vecObjIN;
        for(auto &&box: clsboxes.second){//loop boxes for each class
            bytetrack_origin::Object _box;
            _box.rect.x = box.rect.x;
            _box.rect.y = box.rect.y;
            _box.rect.width = box.rect.width;
            _box.rect.height = box.rect.height;
            _box.prob = box.confidence;
            _box.label = box.objtype;
            vecObjIN.push_back(_box);
            // std::lock_guard<std::mutex> lk(m_mutex);
            vector<bytetrack_origin::STrack> vecTrackOUT = m_trackors[uuid]->update(vecObjIN);
            for(auto &&trackOUT: vecTrackOUT){
                clsboxes.second[trackOUT.detect_idx].track_id = trackOUT.track_id;
            }
        }
        tmp.insert(tmp.end(), clsboxes.second.begin(), clsboxes.second.end() );
    }
    //for untrigger class
    for(auto &&clsboxes: clss_bboxes){
        bool skip = false;
        for(auto &&k: triggered){
            if(k == clsboxes.first){
                skip = true;
                break;
            }
        }
        if(!skip){//即使没有检测到该类别, 只要该类别的trackor存在, 就不能跳帧
            cam_class_uuid uuid = get_cam_class_uuid(cam_uuid,  clsboxes.first);
            vector<bytetrack_origin::Object> vecObjIN;
            m_trackors[uuid]->update(vecObjIN);
        }
    }
    //merge
    bboxIN.swap(tmp);
}

void ByteTrackOriginPool::clear(){
    for(auto && trackor:m_trackors){
        trackor.second->clear();
    }
}


