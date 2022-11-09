#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "kalmanFilter_no_reid.h"

using namespace cv;
using namespace std;


namespace bytetrack_no_reid{
enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
/////20220331 add by lihui,tell the detect match which track
//// 20220719 changed by lihui,add reid and new logic
	STrack(vector<float> tlwh_, float score, int detect_idx);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, float diou, bool new_id = false);
	void update(STrack &new_track, int frame_id, float diou);
	Eigen::Matrix<float,1,-1> kf_gate(vector<DETECTBOX> tlah,bool only_position);
public:
	bool is_activated;
	int track_id;
	int state;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;
	///20220331 add by lihui,tell the detect match which track
	int detect_idx;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;
	float diou;

private:
	byte_kalman::KalmanFilter kalman_filter;
};
}