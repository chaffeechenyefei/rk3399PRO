#pragma once

#include "STrack_origin.h"

namespace bytetrack_origin{
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class BYTETracker
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	BYTETracker(float track_threshold, float high_detect_threshold, int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	void reset(float track_threshold, float high_detect_threshold, int frame_rate = 30, int track_buffer = 30);

	vector<STrack> update(const vector<Object>& objects);
	// Scalar get_color(int idx);
	void clear();

private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh; //用于区分是否是低置信度的检测框，首先进行高置信度的匹配
	float high_thresh;	//如果都没有命中，且置信度高于一定程度，那么进行
	float match_thresh;
	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};

};