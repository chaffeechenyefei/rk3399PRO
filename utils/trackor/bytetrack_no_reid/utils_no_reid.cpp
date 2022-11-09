#include "BYTETracker_no_reid.h"
#include "lapjv_no_reid.h"

#define CENTER(x,y) x/2+y/2

namespace bytetrack_no_reid{

vector<STrack*> BYTETracker::joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack*> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i]->track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(&tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i].track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, STrack> stracks;
	for (int i = 0; i < tlista.size(); i++)
	{
		stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (stracks.count(tid) != 0)
		{
			stracks.erase(tid);
		}
	}

	vector<STrack> res;
	std::map<int, STrack>::iterator  it;
	for (it = stracks.begin(); it != stracks.end(); ++it)
	{
		res.push_back(it->second);
	}

	return res;
}

void BYTETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb)
{
	vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
	vector<pair<int, int> > pairs;
	// select maybe duplicate tracks
	for (int i = 0; i < pdist.size(); i++)
	{
		for (int j = 0; j < pdist[i].size(); j++)
		{
			if (pdist[i][j] < 0.15)
			{
				pairs.push_back(pair<int, int>(i, j));
			}
		}
	}
	//decide tracked or lost track which  to leave
	vector<int> dupa, dupb;
	for (int i = 0; i < pairs.size(); i++)
	{
		int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
		int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
		if (timep > timeq)
			dupb.push_back(pairs[i].second);//push into lost track
		else
			dupa.push_back(pairs[i].first);//push into tracked track
	}

	for (int i = 0; i < stracksa.size(); i++)
	{
		vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
		if (iter == dupa.end())
		{
			resa.push_back(stracksa[i]);
		}
	}

	for (int i = 0; i < stracksb.size(); i++)
	{
		vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
		if (iter == dupb.end())
		{
			resb.push_back(stracksb[i]);
		}
	}
}

void BYTETracker::linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
	vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b)
{
	if (cost_matrix.size() == 0)
	{
		for (int i = 0; i < cost_matrix_size; i++)
		{
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++)
		{
			unmatched_b.push_back(i);
		}
		return;
	}

	vector<int> rowsol; vector<int> colsol;
	float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
	for (int i = 0; i < rowsol.size(); i++)
	{
		if (rowsol[i] >= 0)
		{
			vector<int> match;
			match.push_back(i);
			match.push_back(rowsol[i]);
			matches.push_back(match);
		}
		else
		{
			unmatched_a.push_back(i);
		}
	}

	for (int i = 0; i < colsol.size(); i++)
	{
		if (colsol[i] < 0)
		{
			unmatched_b.push_back(i);
		}
	}
}


vector<vector<float> > BYTETracker::ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs)
{
	vector<vector<float> > ious;
	if (atlbrs.size()*btlbrs.size() == 0)
		return ious;

	ious.resize(atlbrs.size());
	for (int i = 0; i < ious.size(); i++)
	{
		ious[i].resize(btlbrs.size());
	}

	//bbox_ious
	for (int k = 0; k < btlbrs.size(); k++)
	{
		vector<float> ious_tmp;
		float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
		for (int n = 0; n < atlbrs.size(); n++)
		{
			float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
			if (iw > 0)
			{
				float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
				if(ih > 0)
				{
					float theta = diou_theta(atlbrs[n],btlbrs[k], false);
					float ua = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
					ious[n][k] = iw * ih / ua - theta;
				}
				else
				{
					ious[n][k] = 0.0;
				}
			}
			else
			{
				ious[n][k] = 0.0;
			}
		}
	}

	return ious;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size)
{
	vector<vector<float> > cost_matrix;
	if (atracks.size() * btracks.size() == 0)
	{
		dist_size = atracks.size();
		dist_size_size = btracks.size();
		return cost_matrix;
	}
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i]->tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	dist_size = atracks.size();
	dist_size_size = btracks.size();

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	
	for (int i = 0; i < _ious.size();i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
			cout<< "_iou is "<<1-_ious[i][j]<<endl;
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i].tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	vector<vector<float> > cost_matrix;
	for (int i = 0; i < _ious.size(); i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}
///////// add by lihui 20220719///////////

float BYTETracker::diou_theta(vector<float> &atlbr,vector<float> &btlbr,bool diou){
	float a_centerX = CENTER(atlbr[0],atlbr[2]);
	float a_centerY = CENTER(atlbr[1],atlbr[3]);
	float b_centerX = CENTER(btlbr[0],btlbr[2]);
	float b_centerY = CENTER(btlbr[1],btlbr[3]);
	float c_cw = b_centerX - a_centerX;
	float c_ch = b_centerY - a_centerY;
	float tl_x = min(atlbr[0],btlbr[0]);
	float tl_y = min(atlbr[1],btlbr[1]);
	float br_x = max(atlbr[2],btlbr[2]);
	float br_y = max(atlbr[3],btlbr[3]);
	float closure_w = br_x - tl_x;
	float closure_h = br_y - tl_y;
	float c2 = pow(closure_w,2) + pow(closure_h,2);
	float c_c2 = pow(c_cw,2) + pow(c_ch,2);
	if (diou){
		float iw = min(atlbr[2],btlbr[2]) -  max(atlbr[0],btlbr[0]);
		float ih = min(atlbr[3],btlbr[3]) - max(atlbr[1],btlbr[1]);
		float a_area = (atlbr[3]-atlbr[1])*(atlbr[2]- atlbr[0]);
		float b_area = (btlbr[3] - btlbr[1])*(btlbr[2] - btlbr[0]);
		if (iw*ih>0){
			return iw*ih/(a_area+b_area-iw*ih)-c_c2/(c2+ 1e-7);
		}else
		{
			return -c_c2/(c2+1e-7);
		}
	}
	return c_c2/(c2+ 1e-7);

}

static inline float InnerProduct(const vector<float> &lfea, const vector<float> &rfea){
	size_t cnt = lfea.size();
	if (cnt!=rfea.size()){
		cout<<"InnerProduct dims don't match!"<<endl;
	}
	float sum{0.f};
	for (size_t i=0;i<cnt;i++){
		sum += lfea[i]*rfea[i];
	}
	return sum;
}

static inline float L2Norm(const vector<float>& fea){
	return sqrt(InnerProduct(fea,fea));
}


vector<vector<float> > BYTETracker::cosine_similarity(vector<vector<float> > &afea, vector<vector<float> > &bfea){
	vector<vector<float> > cos_mat;
	if (afea.size()*bfea.size()==0){
		return cos_mat;
	}
	cos_mat.resize(afea.size());
	for (int i=0;i<afea.size();i++){
		cos_mat[i].resize(bfea.size());
	}
	for (int i=0;i<afea.size();i++){
		for (int j=0;j<bfea.size();j++){
			float xy = InnerProduct(afea[i],bfea[j]);
			float xx = L2Norm(afea[i]);
			float yy = L2Norm(bfea[j]);
			if (xx==0.f||yy==0.f){
				cos_mat[i][j]=0.f;
			}else{
				float c_tmp = xy/(xx*yy);

				cos_mat[i][j] = (0<=c_tmp)&&(c_tmp<=1)?0:1-c_tmp;
			}
		}
	}
	return cos_mat;

}


// vector<vector<float> > BYTETracker::embed_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size){
// 	vector<vector<float> > mat_dst;
// 	if (atracks.size()*btracks.size()==0){
// 		dist_size = atracks.size();
// 		dist_size_size = btracks.size();
// 		return mat_dst;
// 	}
// 	vector<vector<float> > afeas,bfeas;
// 	for (int i=0;i<atracks.size();i++){
// 		afeas.push_back(atracks[i]->_fea);
// 	}
// 	for (int j=0;j<btracks.size();j++){
// 		bfeas.push_back(btracks[j]._fea);
// 	}
// 	mat_dst = cosine_similarity(afeas,bfeas);
// 	return mat_dst;
// }


// vector<vector<float> > BYTETracker::embed_distance(vector<STrack> &atracks, vector<STrack> &btracks){
// 	vector<vector<float> > mat_dst;
// 	if (atracks.size()*btracks.size()==0){
// 		return mat_dst;
// 	}
// 	vector<vector<float> > afeas,bfeas;
// 	for (int i=0;i<atracks.size();i++){
// 		afeas.push_back(atracks[i]._fea);
// 	}
// 	for (int j=0;j<btracks.size();j++){
// 		bfeas.push_back(btracks[j]._fea);
// 	}
// 	mat_dst = cosine_similarity(afeas,bfeas);
// 	return mat_dst;
// }

// vector<vector<float> > BYTETracker::embed_distance(vector<STrack*> &atracks, vector<STrack*> &btracks,int &dist_size, int &dist_size_size){
// 	vector<vector<float> > mat_dst;
// 	if (atracks.size()*btracks.size()==0){
// 		dist_size = atracks.size();
// 		dist_size_size = btracks.size();
// 		return mat_dst;
// 	}
// 	vector<vector<float> > afeas,bfeas;
// 	for (int i=0;i<atracks.size();i++){
// 		afeas.push_back(atracks[i]->_fea);
// 	}
// 	for (int j=0;j<btracks.size();j++){
// 		bfeas.push_back(btracks[j]->_fea);
// 	}
// 	mat_dst = cosine_similarity(afeas,bfeas);
// 	return mat_dst;
// }




double BYTETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
	bool extend_cost, float cost_limit, bool return_cost)
{
	vector<vector<float> > cost_c;
	cost_c.assign(cost.begin(), cost.end());

	vector<vector<float> > cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols)
	{
		n = n_rows;
	}
	else
	{
		if (!extend_cost)
		{
			cout << "set extend_cost=True" << endl;
			system("pause");
			exit(0);
		}
	}
		
	if (extend_cost || cost_limit < LONG_MAX)
	{
		n = n_rows + n_cols;
		cost_c_extended.resize(n);
		for (int i = 0; i < cost_c_extended.size(); i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX)
		{
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_limit / 2.0;
				}
			}
		}
		else
		{
			float cost_max = -1;
			for (int i = 0; i < cost_c.size(); i++)
			{
				for (int j = 0; j < cost_c[i].size(); j++)
				{
					if (cost_c[i][j] > cost_max)
						cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_max + 1;
				}
			}
		}

		for (int i = n_rows; i < cost_c_extended.size(); i++)
		{
			for (int j = n_cols; j < cost_c_extended[i].size(); j++)
			{
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}

	double **cost_ptr;
	cost_ptr = new double *[sizeof(double *) * n];
	for (int i = 0; i < n; i++)
		cost_ptr[i] = new double[sizeof(double) * n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	int* x_c = new int[sizeof(int) * n];
	int *y_c = new int[sizeof(int) * n];

	int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
	if (ret != 0)
	{
		cout << "Calculate Wrong!" << endl;
		system("pause");
		exit(0);
	}

	double opt = 0.0;

	if (n != n_rows)
	{
		for (int i = 0; i < n; i++)
		{
			if (x_c[i] >= n_cols)
				x_c[i] = -1;
			if (y_c[i] >= n_rows)
				y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++)
		{
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++)
		{
			colsol[i] = y_c[i];
		}

		if (return_cost)
		{
			for (int i = 0; i < rowsol.size(); i++)
			{
				if (rowsol[i] != -1)
				{
					//cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost)
	{
		for (int i = 0; i < rowsol.size(); i++)
		{
			opt += cost_ptr[i][rowsol[i]];
		}
	}

	for (int i = 0; i < n; i++)
	{
		delete[]cost_ptr[i];
	}
	delete[]cost_ptr;
	delete[]x_c;
	delete[]y_c;

	return opt;
}

// Scalar BYTETracker::get_color(int idx)
// {
// 	idx += 3;
// 	return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
// }
}