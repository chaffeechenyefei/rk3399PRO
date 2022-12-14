#include "module_nn_match.hpp"
using std::vector;
using namespace cv;

#define PI 3.1415
#define OMAX 10

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
float calc_iou_cost(BBox &b1, BBox &b2){
    int b1x1 = b1.rect.x;
    int b1y1 = b1.rect.y;
    int b1x2 = b1.rect.width + b1.rect.x;
    int b1y2 = b1.rect.height + b1.rect.y;
    int b2x1 = b2.rect.x;
    int b2y1 = b2.rect.y;
    int b2x2 = b2.rect.width + b2.rect.x;
    int b2y2 = b2.rect.height + b2.rect.y;

    float roiW = std::min(b1x2, b2x2) - std::max(b1x1,b2x1);
    float roiH = std::min(b1y2, b2y2) - std::max(b1y1, b2y1);
    if(roiW<=0 || roiH <=0 ) return 1;
    float area1 = (b1y2 - b1y1 + 1)*(b1x2-b1x1+1);
    float area2 = (b2y2 - b2y1 + 1)*(b2x2-b2x1+1);
    return 1 - roiW*roiH/(area1+area2-roiW*roiH);
}
cv::Mat calc_iou_cost_matrix(VecObjBBox &b1, VecObjBBox &b2){
    int N = b1.size();
    int M = b2.size();
    Mat coef;
    if(M == 0 || N == 0) return coef;
    coef = Mat::zeros(N,M,CV_32FC1);
    for(int n = 0 ; n < N ; n++ ){
        for( int m = 0; m < N; m++ ){
            coef.at<float>(n,m) = calc_iou_cost(b1[n], b2[m]);
        }
    }
    return coef;
}


float calc_distance_cost(BBox &b1, BBox &b2){
    float cx1 = b1.rect.x + b1.rect.width/2;
    float cy1 = b1.rect.y + b1.rect.height/2;
    float cx2 = b2.rect.x + b2.rect.width/2;
    float cy2 = b2.rect.y + b2.rect.height/2;
    float dist1 = std::sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
    float dist2 = std::sqrt((b1.rect.width+b2.rect.width)*(b1.rect.width+b2.rect.width)/4 + (b1.rect.height+b2.rect.height)*(b1.rect.height+b2.rect.height)/4);
    return dist1/dist2;
}
cv::Mat calc_distance_cost_matrix(VecObjBBox &b1, VecObjBBox &b2){
    int N = b1.size();
    int M = b2.size();
    Mat coef;
    if(M == 0 || N == 0) return coef;
    coef = Mat::zeros(N,M,CV_32FC1);
    for(int n = 0 ; n < N ; n++ ){
        for( int m = 0; m < N; m++ ){
            coef.at<float>(n,m) = calc_distance_cost(b1[n], b2[m]);
        }
    }
    return coef;
}

float calc_feature_cost(BBox &b1, BBox &b2){
    float cosine_dist = 0.0;
    float b1norm = 0.0;
    float b2norm = 0.0;
    for(int i = 0; i < b1.trackfeat.size(); i++){
        cosine_dist += b1.trackfeat[i]*b2.trackfeat[i];        
        // b1norm += b1.trackfeat[i]*b1.trackfeat[i];
        // b2norm += b2.trackfeat[i]*b2.trackfeat[i];
    }
    return 1 - cosine_dist;
    // return cosine_dist/(std::sqrt(b1norm+1e-3)*std::sqrt(b2norm+1e-3));
}
cv::Mat calc_feature_cost_matrix(VecObjBBox &b1, VecObjBBox &b2){
    int N = b1.size();
    int M = b2.size();
    Mat coef;
    if(M == 0 || N == 0) return coef;
    coef = Mat::zeros(N,M,CV_32FC1);
    for(int n = 0 ; n < N ; n++ ){
        for( int m = 0; m < N; m++ ){
            coef.at<float>(n,m) = calc_feature_cost(b1[n], b2[m]);
        }
    }
    return coef;
}

//angle between ac and bc. [0-180]
float calc_angle(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;
    /**
     * https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
     * The orientation of this angle matches that of the coordinate system. 
     * In a left-handed coordinate system, i.e. x pointing right and y down as is common for computer graphics, 
     * this will mean you get a positive sign for clockwise angles. 
     * If the orientation of the coordinate system is mathematical with y up, 
     * you get counter-clockwise angles as is the convention in mathematics. 
     * Changing the order of the inputs will change the sign, so if you are unhappy with the signs just swap the inputs.
     * 
     * https://en.cppreference.com/w/cpp/numeric/math/atan2
     * If no errors occur, the arc tangent of y/x (arctan(y/x)) in the range [-?? , +??] radians, is returned.
     */
    float dot = ac.x*bc.x + ac.y*bc.y;
    float det = ac.x*bc.y - ac.y*bc.x;
    float angle = std::atan2(det,dot)*180/PI;
    return std::abs(angle);
}
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

float BoxPoint::calc_cost(BoxPoint &boxPt, float w_iou, float w_dist){    
    float vec_x = boxPt.cx - this->cx;
    float vec_y = boxPt.cy - this->cy;
    float cos_norm = std::sqrt(vec_x*vec_x+vec_y*vec_y);
    float cos_val = vec_y / cos_norm;
    //?????????????????????????????????????????????, ???????????????
    if(cos_val < max_angle) return OMAX; //????????????
    if(cos_norm > max_distance) return OMAX; //????????????
    return calc_space_cost(boxPt, *this, w_iou, w_dist);
}

void BoxPoint::print(){
    // std::cout << "["  << cx << "," 
    // << cy << ", " << t
    // << "], ";
    printf("[%.1f, %.1f] ", cx, cy);
}

//???????????????????????????iou?????????, ?????????1-iou?????????, ???????????????????????????.
float calc_iou_cost(BoxPoint &b1, BoxPoint &b2){
    float roiW = std::min(b1.x1, b2.x1) - std::max(b1.x0,b2.x0);
    float roiH = std::min(b1.y1, b2.y1) - std::max(b1.y0, b2.y0);
    if(roiW<=0 || roiH <=0 ) return 1;
    float area1 = (b1.y1 - b1.y0 + 1)*(b1.x1-b1.x0+1);
    float area2 = (b2.y1 - b2.y0 + 1)*(b2.x1-b2.x0+1);
    return 1 - roiW*roiH/(area1+area2-roiW*roiH);
}
//????????????boxpoint???????????????, ????????????????????????. ???????????????
float calc_distance_cost(BoxPoint &b1, BoxPoint &b2){
    float dist1 = std::sqrt((b1.cx-b2.cx)*(b1.cx-b2.cx) + (b1.cy-b2.cy)*(b1.cy-b2.cy));
    float dist2 = std::sqrt((b1.w+b2.w)*(b1.w+b2.w)/4 + (b1.h+b2.h)*(b1.h+b2.h)/4);
    return dist1/dist2;    
}
float calc_space_cost(BoxPoint &b1, BoxPoint &b2, float w_iou, float w_dist){
    float d1 = w_iou*calc_iou_cost(b1,b2);
    float d2 = w_dist*calc_distance_cost(b1,b2);
    return (d1+d2)/(w_iou+w_dist);
}


float BoxTrace::calc_cost(BoxPoint &boxPt, float w_iou, float w_dist){
    int last_index = m_trace.size()-1;
    BoxPoint boxPtTr = m_trace[last_index];//last boxPt from Trace
    
    float vec_x = boxPt.cx - boxPtTr.cx;
    float vec_y = boxPt.cy - boxPtTr.cy;
    float cos_val = vec_x*m_vec_x +vec_y*m_vec_y;
    float cos_norm1 = std::sqrt(vec_x*vec_x+vec_y*vec_y);
    float cos_norm2 = std::sqrt(m_vec_x*m_vec_x+m_vec_y*m_vec_y);
    cos_val /= (cos_norm1*cos_norm2);
    if(cos_val < max_angle) return OMAX;
    if(cos_norm1 > max_distance) return OMAX;
    int t_eclipse = boxPt.t - boxPtTr.t;
    float velocity = cos_norm1;
    if(t_eclipse!=0) velocity /= t_eclipse;
    float diff_velocity = std::abs((velocity - m_velocity) / m_velocity);
    if(diff_velocity>0.3) return OMAX;
    return calc_space_cost(boxPt, boxPtTr, w_iou, w_dist);
}

void BoxTrace::push_back(BoxPoint &box){
    box.m_trace_id = m_trace_id;//??????????????????trace_id
    m_trace.push_back(box);
    //update speed
    if(m_trace.size()>1){
        m_vec_x = m_trace[m_trace.size()-1].cx - m_trace[m_trace.size()-2].cx;
        m_vec_y = m_trace[m_trace.size()-1].cy - m_trace[m_trace.size()-2].cy;
        m_velocity = std::sqrt(m_vec_x*m_vec_x+m_vec_y*m_vec_y);
        int t_eclipse = m_trace[m_trace.size()-1].t - m_trace[m_trace.size()-2].t;
        if(t_eclipse != 0) m_velocity /= t_eclipse;
    } else{
        m_vec_y = 0;
        m_vec_x = 0;
        m_velocity = 0;
    }
}

void BoxTrace::set_trace_id(int trace_id){
    m_trace_id = trace_id;
    for(auto &&bxpt: m_trace){
        bxpt.m_trace_id = m_trace_id;
    }
}

void BoxTrace::print(){
    // std::cout << "#" << m_trace.size() << " : ";
    printf("[trace_id:%d]#%d: ", m_trace.size(), m_trace_id);
    for(auto &&tr: m_trace){
        tr.print();
    }
    printf("\n");
    // std::cout << std::endl;
}


void BoxTraceSet::push_back(std::vector<BoxPoint> &boxPts){
    // std::cout << "-> BoxTraceSet::push_back(std::vector<BoxPoint> &boxPts)" << std::endl;
    if(boxPts.empty()){
        update();
    } else{
        // for(auto boxPt: boxPts){boxPt.print();}
        if( boxPts[0].t < m_time ){
            /*std::cout << "boxtime = " << boxPts[0].t << ", tracetime = " << m_time << std::endl;*/ 
            clear(); } //????????????, ????????????????????????
        for(auto &&boxPt: boxPts){
            // boxPt.print();
            this->push_back(boxPt);
        }
        update(boxPts[0].t);
    }
    // std::cout << "<- BoxTraceSet::push_back(std::vector<BoxPoint> &boxPts)" << std::endl;
}

void BoxTraceSet::push_back(BoxPoint &boxPt){
    // std::cout << "-> BoxTraceSet::push_back(BoxPoint &boxPt)" << std::endl;
    int ind = -1;
    //1 ???????????????????????????
    std::vector<float> traceScores;
    for(auto &&mboxTr: mboxTrs){
        if(mboxTr.get_last_time() >= boxPt.t) continue;
        traceScores.push_back(mboxTr.calc_cost(boxPt, m_w_iou, m_w_dist));
    }
    //TODO
    float min_score = OMAX;
    for(int i = 0; i < traceScores.size(); i++){
        if(traceScores[i] < min_score ) ind = i;
    }
    if(ind >= 0){
        // boxPt.m_trace_id = mboxTrs[ind].m_trace_id;
        mboxTrs[ind].push_back(boxPt);
        m_time = boxPt.t;
        return;
    }

    //2 ?????????????????????????????????
    ind = -1;
    min_score = OMAX;
    std::vector<float> pointScores;
    for(auto &&mboxPt: mboxPts){
        if(mboxPt.t >= boxPt.t) continue;
        pointScores.push_back( mboxPt.calc_cost(boxPt, m_w_iou, m_w_dist));
    }
    for(int i = 0; i < pointScores.size(); i++){
        if(pointScores[i] < min_score) ind = i;
    }
    if(ind >=0) {//????????????, ????????????trace
        BoxTrace nboxTr;
        /*-----------------------------------------------------*/
        //??????trace?????????id, ????????????
        /*-----------------------------------------------------*/
        // nboxTr.m_trace_id = next_trace_id++;
        nboxTr.set_trace_id(next_trace_id++);
        if(next_trace_id >= 1000) next_trace_id = 1;
        // mboxPts[ind].m_trace_id = nboxTr.m_trace_id;
        // boxPt.m_trace_id = nboxTr.m_trace_id;
        /*-----------------------------------------------------*/
        nboxTr.push_back(mboxPts[ind]);
        nboxTr.push_back(boxPt);
        mboxTrs.push_back(nboxTr);
        //remove point from mboxPts
        auto r_iter = mboxPts.begin();
        std::advance(r_iter, ind); 
        mboxPts.erase(r_iter);
        m_time = boxPt.t;
        return;
    } else {
        mboxPts.push_back(boxPt);
        m_time = boxPt.t;
        return;
    }
}

//Warning
void BoxTraceSet::update(int cur_time){
    // std::cout << "-> BoxTraceSet::update(int cur_time)" << std::endl;
    auto iterPt = mboxPts.begin();
    while(iterPt!=mboxPts.end()){
        if( std::abs(cur_time - iterPt->t) >= nn_max_age ){
            iterPt = mboxPts.erase(iterPt);
        }else{
            iterPt++;
        }
    }
    auto iterTr = mboxTrs.begin();
    while(iterTr!=mboxTrs.end()){
        if(std::abs(cur_time - iterTr->get_last_time()) >= nn_max_age ){
            iterTr = mboxTrs.erase(iterTr);
        }else{
            iterTr++;
        }
    }
    m_time = cur_time;
    // std::cout << "<- BoxTraceSet::update(int cur_time)" << std::endl;
}

void BoxTraceSet::update(){
    // std::cout << "-> BoxTraceSet::update()" << std::endl;
    this->update(m_time+1);
    // std::cout << "<- BoxTraceSet::update()" << std::endl;
}


void BoxTraceSet::output_trace(std::vector<BoxTrace> &vecTrs, std::vector<BoxPoint> &vecPts_uncertain,int min_box_num){
    vecTrs.clear();
    vecPts_uncertain.clear();
    for(auto &&boxTr:mboxTrs){
        if(boxTr.m_trace.size() >= min_box_num && boxTr.get_last_time() <= m_time ){
            //??????????????????
            vecTrs.push_back(boxTr);
        } else if(boxTr.m_trace.size()>0 && boxTr.get_last_time() <= m_time) {
            vecPts_uncertain.push_back(boxTr.m_trace[boxTr.m_trace.size()-1]);
        }
    }
    for(auto &&boxPt:mboxPts){
        if(boxPt.t <= m_time){
            vecPts_uncertain.push_back(boxPt);
        }
    }
}

void BoxTraceSet::output_trace(std::vector<BoxPoint> &vecPts, std::vector<BoxPoint> &vecPts_uncertain,int min_box_num){
    vecPts.clear();
    vecPts_uncertain.clear();
    for(auto &&boxTr:mboxTrs){
        if(boxTr.m_trace.size() >= min_box_num && boxTr.get_last_time() <= m_time ){
            //??????????????????
            vecPts.insert(vecPts.end(), boxTr.m_trace.begin(), boxTr.m_trace.end());
        } else if(boxTr.m_trace.size()>0 && boxTr.get_last_time() <= m_time) {
            vecPts_uncertain.push_back(boxTr.m_trace[boxTr.m_trace.size()-1]);
        }
    }
    for(auto &&boxPt:mboxPts){
        if(boxPt.t <= m_time){
            vecPts_uncertain.push_back(boxPt);
        }
    }
}

void BoxTraceSet::output_last_point_of_trace(std::vector<BoxPoint> &vecPts, std::vector<BoxPoint> &vecPts_uncertain, int min_box_num){
    vecPts.clear();
    vecPts_uncertain.clear();
    for(auto &&boxTr:mboxTrs){
        if(boxTr.m_trace.size() >= min_box_num && boxTr.get_last_time() <= m_time ){
            //?????????????????????????
            // vecPts.insert(vecPts.end(), boxTr.m_trace.begin(), boxTr.m_trace.end());
            vecPts.push_back(boxTr.m_trace[boxTr.m_trace.size()-1]);
        } else if(boxTr.m_trace.size()>0 && boxTr.get_last_time() <= m_time) {
            vecPts_uncertain.push_back(boxTr.m_trace[boxTr.m_trace.size()-1]);
        }
    }
    for(auto &&boxPt:mboxPts){
        if(boxPt.t <= m_time){
            vecPts_uncertain.push_back(boxPt);
        }
    }
}

void BoxTraceSet::print(){
    std::cout << "========================" << std::endl;
    std::cout << "=======TraceSet=======" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Frame No." << m_time << std::endl;
    for(int i = 0; i < mboxTrs.size(); i++ ){
        std::cout << "#Trace No."  << i << " :";
        mboxTrs[i].print();
    }
    for(int i = 0; i < mboxPts.size(); i++){
        std::cout << "#Single Point: ";
        mboxPts[i].print();
        std::cout << std::endl;
    }
    std::cout << "========================" << std::endl;

}