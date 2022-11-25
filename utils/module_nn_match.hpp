#ifndef _MODULE_NN_MATCH_HPP_
#define _MODULE_NN_MATCH_HPP_
#include "../libai_core.hpp"
#include <opencv2/opencv.hpp>

using ucloud::VecObjBBox;
using ucloud::BatchBBoxIN;
using ucloud::BBox;
using ucloud::TvaiRect;
using ucloud::uPoint;
/////////////////////////////////////////////////////////////////////////////////////
// API
/////////////////////////////////////////////////////////////////////////////////////
cv::Mat calc_iou_cost_matrix(VecObjBBox &b1, VecObjBBox &b2);
cv::Mat calc_distance_cost_matrix(VecObjBBox &b1, VecObjBBox &b2);
cv::Mat calc_feature_cost_matrix(VecObjBBox &b1, VecObjBBox &b2);

float calc_iou_cost(BBox &b1, BBox &b2);
float calc_distance_cost(BBox &b1, BBox &b2);
float calc_feature_cost(BBox &b1, BBox &b2);

//angle between ac and bc. [0-180]
float calc_angle(uPoint &a, uPoint &b, uPoint &c);



/////////////////////////////////////////////////////////////////////////////////////
// 新API
/////////////////////////////////////////////////////////////////////////////////////
const int max_t = 1000000;
const float max_angle = std::cos(70.f/180.f*3.1415);
const float max_distance = 200.f;
/**
 * 单个的box
 */
class BoxPoint{
public:
    BoxPoint(){}
    ~BoxPoint(){}
    BoxPoint(BBox &box, int T=0){
        *this = BoxPoint(box.rect, T);
    }
    BoxPoint(TvaiRect &rect, int T=0){
        x = rect.x;
        y = rect.y;
        w = rect.width;
        h = rect.height;
        x0 = x;
        y0 = y;
        x1 = x + w;
        y1 = y + h;
        cx = (x0+x1)/2;
        cy = (y0+y1)/2;
        t = T%max_t;
        FLG_empty = false;
    }
    void updateT(int T){
        t=T%max_t;
    }
    float calc_cost(BoxPoint &boxPt, float w_iou=1, float w_dist=1);
    void print();
    bool empty(){return FLG_empty;}
    bool FLG_empty = true;
    float x0{0}, x1{0}, y0{0}, y1{0};
    float x{0},y{0},w{0},h{0};
    float cx{0}, cy{0};
    int t = 0;
    int m_trace_id = -1;//这个boxPoint属于哪个轨迹
};

float calc_iou_cost(BoxPoint &b1, BoxPoint &b2);
float calc_distance_cost(BoxPoint &b1, BoxPoint &b2);
float calc_space_cost(BoxPoint &b1, BoxPoint &b2, float w_iou=1, float w_dist=1);

/**
 * box的轨迹
 */
class BoxTrace{
public:
    BoxTrace(){}
    ~BoxTrace(){std::vector<BoxPoint>().swap(m_trace);}
    bool empty(){return m_trace.empty();}
    void push_back(BoxPoint &box);
    BoxPoint get_last_trace(){      
        if (m_trace.empty()) return BoxPoint() ;
        else return m_trace[m_trace.size()-1];
    }
    int get_last_time(){
        if(m_trace.empty()) return -1;
        else return m_trace[m_trace.size()-1].t;
    }
    float calc_cost(BoxPoint &boxPt, float w_iou=1, float w_dist=1);
    void set_trace_id(int trace_id);
    void print();
    std::vector<BoxPoint> m_trace;
    float m_velocity = 0;
    float m_vec_x = 0;
    float m_vec_y = 0;
    int m_trace_id = -1;
};
/**
 * 聚合点到轨迹中 
 */
class BoxTraceSet{
public:
    void push_back(BoxPoint &boxPt);
    void push_back(std::vector<BoxPoint> &boxPts);
    void update(int cur_time);
    void update();

    void print();

    void clear(){
        std::vector<BoxPoint>().swap(mboxPts);
        std::vector<BoxTrace>().swap(mboxTrs);
        m_time = 0;
    }

    void output_last_point_of_trace(std::vector<BoxPoint> &vecPts, std::vector<BoxPoint> &vecPts_uncertain,int min_box_num=3);
    void output_trace(std::vector<BoxTrace> &vecTrs, std::vector<BoxPoint> &vecPts_uncertain,int min_box_num=3);
    void output_trace(std::vector<BoxPoint> &vecPts, std::vector<BoxPoint> &vecPts_uncertain,int min_box_num=3);
    void output_trace(std::vector<BoxPoint> &vecPts_S, std::vector<BoxPoint> &vecPts_M, std::vector<BoxPoint> &vecPts_uncertain, int min_box_num=3);
    std::vector<BoxPoint> mboxPts;
    std::vector<BoxTrace> mboxTrs;

    float m_w_iou = 1;
    float m_w_dist = 1;
    
#ifdef MLU220
    int fps = 5;
#else
#ifdef SIM_MLU220
    int fps = 5;
#else
    int fps = 25;//25
#endif
#endif
    // int nn_max_age = 4*fps;
    int nn_max_age = 2;
    int m_time = 0;

private:
    int next_trace_id = 1;
};





#endif