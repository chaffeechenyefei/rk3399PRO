#include "module_posenet.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>

using namespace std;
using namespace ucloud;


static void get_max_coords_of_heatmap(float* heatmap, int C, int H, int W, std::vector<Point2f>& coords);
static void rescale_coords_to_sklandmark(std::vector<Point2f>& coords, TvaiRect offset, float aX, float aY, float zX, float zY ,LandMark &landmk);

void get_max_coords_of_heatmap(float* heatmap, int C, int H, int W, std::vector<Point2f>& coords){
    float dr = 0.25;
    for(int c = 0; c < C; c++){
        float* _hmap = heatmap + c*H*W;
        int max_idx; float max_val;
        argmax(_hmap, H*W, max_idx, max_val);
        int x = max_idx%W;
        int y = max_idx/W;
        float fx = 1.0*x;
        float fy = 1.0*y;
        // if(fx>2000 || fy>2000)
        //     LOGI << "max_idx = " << max_idx;
        if( 1 < x && x < W - 1 && 1 < y && y < H - 1){
            float dx = _hmap[x+1 + y*W] - _hmap[x-1 + y*W];
            float dy = _hmap[x+(y+1)*W] - _hmap[x+(y-1)*W];
            fx += dx*dr;
            fy += dy*dr;
        }
        // LOGI << "fx,fy" << fx << ", " << fy;
        coords.push_back(Point2f(fx,fy));
    }
#ifdef VERBOSE
    cv::Mat im(cv::Size(W,H),CV_8UC1);
    im = 0;
    for(auto &&coord: coords){
        im.at<uint8_t>(int(coord.y), int(coord.x)) = 255;
    }
    cv::imwrite("coord.bmp", im);
#endif
}

void rescale_coords_to_sklandmark(std::vector<Point2f>& coords,TvaiRect offset , float aX, float aY, float zX, float zY, LandMark &landmk){
    if(!landmk.pts.empty()){
        landmk.pts.clear();
    }
    for(int i = 0; i < coords.size(); i++ ){
        landmk.pts.push_back(uPoint(zX*coords[i].x /aX + offset.x, zY*coords[i].y /aY + offset.y));
     }
     landmk.refcoord = RefCoord::IMAGE_ORIGIN;
     landmk.type = LandMarkType::SKELETON;
}


/*******************************************************************************
 * postprocess
 * 1,17,h,w
 * 3,2,1,0
 * chaffee.chen@2022-11-03
*******************************************************************************/
RET_CODE PoseNet::postprocess(std::vector<float*> &output_datas ,ucloud::BBox &bbox, float aX, float aY){
    LOGI << "-> PoseNet::postprocess";
    int _W = m_param_img2tensor.model_input_shape.w;
    int _H = m_param_img2tensor.model_input_shape.h;
    int _oW = m_OutEleDims[0][0];
    int _oH = m_OutEleDims[0][1];
    int _oC = m_OutEleDims[0][2];
    LOGI << "_W,_H,_oW,_oH,_oC = " << _W << ", " << _H << ", "
        << _oW << ", " << _oH << ", " << _oC; 

    ucloud::LandMark kypts;
    std::vector<cv::Point2f> coords;
    float zX = (1.0*_W)/_oW;
    float zY = (1.0*_H)/_oH;
    get_max_coords_of_heatmap(output_datas[0], _oC, _oH, _oW , coords);
    rescale_coords_to_sklandmark(coords, bbox.rect , aX, aY, zX, zY, kypts);

    bbox.Pts = kypts;

    LOGI << "<- PoseNet::postprocess";
    return RET_CODE::SUCCESS;
}
