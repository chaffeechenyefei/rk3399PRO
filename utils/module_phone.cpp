#include "module_phone.hpp"

PhoneDetector::PhoneDetector(){
    LOGI<<"->PhoneDetector";
    m_ped_detectHandle = std::make_shared<YOLO_DETECTION>();
    std::vector<ucloud::CLS_TYPE> model_output_clss = {ucloud::CLS_TYPE::PEDESTRIAN, ucloud::CLS_TYPE::NONCAR, ucloud::CLS_TYPE::CAR, ucloud::CLS_TYPE::CAR, ucloud::CLS_TYPE::CAR, ucloud::CLS_TYPE::NONCAR, ucloud::CLS_TYPE::NONCAR, ucloud::CLS_TYPE::CAR, ucloud::CLS_TYPE::NONCAR};
    m_ped_detectHandle->set_output_cls_order(model_output_clss);
    m_clsHandle = std::make_shared<Classification>();
    // printf("--->>PhoneDetector Cls_handle init m_clss size :%d\n",m_clss.size());
    // m_clsHandle->set_output_cls_order(m_clss,m_select);
}

PhoneDetector::~PhoneDetector(){
}


void PhoneDetector::transform_box_to_ped_box(ucloud::VecObjBBox &in_boxes, ucloud::VecObjBBox &out_boxes,int &im_width, int &im_height){
    for( auto &&in_box: in_boxes){
        ucloud::BBox out_box;
        printf("--->> Transformer Before:transform box topx %f topy %f width %f height %f\n",in_box.x0,in_box.y0,in_box.w,in_box.h);
        float topx = clip<float>(in_box.x0,0,im_width-1);
        float topy = clip<float>(in_box.y0,0,im_height-1);
        float bottomx = clip<float>(in_box.x1,0,im_width-1);
        float bottomy = clip<float>(in_box.y1,0,im_height-1);
        float w = bottomx - topx;
        float h = bottomy - topy;
        float hw_ratio = ((float)(1.0*h))/w;
        // ucloud::TvaiRect body_rect = in_box.rect;
        // printf("ped box rect is x:%f, y:%f,width:%f,height:%f\n",topx,topy,bw,bh);
        // float hw_ratio = ((float)(1.0*body_rect.height))/ body_rect.width;
        // float top_x = clip<float>(body_rect.x,0,im_width-1);
        // float bottom_x  = clip<float>(top_x + body_rect.width,0,im_width-1);
        // float center_y = body_rect.y + body_rect.height*0.5;
        // float top_y = clip<float>(body_rect.y,0,im_height-1);
        if(hw_ratio >= 2){
            h *= 0.5;
        }else if(hw_ratio >= 1.5){
            h *= 0.8;
        }
        float bottom_y = clip<float>(topy+h,0,im_height-1);
        // float nty = center_y - 0.5*body_rect.height;
        // float nby = center_y + 0.5*body_rect.height;
        // float nyt = clip<float>(nty,0,im_height-1);
        // float nyb = clip<float>(nby,0,im_height-1);
        // float xt = clip<float>(body_rect.x,0,im_width);
        // float xb = clip<float>(bx,0,im_width-1);
        out_box.x0 = topx;
        out_box.y0 = topy;
        out_box.x1 = bottomx;
        out_box.y1 = bottom_y;
        // float w = bottom_x-top_x;
        h = bottom_y - topy;
        // out_box.target = in_box;
        out_box.rect  = {x:(int)topx,y:(int)topy,width:(int)w,height:(int)h};
        out_boxes.push_back(out_box);
        printf("--->> Transformer After:transform box topx %f topy %f width %f height %f\n",topx,topy,w,h);
    }
    return;
}

ucloud::RET_CODE PhoneDetector::init(std::map<ucloud::InitParam,std::string> &modelpath){
    LOGI<<"->PhoneDetector init";
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    std::map<ucloud::InitParam,std::string>  dnetPath,cnetPath;
    if (modelpath.find(ucloud::InitParam::BASE_MODEL)==modelpath.end()||
    modelpath.find(ucloud::InitParam::SUB_MODEL)==modelpath.end()){
        LOGI<<"PhoneDetector fail to search detector and classify model";
        ret = ucloud::RET_CODE::ERR_INIT_PARAM_FAILED;
        return ret;
    }
    dnetPath = {{ucloud::InitParam::BASE_MODEL,modelpath[ucloud::InitParam::BASE_MODEL]}};
    cnetPath = {{ucloud::InitParam::SUB_MODEL,modelpath[ucloud::InitParam::SUB_MODEL]}};
    ret = m_ped_detectHandle->init(dnetPath);

    // 将分类器的类别初始化从构造函数移至init函数，防止引用未定义变量
    printf("--->>PhoneDetector Cls_handle init m_clss size :%d\n",m_clss.size());
    m_clsHandle->set_output_cls_order(m_clss,m_select);
    if (ret!=ucloud::RET_CODE::SUCCESS){
        LOGI<<"->PhoneDetector person detector failed";
        return ret;
    }
    
    ret = m_clsHandle->init(cnetPath);
      if (ret!=ucloud::RET_CODE::SUCCESS){
        LOGI<<"->PhoneDetector phone classfication failed";
        return ret;
    } 
    return ret;
}


ucloud::RET_CODE PhoneDetector::run(ucloud::TvaiImage& tvimage,ucloud::VecObjBBox &bboxes,float threshold, float nms_threshold){
    ucloud::VecObjBBox detBboxes,clsBboxes,ped_bboxes;
    ucloud::RET_CODE ret = ucloud::RET_CODE::SUCCESS;
    ret = m_ped_detectHandle->run(tvimage,bboxes,m_ped_threshold);
    // printf("person detect success! \n");
    if(ret!=RET_CODE::SUCCESS){
        LOGI<<"->PhoneDetector detect person failed!\n";
        // printf("err [%d] in ptrHandle->run(tvInp, bboxes) \n", int(ret));
        return ret;
    }
    for(auto &&box: bboxes){
        if(box.objtype == ucloud::CLS_TYPE::PEDESTRIAN){
            ped_bboxes.push_back(box);
            // printf("ped bboxes is topx:%f,topy:%f,bottomx:%f,bottomy:%f\n",box.x0,box.y0,box.x1,box.y1);
        }
    }
    // printf("select person box success! \n");
    if (ped_bboxes.empty()){
        LOGI<<"->PhoneDetector person box is None!\n";
        return ucloud::RET_CODE::ERR_EMPTY_BOX; 
    }
    transform_box_to_ped_box(ped_bboxes,clsBboxes,tvimage.width,tvimage.height);
    // printf("transform box success! \n");
    if (clsBboxes.empty()){
        // printf("clsBboxes is empty!\n");
        return ucloud::RET_CODE::ERR_EMPTY_BOX;
    }
    bboxes.clear();
    bboxes.assign(clsBboxes.begin(),clsBboxes.end());
    // printf("reassign bboxes success!");


    ret = m_clsHandle->run(tvimage,bboxes,threshold);
    if(ret!=RET_CODE::SUCCESS){
        LOGI<<"->PhoneDetector phone classify  images failed!\n";
        // printf("err [%d] in ptrHandle->run(tvInp, bboxes) \n", int(ret));
        return ret;
    } 
    return ret;
}

ucloud::RET_CODE PhoneDetector::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return ucloud::RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&m: m_clss){
        bool FLAG_exsit_class = false;
        for( auto &&n:valid_clss){
            if( m==n){
                FLAG_exsit_class = true;
                break;
            }
        }
        if(!FLAG_exsit_class) valid_clss.push_back(m);
    }
    return ucloud::RET_CODE::SUCCESS;
}

static inline int get_unique_cls_num(std::vector<ucloud::CLS_TYPE>& output_clss, std::map<ucloud::CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::vector<ucloud::CLS_TYPE> unique_cls;
    for(auto i=output_clss.begin(); i !=output_clss.end(); i++){
        bool conflict = false;
        for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
            if( *i == *iter ){
                conflict = true;
                break;
            }
        }
        if(!conflict) unique_cls.push_back(*i);
    }
    for(int i=0; i < unique_cls.size(); i++ ){
        unique_cls_order.insert(std::pair<ucloud::CLS_TYPE,int>(unique_cls[i],i));
    }
    return unique_cls.size();
}

ucloud::RET_CODE PhoneDetector::set_output_cls_order(std::vector<ucloud::CLS_TYPE>& output_clss,int select_idx){
    m_nc = output_clss.size();
    m_clss = output_clss;
    m_select = select_idx;
    // printf("---->>> PhoneDetector set output cls m_clss size :%d ,m_select :%d\n",m_clss.size(),m_select);
    get_unique_cls_num(output_clss, m_unique_clss_map);
    return ucloud::RET_CODE::SUCCESS;
}
