# RK3399PRO API

## 1. 项目地址:
https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=119935816

## 2. 推理速度:
以yolov5s-conv为例, 推理Tensor尺寸=736x416(wxh)

rknn推理速度: 60ms

cpu前处理时间: 34ms

cpu后处理时间: 2ms

## 3. CMakeLists开关:
add_definitions(-DVERBOSE=True)用于设定时 LOGI<< 触发输出, 否则不输出

add_definitions(-DTIMING=True) 用于设定是否进行内部耗时显示

如果不想使用, 则需要在CMakeLists.txt中注释掉.

## 4. 接口说明:
见test.cpp中的使用方式