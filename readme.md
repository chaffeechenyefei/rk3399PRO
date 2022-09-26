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

## 5. 编译相关:
* 编译使用gcc: gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu.tgz

source: ufs/edge_box/gcc/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu.tgz

* 编译使用第三方静态库: ufs/edge_box/third_libs_compiled/aarch64_gcc11/opencv_aarch64.tar
解压后, 即ffmpeg gflags glog opencv这四个库, 当时为寒武纪aach64编译的, 可以通用.

* 编译使用的rknn的头文件和动态库:
即rknpu中的firefly定制版本：https://gitlab.com/firefly-linux/external/rknpu/-/tree/firefly

git头文件: https://gitlab.com/firefly-linux/external/rknpu/-/tree/firefly/rknn/rknn_api/librknn_api/include/rknn_api.h

git动态库: https://gitlab.com/firefly-linux/external/rknpu/-/tree/firefly/rknn/rknn_api/librknn_api/lib64/librknn_api.so(与嵌入式上不兼容)

实际动态库: ufs/edge_box/rk3399/lib64_firefly/librknn_api.so (从嵌入式中提取出的, 仅用于firefly rk3399PRO)

* 如何让vscode直接使用指定gcc编译器:

在vscode中通过command+shift+p组合键调出命令框, 找到CMake: Editor User-Local CMake Kits, 在其中添加如下内容, 便可以直接在vscode界面中进行build, 只需要选择"x86_64-aarch64"作为Kits即可.
```commandline
  {
    "name": "x86_64-aarch64",
    "compilers": {
      "C": "/project/rk3399_workspace/gcc_tool/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc",
      "CXX": "/project/rk3399_workspace/gcc_tool/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++"
    }
  }
```

* 时钟同步:
```find ./ -type f |xargs touch```

## 6. 性能测试
https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=121051861

## 7.已同步算法
* AlgoAPIName::GENERAL_DETECTOR 人车非通用检测
* AlgoAPIName::SAFETY_HAT_DETECTOR 安全帽检测
