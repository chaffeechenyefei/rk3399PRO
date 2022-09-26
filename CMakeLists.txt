cmake_minimum_required(VERSION 2.8)
project(RK3399PRO)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(CMAKE_TOOLCHAIN_FILE) 
  message("@@ cross compile enabled, CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
  set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_edge)
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
  # add_definitions(-DDEBUG=True)
else()
  set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)  
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
  add_definitions(-DDEBUG=True)
endif()

#==========全局宏定义==========
# add_definitions(-DVERBOSE=True) #设定时 LOGI<< 触发输出, 否则不输出
# add_definitions(-DTIMING=True) #是否进行内部耗时显示


# ========RK3399========
set(RK3399_HOME "/project/rk3399_workspace/rknpu_firefly/rknn/rknn_api/librknn_api")
message("-- RK3399_HOME: ${RK3399_HOME}")
set(RK_LIBS "${RK3399_HOME}/lib64/librknn_api.so")
set(RK_INCLUDE_DIRS "${RK3399_HOME}/include")

set(OPENCV_VERSION "opencv")
# set(OPENCV_VERSION "opencv_ffmpeg")

# ========OPENCV========
set(THIRD_PARTY_LIBS "/project/rk3399_workspace/libs")
set(OPENCV_INCLUDE_DIRS ${THIRD_PARTY_LIBS}/${OPENCV_VERSION}/include)
set(OPENCV_LIB_PATH ${THIRD_PARTY_LIBS}/${OPENCV_VERSION}/lib)
set(OPENCV_EXTRA_LIB_PATH ${THIRD_PARTY_LIBS}/${OPENCV_VERSION}/share/OpenCV/3rdparty/lib)
set(OPENCV_LIBS "")
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_video.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_objdetect.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_shape.a)
# list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_stitching.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_superres.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_videostab.a)
# list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_calib3d.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_features2d.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_highgui.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_videoio.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_imgcodecs.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_photo.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_imgproc.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_flann.a)
list(APPEND OPENCV_LIBS ${OPENCV_LIB_PATH}/libopencv_core.a)


list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibprotobuf.a)
list(APPEND OPENCV_LIBS "-Wl,--whole-archive ${OPENCV_EXTRA_LIB_PATH}/libzlib.a -Wl,--no-whole-archive")
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibjpeg-turbo.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibwebp.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibpng.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibtiff.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/liblibjasper.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/libIlmImf.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/libquirc.a)
list(APPEND OPENCV_LIBS ${OPENCV_EXTRA_LIB_PATH}/libtegra_hal.a)
message(STATUS "opencv libraries: ${OPENCV_LIBS}")

# ========ffmpeg========
set(FFMPEG_LIB_PATH "${THIRD_PARTY_LIBS}/ffmpeg_n4.3.4/lib")
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libavformat.a)
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libavdevice.a)
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libavfilter.a)
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libavcodec.a)
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libavutil.a)
# list(APPEND FFMPEG_LIBRARIES "${FFMPEG_LIBRARIES_PATH}/libpostproc.a")
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libswresample.a)
list(APPEND OPENCV_LIBS ${FFMPEG_LIB_PATH}/libswscale.a)

# ========Google-gflags========
set(GFLAGS_INCLUDE_DIRS ${THIRD_PARTY_LIBS}/gflags/include)
set(GFLAGS_LIBRARIES ${THIRD_PARTY_LIBS}/gflags/lib/libgflags.a)
list(APPEND COMMON_INCLUDE_FILES ${GFLAGS_INCLUDE_DIRS})
list(APPEND COMMON_LIB_PATHS ${GFLAGS_LIBRARIES})

# ========Google-glog========
set(GLOG_INCLUDE_DIRS ${THIRD_PARTY_LIBS}/glog/include)
set(GLOG_LIBRARIES ${THIRD_PARTY_LIBS}/glog/lib/libglog.a)
list(APPEND COMMON_INCLUDE_FILES ${GLOG_INCLUDE_DIRS})
list(APPEND COMMON_LIB_PATHS ${GLOG_LIBRARIES})

# ========DRM-RGA Eigen========
set(STB_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty)
list(APPEND DRM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/drm/include ${CMAKE_SOURCE_DIR}/3rdparty/drm/include/libdrm )
set(RGA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/rga/include )
list(APPEND COMMON_INCLUDE_FILES ${STB_INCLUDE_DIRS} ${DRM_INCLUDE_DIRS} ${RGA_INCLUDE_DIRS} )



message("common link libs: ${COMMON_LIB_PATHS}")

# ========WORKSPACE_FILES========
set(UTIL_DIRS "${PROJECT_SOURCE_DIR}/utils")
file(GLOB_RECURSE UTIL_CPP "${UTIL_DIRS}/*.cpp" 
  "${UTIL_DIRS}/drm/*.c" "${UTIL_DIRS}/trackor/bytetrack_no_reid/*.cpp" 
  "${UTIL_DIRS}/trackor/bytetrack_origin/*.cpp" )


# ========CMAKE_CXX_FLAGS========
# -Wfatal-errors 出现错误的时候停止编译
# -Werror 将警告信息当做错误显示
# -Wall 提示警告信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_REENTRANT -Wno-sign-compare") #-Wl,--exclude-libs=ALL
if(CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -ldl -Wno-pointer-arith -Wno-narrowing")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pthread -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -Wfatal-errors -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()
if($ENV{USE_ABI} MATCHES "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
elseif($ENV{USE_ABI} MATCHES "0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
  message("-- Use default ABI settings")
endif()
message("@@ CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

message("@@ DLL MODE ACTIVATED")
add_definitions(-DUCLOUD_EXPORT)
message("visibility activated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")#-fvisibility=hidden
add_library(ai_core SHARED libai_core.cpp ${UTIL_CPP} )
target_include_directories(ai_core PUBLIC ${RK_INCLUDE_DIRS} ${COMMON_INCLUDE_FILES} ${OPENCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdparty/ )
target_link_libraries(ai_core pthread ${COMMON_LIB_PATHS} ${OPENCV_LIBS} ${RK_LIBS} dl)
set_target_properties(ai_core PROPERTIES PUBLIC_HEADER "libai_core.hpp" )


include_directories( ${CMAKE_SOURCE_DIR}/3rdparty/ )
#yolo检测模型单帧测试
add_executable(test_yolo test_yolo.cpp)
target_link_libraries(test_yolo ai_core ${RK_LIBS} dl ${OPENCV_LIBS})

add_executable(test_resnet test_resnet.cpp)
target_include_directories(test_resnet PUBLIC ${RK_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
target_link_libraries(test_resnet ${RK_LIBS} dl ${OPENCV_LIBS})

#任意模型读取后, 输入输出shape打印
add_executable(test_any test_any_model.cpp)
target_include_directories(test_any PUBLIC ${RK_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
target_link_libraries(test_any ${RK_LIBS} dl ${OPENCV_LIBS})

#视频读写推理测试
add_executable(test_video test_video.cpp)
target_link_libraries(test_video ai_core ${RK_LIBS} dl ${OPENCV_LIBS})

#yolo检测模型单帧测试
add_executable(test_loop test_loop.cpp)
target_link_libraries(test_loop ai_core ${RK_LIBS} dl ${OPENCV_LIBS})

#thread性能测试
add_executable(test_thread test_thread.cpp)
target_link_libraries(test_thread ai_core ${RK_LIBS} dl ${OPENCV_LIBS})


message("=========END=========")



