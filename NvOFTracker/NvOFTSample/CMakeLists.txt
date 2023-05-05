# Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cmake_minimum_required(VERSION 3.14.0)
project(NvOFTSample LANGUAGES CXX)
set(CMAKE_CXX_STANDARD  11)

if (NOT DEFINED VIDEOCODEC_SDK_ROOT)
message(FATAL_ERROR "Could not find Nvidia Video Codec SDK. Please install or provide the variable VIDEOCODEC_SDK_ROOT.")
endif ()

set(NvOFTSample_SRCS
${CMAKE_CURRENT_SOURCE_DIR}/src/NvOFTSample.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CFramesProducer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/detector/src/CDetector.cpp
${CMAKE_CURRENT_SOURCE_DIR}/detector/src/CTRTManager.cpp
${CMAKE_CURRENT_SOURCE_DIR}/detector/src/logger.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tracker/src/COFTracker.cpp
${VIDEOCODEC_SDK_ROOT}/Samples/NvCodec/NvDecoder/NvDecoder.cpp
)

set(NvOFTSample_CUDA_SRCS
${CMAKE_CURRENT_SOURCE_DIR}/src/ColorSpace.cu
)
set(NvOFTSample_HDRS
${CMAKE_CURRENT_SOURCE_DIR}/include/NvOFTSampleUtils.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CNvOFTSampleException.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CFramesProducer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/ColorSpace.h
${CMAKE_CURRENT_SOURCE_DIR}/detector/include/IDetector.h
${CMAKE_CURRENT_SOURCE_DIR}/detector/include/CDetector.h
${CMAKE_CURRENT_SOURCE_DIR}/detector/include/CTRTManager.h
${CMAKE_CURRENT_SOURCE_DIR}/detector/include/logger.h
${CMAKE_CURRENT_SOURCE_DIR}/detector/include/logging.h
${CMAKE_CURRENT_SOURCE_DIR}/tracker/include/COFTracker.h
${VIDEOCODEC_SDK_ROOT}/Samples/NvCodec/NvDecoder/NvDecoder.h
)

# NvOFTracker
find_path(NVOFT_INCLUDE_DIR NvOFTracker.h HINTS ${CMAKE_CURRENT_SOURCE_DIR} PATH_SUFFIXES ../interface)
MESSAGE(STATUS "Found NvOFTracker include at ${NVOFT_INCLUDE_DIR}")

# OpenCV
if (NOT DEFINED OpenCV_LIBS OR NOT DEFINED OpenCV_INCLUDE_DIRS)
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV config: ${OpenCV_DIR}")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# CUDA & cudnn
find_package(CUDA)
message(STATUS "Cuda toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
find_path(CUDNN_INCLUDE_DIR cudnn.h  HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
find_library(CUDNN_LIBS NAMES cudnn HINTS  ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64)
MESSAGE(STATUS "Found Cuda libs at ${CUDA_CUDA_LIBRARY}")
MESSAGE(STATUS "Found CudaRT libs at ${CUDA_CUDART_LIBRARY}")
MESSAGE(STATUS "Found Cuda headers at ${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "Found Cudnn libs at ${CUDNN_LIBS}")
MESSAGE(STATUS "Found Cudnn headers at ${CUDNN_INCLUDE_DIR}")
# NPP
find_library(NPP_IG_LIB NAMES nppig HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64)
find_library(NPP_ICC_LIB NAMES nppicc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64)
find_library(NPP_S_LIB NAMES npps HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64)
find_library(NPP_C_LIB NAMES nppc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64)
set(NPP_LIBS
${NPP_IG_LIB}
${NPP_ICC_LIB}
${NPP_S_LIB}
${NPP_C_LIB}
)
MESSAGE(STATUS "Found NPP libs at ${NPP_LIBS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=\"sm_75,compute_75\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

# nvcuvid
set(NVCUVID_INCLUDE_DIR ${VIDEOCODEC_SDK_ROOT}/include)
find_library(NVCUVID_LIBS NAMES nvcuvid HINTS  ${VIDEOCODEC_SDK_ROOT} PATH_SUFFIXES Lib/x64 Lib/linux/stubs/x86_64)
MESSAGE(STATUS "Found nvcuvid libs at ${NVCUVID_LIBS}")
#ffmpeg
set(FFMPEG_HDRS_PATH ${VIDEOCODEC_SDK_ROOT}/Samples/External/FFmpeg/include)
find_library(FFMPEG_LIB_AVCODEC avcodec HINTS ${VIDEOCODEC_SDK_ROOT} PATH_SUFFIXES Samples/External/FFmpeg/lib/x64)
find_library(FFMPEG_LIB_AVFORMAT avformat HINTS ${VIDEOCODEC_SDK_ROOT} PATH_SUFFIXES Samples/External/FFmpeg/lib/x64)
find_library(FFMPEG_LIB_AVUTIL avutil HINTS ${VIDEOCODEC_SDK_ROOT} PATH_SUFFIXES Samples/External/FFmpeg/lib/x64)
set(FFMPEG_LIBS
${FFMPEG_LIB_AVCODEC}
${FFMPEG_LIB_AVFORMAT}
${FFMPEG_LIB_AVUTIL}
)
MESSAGE(STATUS "Found FFMPEG libs at ${FFMPEG_LIBS}")

# TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES include cuda/include)
find_library(NVINFER nvinfer HINTS ${TRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
set(TRT_LIBS ${NVINFER})
MESSAGE(STATUS "Found TRT_INCLUDE_DIR at ${TRT_INCLUDE_DIR}")
MESSAGE(STATUS "Found TRT_LIBS at ${TRT_LIBS}")

cuda_add_executable(NvOFTSample ${NvOFTSample_SRCS} ${NvOFTSample_CUDA_SRCS} ${NvOFTSample_HDRS})

target_include_directories(NvOFTSample PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/detector/include
${CMAKE_CURRENT_SOURCE_DIR}/tracker/include
${VIDEOCODEC_SDK_ROOT}/Samples/NvCodec
${VIDEOCODEC_SDK_ROOT}/Samples
${NVCUVID_INCLUDE_DIR}
${NVOFT_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${CUDA_INCLUDE_DIR}
${CUDNN_INCLUDE_DIR}
${TRT_INCLUDE_DIR}
${FFMPEG_HDRS_PATH}
)
target_link_libraries(NvOFTSample
nvoftracker
${OpenCV_LIBS}
${CUDA_CUDART_LIBRARY}
${CUDA_CUDA_LIBRARY}
${NVCUVID_LIBS}
${CUDNN_LIBS}
${TRT_LIBS}
${FFMPEG_LIBS}
${NPP_LIBS}
)

set_target_properties(NvOFTSample PROPERTIES LINKER_LANGUAGE CXX)
