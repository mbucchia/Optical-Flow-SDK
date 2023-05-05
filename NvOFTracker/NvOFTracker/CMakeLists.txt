# Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cmake_minimum_required(VERSION 3.14.0)
project(nvoftracker VERSION 0.1.0)
set(CMAKE_CXX_STANDARD  11)

set(NVOFTRACKER_SRCS
${CMAKE_CURRENT_SOURCE_DIR}/src/NvOFTracker.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CROITracker.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CNvOpticalFlowOCV.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CNvOFTracker.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CConnectedRegionGenerator.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/CHungarianAlgorithm.cpp
)

set(NVOFTRACKER_HDRS
${CMAKE_CURRENT_SOURCE_DIR}/../interface/NvOFTracker.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CROITracker.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CNvOpticalFlowOCV.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CNvOFTracker.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CConnectedRegionGenerator.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CHungarianAlgorithm.h
${CMAKE_CURRENT_SOURCE_DIR}/include/IOpticalFlow.h
${CMAKE_CURRENT_SOURCE_DIR}/include/FlowData.h
${CMAKE_CURRENT_SOURCE_DIR}/include/CNvOFTrackerException.h
)

if (NOT DEFINED OpenCV_LIBS OR NOT DEFINED OpenCV_INCLUDE_DIRS)
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV config: ${OpenCV_DIR}")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

find_package(CUDA)
message(STATUS "Cuda toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Found CudaRT libs at ${CUDA_CUDART_LIBRARY}")
message(STATUS "Found Cuda headers at ${CUDA_INCLUDE_DIRS}")

add_library(nvoftracker SHARED ${NVOFTRACKER_SRCS} ${NVOFTRACKER_HDRS} NvOFTracker.def)
target_include_directories(nvoftracker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../interface ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(nvoftracker ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARY})
set_target_properties(nvoftracker PROPERTIES LINKER_LANGUAGE CXX)
