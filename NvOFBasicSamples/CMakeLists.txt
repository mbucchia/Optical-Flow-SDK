# Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# 3.9.0 is required for CUDA as a first-class language with Visual Studio generator support
cmake_minimum_required(VERSION 3.9)

project(NvOFSamples)

# Set C++11 for all projects and disable non-standard extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NVOF_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../NvOFInterface)

set(NVOF_UTILS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils/NvOFUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils/NvOFDataLoader.cpp
 )

set(NVOF_UTILS_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils/NvOFUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils/NvOFDataLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils/NvOFCmdParser.h
 )

 set(NVOF_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Common/Utils)

 if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(NVOF_SAMPLES_INSTALL_DIR "bin/${CMAKE_INSTALL_CONFIG_NAME}/x64")
else()
    set(NVOF_SAMPLES_INSTALL_DIR "bin/${CMAKE_INSTALL_CONFIG_NAME}/x86")
endif()

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8) 
        Set(FREEIMAGE_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage)
        Set(FREEIMAGE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage/x64/FreeImage.lib)
        set(FREEIMAGE_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage/x64/FreeImage.dll)
    else()
        Set(FREEIMAGE_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage)
        Set(FREEIMAGE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage/x32/FreeImage.lib)
        set(FREEIMAGE_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../Common/External/FreeImage/x32/FreeImage.dll)
    endif()
    install(FILES ${FREEIMAGE_DLL} DESTINATION ${NVOF_SAMPLES_INSTALL_DIR})
else ()
    find_path(FREEIMAGE_INC_DIR NAMES FreeImage.h)
    find_library(FREEIMAGE_LIB freeimage)
endif()

message(STATUS "FreeImage include dir : ${FREEIMAGE_INC_DIR}")
message(STATUS "FreeImage lib dir : ${FREEIMAGE_LIB}")

option(ENABLE_RAW_NVOF_OUTPUT "Enable output dump in raw NVOF format" OFF)
if(ENABLE_RAW_NVOF_OUTPUT)
    add_compile_definitions(ENABLE_RAW_NVOF_OUTPUT)
endif()

if(WIN32)
    add_subdirectory(AppOFD3D12)
    add_subdirectory(AppOFD3D11)
    add_subdirectory(AppStereoDisparityD3D11)
endif()

add_subdirectory(AppOFCuda)
add_subdirectory(AppStereoDisparityCuda)
