# Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# 3.9.0 is required for CUDA as a first-class language with Visual Studio generator support
cmake_minimum_required(VERSION 3.9)

project(AppOFCuda)

set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/AppOFCuda.cpp")

set(NVOF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOF.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFCuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/Utils/NvOFUtilsCuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/Utils/kernel.cu
 )

 set(NVOF_HDRS
     ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOF.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFCuda.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFDefines.h
 ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowCommon.h
 ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowCuda.h
 )

source_group( "include" FILES ${NVOF_HDRS} ${NVOF_UTILS_HDRS})
source_group( "source" FILES ${APP_SOURCES} ${NVOF_SOURCES} ${NVOF_UTILS_SOURCES})

find_package(CUDA)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=\"sm_70,compute_70\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

#message("cuda lib: ${CUDA_CUDA_LIBRARY}")
#message("cuda include: ${CUDA_INCLUDE_DIRS}")

cuda_add_executable(${PROJECT_NAME}  ${APP_SOURCES} ${NVOF_SOURCES} ${NVOF_UTILS_SOURCES} ${NVOF_HDRS} ${NVOF_HDRS} ${NVOF_UTILS_HDRS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS}
 ${FREEIMAGE_INC_DIR}
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase
 ${NVOF_UTILS_INCLUDE_DIR}
 ${NVOF_PUBLIC_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${FREEIMAGE_LIB})

if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME}  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FREEIMAGE_DLL}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${NVOF_SAMPLES_INSTALL_DIR})
if (MSVC)
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION ${NVOF_SAMPLES_INSTALL_DIR} CONFIGURATIONS Debug)
endif()
