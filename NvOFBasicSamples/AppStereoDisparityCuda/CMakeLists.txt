# Copyright (c) 2018-2023 NVIDIA Corporation
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the software, and to permit persons to whom the
# software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.9)
project(AppStereoDisparityCuda)

set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/AppStereoDisparityCuda.cpp")

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

