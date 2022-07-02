# Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# 3.9.0 is required for CUDA as a first-class language with Visual Studio generator support
cmake_minimum_required(VERSION 3.9)

project(AppStereoDisparityD3D11)

set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/AppStereoDisparityD3D11.cpp")

set(NVOF_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOF.cpp
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3D11.cpp
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3DCommon.cpp
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/Utils/NvOFUtilsD3D11.cpp
 )

set(NVOF_HDRS
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOF.h
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3D11.h
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFDefines.h
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3DCommon.h
 ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowCommon.h
 ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowD3D11.h
 )

set(NVOF_CSSHADER
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/Utils/shader.hlsl
 )

set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_TYPE Compute)
set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_ENTRYPOINT CSStereoMain)
set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_MODEL 5.0)
set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_OUTPUT_HEADER_FILE "UpsampleCS.h")
set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_VARIABLE_NAME "g_UpsampleCS")

source_group( "include" FILES ${NVOF_HDRS} ${NVOF_UTILS_HDRS})
source_group( "source" FILES ${APP_SOURCES} ${NVOF_SOURCES} ${NVOF_UTILS_SOURCES})
source_group( "shader" FILES ${NVOF_CSSHADER})

set(D3D11_LIBS d3d11.lib dxgi.lib d3dcompiler.lib)

add_executable(${PROJECT_NAME} ${NVOF_CSSHADER} ${APP_SOURCES} ${NVOF_SOURCES} ${NVOF_UTILS_SOURCES} ${NVOF_HDRS} ${NVOF_HDRS} ${NVOF_UTILS_HDRS})

target_include_directories(${PROJECT_NAME} PUBLIC  ${FREEIMAGE_INC_DIR}
 ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase
 ${NVOF_UTILS_INCLUDE_DIR}
 ${NVOF_PUBLIC_INCLUDE_DIR}
 ${CMAKE_CURRENT_BINARY_DIR}
 )

target_link_libraries(${PROJECT_NAME} ${D3D11_LIBS} ${FREEIMAGE_LIB})

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
