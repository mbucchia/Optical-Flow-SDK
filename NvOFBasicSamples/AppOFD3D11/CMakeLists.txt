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

# 3.9.0 is required for CUDA as a first-class language with Visual Studio generator support
cmake_minimum_required(VERSION 3.9)

project(AppOFD3D11)

set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/AppOFD3D11.cpp")

set(NVOF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOF.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3D11.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/Utils/NvOFUtilsD3D11.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/NvOFBase/NvOFD3DCommon.cpp
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
set_property(SOURCE ${NVOF_CSSHADER} PROPERTY VS_SHADER_ENTRYPOINT CSOpticalFlowMain)
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
