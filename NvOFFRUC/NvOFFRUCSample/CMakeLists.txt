# Copyright 2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cmake_minimum_required(VERSION 3.14.0)
project(NvOFFRUCSample VERSION 0.1.0)
set(CMAKE_CXX_STANDARD  11)

set(NvOFFRUCSample_SRCS
${CMAKE_CURRENT_SOURCE_DIR}/src/NvOFFRUCSample.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Arguments.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/FrameGenerator.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Bitmap.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Util.cpp
)

set(NvOFFRUCSample_HDRS
${CMAKE_CURRENT_SOURCE_DIR}/inc/Bitmap.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/Arguments.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/FrameGenerator.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/CMDParser.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/Common.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/DriverAPIHandler.h
${CMAKE_CURRENT_SOURCE_DIR}/inc/SecureLibraryLoader.h
${CMAKE_CURRENT_SOURCE_DIR}/cudart_header/cuda.h
)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
	set(FREEIMAGE_INC_DIR /usr/include)
	set(FREEIMAGE_LIB /usr/include)
	set(NvOFFRUC_INCLUDE_DIR  ${CMAKE_SOURCE_DIR}/../interface)
	set(INSTALL_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/bin/ubuntu)
else()
	if(CMAKE_SIZEOF_VOID_P EQUAL 8) 
		set(FREEIMAGE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/External/FreeImage/x64/FreeImage.lib)
		set(FREEIMAGE_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/External/FreeImage/x64/FreeImage.dll)
		find_library(NvOFFRUC_LIB NAMES NvOFFRUC HINTS ${CMAKE_CURRENT_SOURCE_DIR} PATH_SUFFIXES ../lib/win64)
		set(INSTALL_FOLDER ${CMAKE_BINARY_DIR}/../bin/win64)
	else()
		set(FREEIMAGE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/External/FreeImage/x32/FreeImage.lib)
		set(FREEIMAGE_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/External/FreeImage/x32/FreeImage.dll)
		find_library(NvOFFRUC_LIB NAMES NvOFFRUC HINTS ${CMAKE_CURRENT_SOURCE_DIR} PATH_SUFFIXES ../lib/win32)
		set(INSTALL_FOLDER ${CMAKE_BINARY_DIR}/../bin/win32)
	endif()
	set(FREEIMAGE_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/External/FreeImage)
	find_path(NvOFFRUC_INCLUDE_DIR NvOFFRUC.h HINTS ${CMAKE_CURRENT_SOURCE_DIR} PATH_SUFFIXES ../interface)
endif()

add_executable(NvOFFRUCSample ${NvOFFRUCSample_SRCS} ${NvOFFRUCSample_HDRS})
target_include_directories(NvOFFRUCSample PRIVATE ${FREEIMAGE_INC_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/inc
)

if(LINUX)
link_directories(${FREEIMAGE_LIB})
link_directories(${CMAKE_SOURCE_DIR}/NvOFFRUC)
target_link_libraries(NvOFFRUCSample PUBLIC freeimage )
target_link_libraries(NvOFFRUCSample PUBLIC ${CMAKE_DL_LIBS})
else()
target_link_libraries(NvOFFRUCSample PUBLIC d3d9.lib dxva2.lib d3d11.lib dxguid.lib ${FREEIMAGE_LIB})
endif()
set_target_properties(NvOFFRUCSample PROPERTIES LINKER_LANGUAGE CXX)


set(NvOFFRUCSample_INSTALL_DIR ${INSTALL_FOLDER})
install(TARGETS NvOFFRUCSample RUNTIME DESTINATION ${NvOFFRUCSample_INSTALL_DIR})

if(WIN32)
install (FILES ${FREEIMAGE_DLL} DESTINATION ${NvOFFRUCSample_INSTALL_DIR})
endif()
