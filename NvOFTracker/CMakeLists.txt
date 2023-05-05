# Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cmake_minimum_required(VERSION 3.14.0)
project(NvOFTrackerMain)
set(CMAKE_CXX_STANDARD  11)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(NvOFTracker)
add_subdirectory(NvOFTSample)

install (TARGETS nvoftracker LIBRARY DESTINATION bin)
install (TARGETS NvOFTSample RUNTIME DESTINATION bin)
