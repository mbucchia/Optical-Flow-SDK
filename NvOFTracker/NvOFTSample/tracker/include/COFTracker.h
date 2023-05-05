/*
* Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once
#include "NvOFTracker.h"

#include <vector>

struct TrackedObjects
{
    std::vector<NvOFT_TRACKED_OBJ> objects;
    uint32_t filledSize;
    uint32_t frameNum;
};

class COFTracker
{
public:
    COFTracker(uint32_t width, uint32_t height, NvOFT_SURFACE_MEM_TYPE surfType = NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR, NvOFT_SURFACE_FORMAT surfFormat = NvOFT_SURFACE_FORMAT_Y, uint32_t gpuID = 0);
    COFTracker(const COFTracker& ) = delete;
    COFTracker& operator=(COFTracker& ) = delete;
    COFTracker(COFTracker&& ) = delete;
    COFTracker& operator=(COFTracker&& ) = delete;
    ~COFTracker();

public:
    void InitTracker();
    TrackedObjects TrackObjects(const void* frame, size_t frameSize, size_t framePitch, const std::vector<NvOFT_OBJ_TO_TRACK>& inObjVec,
                                      bool detectionDone = true, bool reset = false);

private:
    NvOFTrackerHandle m_OFT;
    uint32_t m_Width;
    uint32_t m_Height;
    NvOFT_SURFACE_MEM_TYPE m_SurfType;
    NvOFT_SURFACE_FORMAT m_SurfFormat;
    uint32_t m_GPUId;
    uint32_t m_FrameNum = 0;
    const static uint32_t OUTLIST_SIZE = 256;
};