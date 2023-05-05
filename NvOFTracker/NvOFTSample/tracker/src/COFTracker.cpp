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

#include "CNvOFTSampleException.h"
#include "COFTracker.h"

#include <cstring>

COFTracker::COFTracker(uint32_t width, uint32_t height, NvOFT_SURFACE_MEM_TYPE surfType, NvOFT_SURFACE_FORMAT surfFormat,uint32_t gpuID)
{
    m_Width = width;
    m_Height = height;
    m_SurfType = surfType;
    m_SurfFormat = surfFormat;
    m_GPUId = gpuID;
}

COFTracker::~COFTracker()
{
    NvOFTDestroy(m_OFT);
    m_OFT = nullptr;
}

void COFTracker::InitTracker()
{
    NvOFT_STATUS status = NvOFT_SUCCESS;
    NvOFT_CREATE_PARAMS createParams;
    createParams.width = m_Width;
    createParams.height = m_Height;
    createParams.surfType = m_SurfType;
    createParams.surfFormat = m_SurfFormat;
    createParams.gpuID = m_GPUId;

    CK_NVOFTRACKER(NvOFTCreate(&createParams, &m_OFT));
}

TrackedObjects COFTracker::TrackObjects(const void* frame, size_t frameSize, size_t framePitch, 
    const std::vector<NvOFT_OBJ_TO_TRACK>& inObjVec, bool detectionDone, bool reset)
{
    NvOFT_STATUS status = NvOFT_SUCCESS;
    // Create input params
    NvOFT_PROCESS_IN_PARAMS inParams;
    inParams.frameNum = m_FrameNum++;
    inParams.detectionDone = detectionDone;
    inParams.reset = reset;

    inParams.surfParams.width = m_Width;
    inParams.surfParams.height = m_Height;
    inParams.surfParams.pitch = framePitch;
    inParams.surfParams.surfType = m_SurfType;
    inParams.surfParams.surfFormat = m_SurfFormat;
    inParams.surfParams.frameDataPtr = frame;
    inParams.surfParams.frameDataSize = frameSize;

    inParams.list = inObjVec.data();
    inParams.listSize = (uint32_t)inObjVec.size();

    // Create output params
    NvOFT_PROCESS_OUT_PARAMS outParams;
    std::memset(&outParams, 0, sizeof(outParams));
    TrackedObjects trackedObjects;
    trackedObjects.objects.resize(OUTLIST_SIZE);
    trackedObjects.filledSize = 0;
    outParams.list = trackedObjects.objects.data();
    outParams.listSizeAllocated = (uint32_t)trackedObjects.objects.size();

    // Call tracker
    CK_NVOFTRACKER(NvOFTProcess(m_OFT, &inParams, &outParams));
    trackedObjects.filledSize = outParams.listSizeFilled;
    trackedObjects.frameNum = outParams.frameNum;

    return trackedObjects;
}