/*
* Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
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
#include "CROITracker.h"
#include "CNvOpticalFlowOCV.h"

#include<memory>
#include <algorithm>

class CNvOFTracker
{
public:
    CNvOFTracker(const NvOFT_CREATE_PARAMS* createParams);
    void TrackObjects(const NvOFT_PROCESS_IN_PARAMS* inParams, NvOFT_PROCESS_OUT_PARAMS* outParams);

private:
    void SanitizeRect(cv::Rect& rect)
    {
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(rect.width, (int)m_Width - rect.x);
        rect.height = std::min(rect.height, (int)m_Height - rect.y);
    }

private:
    std::unique_ptr<CROITracker> m_ROITracker;
    std::unique_ptr<IOpticalFlow> m_OpticalFlow;
    uint32_t m_Width;
    uint32_t m_Height;
    uint32_t m_GPUId;
    NvOFT_SURFACE_MEM_TYPE m_SurfType;
    NvOFT_SURFACE_FORMAT m_SurfFormat;
};
