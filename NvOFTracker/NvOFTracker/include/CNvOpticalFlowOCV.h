/*
* Copyright 2019-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once
#include "IOpticalFlow.h"
#include "NvOFTracker.h"
#include "CNvOFTrackerException.h"

#include <cuda_runtime.h>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <memory>
#include <functional>

struct NvOpticalFlowParams
{
    uint32_t width;
    uint32_t height;
    NvOFT_SURFACE_MEM_TYPE surfType;
    NvOFT_SURFACE_FORMAT surfFormat;
    uint32_t gpuID;
};

struct NV_OF_FLOW_VECTOR
{
    int16_t flowx;        /**< x component of flow in S10.5 format */
    int16_t flowy;        /**< y component of flow in S10.5 format */
};

class CNvOpticalFlowOCV : public IOpticalFlow
{
public:
    CNvOpticalFlowOCV(const NvOpticalFlowParams& surfParams);
    ~CNvOpticalFlowOCV() { if (m_OFHandle) m_OFHandle->collectGarbage(); }
    void GetFlow(const void* inputFrame, const size_t inputFrameSize, const size_t inputFramePitch, cv::InputOutputArray flowVectors) override;
    void GetFlowCost(cv::OutputArray flowCost) override { /* NOT Implemented */ return; }

private:
    void* AllocCudaMemory(size_t size)
    {
        void* pCudaMem;
        CK_CUDA(cudaMalloc((void**)&pCudaMem, size));

        return pCudaMem;
    }
    static void ConvertABGRToY(cv::InputArray input, cv::OutputArray output);

private:
    std::function<void(void*)> m_cudaRelease = [](void* pCudaMem) { if (pCudaMem) { cudaFree(pCudaMem); pCudaMem = nullptr; }};
    using CVPtr = cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0>;
    CVPtr m_OFHandle;
    NvOpticalFlowParams m_OFParams;
    cv::cuda::GpuMat m_FrameMat[2];
    std::unique_ptr<void, std::function<void(void*)>> m_inputGPU;
    uint32_t m_CurrIndex = 0;
    uint32_t m_PrevIndex = 1;
};
