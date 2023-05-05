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

#include "CNvOpticalFlowOCV.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

#include <assert.h>
#include <utility>
#include <vector>

CNvOpticalFlowOCV::CNvOpticalFlowOCV(const NvOpticalFlowParams& ofParams) : m_OFParams(ofParams)
{
    m_OFHandle = cv::cuda::NvidiaOpticalFlow_1_0::create(cv::Size(m_OFParams.width, m_OFParams.height), cv::cuda::NvidiaOpticalFlow_1_0::NV_OF_PERF_LEVEL_SLOW, true, false, false, m_OFParams.gpuID);
}

void CNvOpticalFlowOCV::GetFlow(const void* inputFrame, const size_t inputFrameSize, const size_t inputFramePitch, cv::InputOutputArray flowVectors)
{
    if (inputFrame == NULL)
    {
        NVOFTRACKER_THROW_ERROR("Invalid Input Frame", NvOFT_ERR_INVALID_PTR);
    }

    cv::cuda::GpuMat inFrameProcessedGPU;
    if (m_OFParams.surfFormat == NvOFT_SURFACE_FORMAT_ABGR)
    {
        // Convert to Gray scale for ABGR input
        if (m_OFParams.surfType == NvOFT_SURFACE_MEM_TYPE_SYSTEM)
        {
            if (m_inputGPU == nullptr)
            {
                m_inputGPU = { AllocCudaMemory(inputFrameSize), m_cudaRelease };
            }
            CK_CUDA(cudaMemcpy(m_inputGPU.get(), inputFrame, inputFrameSize, cudaMemcpyHostToDevice));
            cv::cuda::GpuMat gpuInFrame(m_OFParams.height, m_OFParams.width, CV_8UC4, m_inputGPU.get(), inputFramePitch);
            ConvertABGRToY(gpuInFrame, inFrameProcessedGPU);
        }
        else // NvOFT_SURFACE_MEM_TYPE_DEFAULT || NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR
        {
            // HACK, FIX IT: Ideally we would not want to const_cast here.
            cv::cuda::GpuMat gpuInFrame(m_OFParams.height, m_OFParams.width, CV_8UC4, const_cast<void*>(inputFrame), inputFramePitch);
            ConvertABGRToY(gpuInFrame, inFrameProcessedGPU);
        }
    }
    else // NvOFT_SURFACE_FORMAT_DEFAULT || NvOFT_SURFACE_FORMAT_Y || NvOFT_SURFACE_FORMAT_NV12
    {
        if (m_OFParams.surfType == NvOFT_SURFACE_MEM_TYPE_SYSTEM)
        {
            if (m_inputGPU == nullptr)
            {
                m_inputGPU = { AllocCudaMemory(inputFrameSize), m_cudaRelease };
            }
            CK_CUDA(cudaMemcpy(m_inputGPU.get(), inputFrame, inputFrameSize, cudaMemcpyHostToDevice));
            cv::cuda::GpuMat tempMat(m_OFParams.height, m_OFParams.width, CV_8UC1, m_inputGPU.get(), inputFramePitch);
            inFrameProcessedGPU = tempMat; // shallow copy
        }
        else // NvOFT_SURFACE_MEM_TYPE_DEFAULT || NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR
        {
            // HACK, FIX IT: Ideally we would not want to const_cast here.
            cv::cuda::GpuMat tempMat(m_OFParams.height, m_OFParams.width, CV_8UC1, const_cast<void*>(inputFrame), inputFramePitch);
            inFrameProcessedGPU = tempMat; // shallow copy
        }
    }

    // At this stage surface is in GPU and surface type is NvOFT_SURFACE_FORMAT_Y
    m_FrameMat[m_CurrIndex] = inFrameProcessedGPU.clone();
    cv::Mat flow;

    if (!m_FrameMat[m_PrevIndex].empty())
    {
        m_OFHandle->calc(m_FrameMat[m_PrevIndex], m_FrameMat[m_CurrIndex], flow);

        // TODO move to cuda based on perf profile
        int flowWidth = flow.cols;
        int flowHeight = flow.rows;

        std::unique_ptr<float[]> flowData = nullptr;
        flowData.reset(new float[2 * flowWidth * flowHeight]);
        const NV_OF_FLOW_VECTOR* _flowVectors = static_cast<const NV_OF_FLOW_VECTOR*>((const void*)flow.data);
        for (int y = 0; y < flowHeight; ++y)
        {
            for (int x = 0; x < flowWidth; ++x)
            {
                int linearIndex = y * flowWidth + x;
                flowData[2 * linearIndex] = (float)(_flowVectors[linearIndex].flowx / float(1 << 5));
                flowData[2 * linearIndex + 1] = (float)(_flowVectors[linearIndex].flowy / float(1 << 5));
            }
        }
        cv::Mat outFlow(cv::Size(flowWidth, flowHeight), CV_32FC2, flowData.get());
        outFlow.copyTo(flowVectors);
    }
    std::swap(m_CurrIndex, m_PrevIndex);
}

void CNvOpticalFlowOCV::ConvertABGRToY(cv::InputArray input, cv::OutputArray output)
{
    if (input.isMat())
    {
        assert(output.isMat());
        cv::Mat inBGRA;
        std::vector<cv::Mat> perChannelABGR;
        cv::split(input, perChannelABGR);
        std::vector<cv::Mat> perChannelBGRA = { perChannelABGR[1], perChannelABGR[2], perChannelABGR[3], perChannelABGR[0] };
        cv::merge(perChannelBGRA, inBGRA);

        cv::cvtColor(inBGRA, output, cv::COLOR_BGRA2GRAY);
    }
    else if (input.isGpuMat())
    {
        assert(output.isGpuMat());
        cv::cuda::GpuMat inBGRA;
        std::vector<cv::cuda::GpuMat> perChannelABGR;
        cv::cuda::split(input, perChannelABGR);
        std::vector<cv::cuda::GpuMat> perChannelBGRA = { perChannelABGR[1], perChannelABGR[2], perChannelABGR[3], perChannelABGR[0] };
        cv::cuda::merge(perChannelBGRA, inBGRA);

        cv::cuda::cvtColor(inBGRA, output, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        assert(0);
    }
}
