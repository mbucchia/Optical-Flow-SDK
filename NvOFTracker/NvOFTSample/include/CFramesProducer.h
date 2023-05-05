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
#include "CNvOFTSampleException.h"

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <stdint.h>

class CFramesProducer
{
public:
    CFramesProducer(std::string& file, uint32_t width, uint32_t height, int gpuId = 0);
    int Decode(int& videoBytes);
    void GetFrames(std::vector<void*>& frames);
    uint32_t GetDecodeWidth() { return m_decodeWidth; }
    uint32_t GetDecodeHeight() { return m_decodeHeight; }

private:
    void* AllocCudaMemory(size_t size)
    {
        void* pCudaMem;
        CK_CUDA(cudaMalloc((void**)&pCudaMem, size));

        return pCudaMem;
    }

private:
    std::function<void(void*)> m_cudaRelease = [](void* pCudaMem) { if (pCudaMem) { cudaFree(pCudaMem); pCudaMem = nullptr; }};
    CUcontext m_context;
    std::unique_ptr<FFmpegDemuxer> m_demuxer;
    std::unique_ptr<NvDecoder> m_decoder;
    uint32_t m_scaledWidth;
    uint32_t m_scaledHeight;
    uint32_t m_decodeWidth;
    uint32_t m_decodeHeight;
    uint32_t m_decodePitch;
    std::unique_ptr<void, std::function<void(void*)>> m_RGBFrame;
    std::unique_ptr<void, std::function<void(void*)>> m_RGBScaledFrame;
    std::unique_ptr<void, std::function<void(void*)>> m_YScaledFrame;
};
