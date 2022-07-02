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

#include "cuda.h"
#include "cuda_runtime.h"
#include "NvOF.h"
#include "NvOFCuda.h"
#include "NvOFUtils.h"

void FlowUpsample(cudaArray_t srcArray, void* srcDevPtr, uint32_t nSrcWidth, uint32_t nSrcPitch, uint32_t nSrcHeight,
                  cudaArray_t dstArray, void* dstDevPtr, uint32_t nDstWidth, uint32_t nDstPitch, uint32_t nDstHeight,
                  uint32_t nScaleFactor);
void StereoDisparityUpsample(cudaArray_t srcArray, void* srcDevPtr, uint32_t nSrcWidth, uint32_t nSrcPitch, uint32_t nSrcHeight,
                             cudaArray_t dstArray, void* dstDevPtr, uint32_t nDstWidth, uint32_t nDstPitch, uint32_t nDstHeight,
                             uint32_t nScaleFactor);

NvOFUtilsCuda::NvOFUtilsCuda(NV_OF_MODE eMode)
    : NvOFUtils(eMode)
{
}

static inline bool isCudaArray(NvOFBuffer* buffer)
{
    NvOFBufferCudaArray* cuarr = dynamic_cast<NvOFBufferCudaArray*>(buffer);
    return (cuarr) ? true : false;
}

void NvOFUtilsCuda::Upsample(NvOFBuffer* srcBuf, NvOFBuffer* dstBuf, uint32_t nScaleFactor)
{
    CUarray srcArr = nullptr, dstArr = nullptr;
    CUdeviceptr srcDevPtr = 0, dstDevPtr = 0;
    uint32_t nSrcWidth = srcBuf->getWidth();
    uint32_t nSrcHeight = srcBuf->getHeight();
    uint32_t nSrcPitch = 0;
    uint32_t nDstWidth = dstBuf->getWidth();
    uint32_t nDstHeight = dstBuf->getHeight();
    uint32_t nDstPitch = 0;

    bool srcIsCuArray = isCudaArray(srcBuf);
    if (srcIsCuArray)
    {
        NvOFBufferCudaArray* srcBufCUarray = dynamic_cast<NvOFBufferCudaArray*>(srcBuf);
        NvOFBufferCudaArray* dstBufCUarray = dynamic_cast<NvOFBufferCudaArray*>(dstBuf);
        srcArr = srcBufCUarray->getCudaArray();
        dstArr = dstBufCUarray->getCudaArray();
    }
    else
    {
        NvOFBufferCudaDevicePtr* srcBufDevPtr = dynamic_cast<NvOFBufferCudaDevicePtr*>(srcBuf);
        NvOFBufferCudaDevicePtr* dstBufDevPtr = dynamic_cast<NvOFBufferCudaDevicePtr*>(dstBuf);
        srcDevPtr = srcBufDevPtr->getCudaDevicePtr();
        dstDevPtr = dstBufDevPtr->getCudaDevicePtr();
        nSrcPitch = srcBufDevPtr->getStrideInfo().strideInfo[0].strideXInBytes;
        nDstPitch = dstBufDevPtr->getStrideInfo().strideInfo[0].strideXInBytes;
    }

    if (m_eMode == NV_OF_MODE_OPTICALFLOW)
    {
        FlowUpsample((cudaArray_t)srcArr, (void*)srcDevPtr, nSrcWidth, nSrcPitch, nSrcHeight,
                     (cudaArray_t)dstArr, (void*)dstDevPtr, nDstWidth, nDstPitch, nDstHeight,
                     nScaleFactor);
    }
    else
    {
        StereoDisparityUpsample((cudaArray_t)srcArr, (void*)srcDevPtr, nSrcWidth, nSrcPitch, nSrcHeight,
                                (cudaArray_t)dstArr, (void*)dstDevPtr, nDstWidth, nDstPitch, nDstHeight,
                                nScaleFactor);
    }
}