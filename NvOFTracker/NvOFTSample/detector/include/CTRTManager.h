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
// TensorRT related headers
#include "logger.h"
#include "NvInfer.h"

#include "CNvOFTSampleException.h"
// Cuda headers
#include <cuda_runtime.h>
// Others
#include <vector>
#include <memory>
#include <fstream>
#include <functional>
#include <iostream>

// type definition for cudaMalloc RAII
using CuAlloc = std::unique_ptr<void, std::function<void(void*)>>;
struct DeviceAlloc
{
    CuAlloc alloc;
    size_t size;
};

/*
Class to manage all TensorRT(TRT) operations. It takes in serialized engine file name and creates all
necessary TRT components(plugin factory, infer runtime, engine and execution context) needed to run
inference. Inference supports gpu/cpu surfaces and cuda streams(but, stream synch is client responsibility)
*/
class CTRTManager 
{
public:
    CTRTManager(const std::string& engineFile, uint32_t gpuId = 0, bool allocInputDeviceMemory = true, bool allocOutputDeviceMemory = true);
    CTRTManager(const CTRTManager& CTRTManager) = delete;
    CTRTManager& operator=(CTRTManager CTRTManager) = delete;
    void RunInference(std::vector<void*>& inputBuffers, std::vector<void*>& outputBuffers, cudaStream_t trtStream = NULL);
    // Gives a vector containing size of each of the inputs
    std::vector<size_t> GetInputSize() const
    {
        return m_InputSize;
    }
    // Gives a vector containing size of each of the outputs
    std::vector<size_t> GetOutputSize() const
    {
        return m_OutputSize;
    }

private:
    size_t GetSize(int bindingIndex);
    void* AllocCudaMemory(size_t size)
    {
        void* pCudaMem;
        CK_CUDA(cudaMalloc((void**)&pCudaMem, size));

        return pCudaMem;
    }

private:
    // Destruction lambdas for each of the TRT components.
    std::function<void(nvinfer1::IRuntime*)> m_RuntimeDestroy           = [](nvinfer1::IRuntime* pRuntime) { if (pRuntime) { pRuntime->destroy(); pRuntime = nullptr; }};
    std::function<void(nvinfer1::ICudaEngine*)> m_CudaEngineDestroy     = [](nvinfer1::ICudaEngine* pCudaEngine) { if (pCudaEngine) { pCudaEngine->destroy(); pCudaEngine = nullptr; }};
    std::function<void(nvinfer1::IExecutionContext* )> m_ContextDestroy = [](nvinfer1::IExecutionContext* pContext) { if (pContext) { pContext->destroy(); pContext = nullptr; }};
    // Destruction lambda for realising memory allocated through cudaMalloc
    std::function<void(void*)> m_CudaRelease                            = [](void* pCudaMem) { if (pCudaMem) { cudaFree(pCudaMem); pCudaMem = nullptr; }};
    // A char stream of deserialized model.
    std::vector<char> m_TrtModelStream;
    // Deserialized model stream size
    size_t m_Size;
    // TRT components
    std::unique_ptr<nvinfer1::IRuntime, decltype(m_RuntimeDestroy)> m_InferRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, decltype(m_CudaEngineDestroy)> m_CudaEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, decltype(m_ContextDestroy)> m_Context;
    // Input and Output memory on the device used during inference
    std::vector<DeviceAlloc> m_DevInputs;
    std::vector<DeviceAlloc> m_DevOutputs;
    std::vector<size_t> m_InputSize;
    std::vector<size_t> m_OutputSize;
    // Controls to instruct on whether the class needs to allocate device memory on behalf of the client
    bool m_AllocateInDevMem = true;
    bool m_AllocateOutDevMem = true;
};
