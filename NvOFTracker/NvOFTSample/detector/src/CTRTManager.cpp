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

#include "CTRTManager.h"

CTRTManager::CTRTManager(const std::string& engineFile, uint32_t gpuId, bool allocInputDeviceMemory, bool allocOutputDeviceMemory) :
        m_AllocateInDevMem(allocInputDeviceMemory), m_AllocateOutDevMem(allocOutputDeviceMemory)
{
    if (engineFile.empty())
    {
        NVOFTSAMPLE_THROW_ERROR("Engine file empty");
    }
    std::ifstream file(engineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        m_Size = file.tellg();
        file.seekg(0, file.beg);
        m_TrtModelStream.resize(m_Size);
        file.read(m_TrtModelStream.data(), m_Size);
        file.close();
    }
    else
    {
        std::ostringstream oss;
        oss << "Engine file(" << engineFile << ") " << "read failed";
        NVOFTSAMPLE_THROW_ERROR(oss.str());
    }
    CK_CUDA(cudaSetDevice(gpuId));
    m_InferRuntime = {nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()), m_RuntimeDestroy};
    if (m_InferRuntime == nullptr)
    {
        NVOFTSAMPLE_THROW_ERROR("Infer runtime creation failed");
    }
    m_CudaEngine = {m_InferRuntime->deserializeCudaEngine(m_TrtModelStream.data(), m_Size, nullptr), m_CudaEngineDestroy};
    if (m_CudaEngine == nullptr)
    {
        NVOFTSAMPLE_THROW_ERROR("Deserializing the stream failed. Engine not created");
    }
    m_InferRuntime.release();
    m_Context = {m_CudaEngine->createExecutionContext(), m_ContextDestroy};
    if (m_Context == nullptr)
    {
        NVOFTSAMPLE_THROW_ERROR("TRT Execution context creation failed.");
    }

    // Allocate necessary device memory for inputs and outputs
    for (int b = 0; b < m_CudaEngine->getNbBindings(); ++b)
    {
        if (m_CudaEngine->bindingIsInput(b))
        {
            auto size = GetSize(b);
            m_InputSize.push_back(size);
            if (m_AllocateInDevMem)
            {
                DeviceAlloc allocation;
                allocation.alloc = { AllocCudaMemory(size), m_CudaRelease };
                allocation.size = size;
                m_DevInputs.push_back(std::move(allocation));
            }
        }
        else // output binding
        {
            auto size = GetSize(b);
            m_OutputSize.push_back(size);
            if (m_AllocateOutDevMem)
            {
                DeviceAlloc allocation;
                allocation.alloc = { AllocCudaMemory(size), m_CudaRelease };
                allocation.size = size;
                m_DevOutputs.push_back(std::move(allocation));
            }
        }
    }
}

/*
Call TensorRT to do the actual inference. This function will create device buffers for use in inference if the class 
was created with m_AllocateInDevMem/m_AllocateOutDevMem = true
Params:
inputBuffers: a vector of void* each pointing to deivce mem (when m_AllocateInDevMem = false) or host mem (when m_AllocateInDevMem = true)
outputBuffers: a vector of void* each pointing to deivce mem (when m_AllocateOutDevMem = false) or host mem (when m_AllocateOutDevMem = true)
trtstream: cuda stream on which the inference needs to run.
Returns:
void. The result of the inference will be present in outputBuffers.
*/
void CTRTManager::RunInference(std::vector<void*>& inputBuffers, std::vector<void*>& outputBuffers, cudaStream_t trtStream)
{
    std::vector<void *> buffers(m_CudaEngine->getNbBindings());
    using devAllocIter = std::vector<DeviceAlloc>::iterator;
    using voidPIter = std::vector<void *>::iterator;

    // Copy inputs from host memory to device memory
    if (m_AllocateInDevMem)
    {
        for (std::pair<devAllocIter, voidPIter> it(m_DevInputs.begin(), inputBuffers.begin());
             it.first != m_DevInputs.end() && it.second != inputBuffers.end();
             ++it.first, ++it.second)
        {
            CK_CUDA(cudaMemcpy(it.first->alloc.get(), *it.second, it.first->size, cudaMemcpyHostToDevice));
        }
    }

    // TensorRT needs input and output buffers on device as an array of pointers.
    // Based on the binding index obtained from engine, associate the input and output buffers
    // accordingly
    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;
    for (int b = 0; b < m_CudaEngine->getNbBindings(); ++b)
    {
        if (m_CudaEngine->bindingIsInput(b))
        {
            if (m_AllocateInDevMem)
            {
                buffers[b] = m_DevInputs[inputIndex++].alloc.get();
            }
            else
            {
                buffers[b] = inputBuffers[inputIndex++];
            }
        }
        else // output binding
        {
            if (m_AllocateOutDevMem)
            {
                buffers[b] = m_DevOutputs[outputIndex++].alloc.get();
            }
            else
            {
                buffers[b] = outputBuffers[outputIndex++];
            }
        }
    }

    // Call TensorRT enqueue do the actual inference
    if (m_Context->enqueueV2(buffers.data(), trtStream, nullptr) != true)
    {
        NVOFTSAMPLE_THROW_ERROR("TensorRT Kernel Enqueue failed");
    }

    // Copy outputs from device to host memory
    if (m_AllocateOutDevMem)
    {
        for (std::pair<devAllocIter, voidPIter> it(m_DevOutputs.begin(), outputBuffers.begin());
             it.first != m_DevOutputs.end() && it.second != outputBuffers.end();
             ++it.first, ++it.second)
        {
            CK_CUDA(cudaMemcpy(*it.second, it.first->alloc.get(), it.first->size, cudaMemcpyDeviceToHost));
        }
    }
}

/*
Get the size of a buffer based on given biding index.
Params:
bindingIndex: TRT assigned binding index for the surface whose size is being queried
Returns:
size_t: size of the surface attached to bindingIndex.
*/
size_t CTRTManager::GetSize(int bindingIndex)
{
    using namespace nvinfer1;
    Dims dims = m_CudaEngine->getBindingDimensions(bindingIndex);
    DataType dataType = m_CudaEngine->getBindingDataType(bindingIndex);
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    int dataTypeSize = 0;
    switch (dataType)
    {
    case DataType::kFLOAT:
        dataTypeSize = sizeof(float);
        break;
    case DataType::kHALF:
        dataTypeSize = sizeof(uint16_t);
        break;
    case DataType::kINT8:
        dataTypeSize = sizeof(uint8_t);
        break;
    case DataType::kINT32:
        dataTypeSize = sizeof(uint32_t);
        break;
    default:
        assert(0);
        break;
    }

    return size * dataTypeSize;
}
