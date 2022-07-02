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

#include <fstream>
#include <iostream>
#include <memory>
#include <d3d11.h>
#include "NvOFD3D12.h"
#include <Windows.h>
#include <wrl.h>
#include <unordered_map>
#include <memory>
#include <wchar.h>
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFUtilsD3D12.h"
#include "NvOFCmdParser.h"
#include "NvOFD3DCommon.h"

void CPUWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue)
{
    if (pFence->GetCompletedValue() < nFenceValue)
    {
        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        D3D_API_CALL(pFence->SetEventOnCompletion(nFenceValue, event));
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
    }
}

void NvOFBatchExecute(NvOFObj &nvOpticalFlow,
    std::vector<NvOFBufferObj> &inputBuffers,
    std::vector<NvOFBufferObj> &outputBuffers,
    uint32_t batchSize,
    double &executionTime,
    bool measureFPS,
    NV_OF_FENCE_POINT* appFence,
    NV_OF_FENCE_POINT* ofaFence)
{
    //measureFPS makes sure that data upload is finished before kicking off
    //optical flow execute for a batch of frames. It also makes sure that 
    //optical flow batch execute is finished before output download.
    if (measureFPS)
    {
        NvOFStopWatch     nvStopWatch;

        nvStopWatch.Start();
        for (uint32_t i = 0; i < batchSize; i++)
        {
            ofaFence->value++;
            nvOpticalFlow->Execute(inputBuffers[i].get(),
                inputBuffers[i + 1].get(),
                outputBuffers[i].get(),
                nullptr,
                nullptr,
                0,
                nullptr,
                appFence,
                1,
                ofaFence);
        }
        CPUWaitForFencePoint(ofaFence->fence, ofaFence->value);
        executionTime += nvStopWatch.Stop();
    }
    else
    {
        for (uint32_t i = 0; i < batchSize; i++)
        {
            ofaFence->value++;
            nvOpticalFlow->Execute(inputBuffers[i].get(),
                inputBuffers[i + 1].get(),
                outputBuffers[i].get(), 
                nullptr,
                nullptr,
                0,
                nullptr,
                appFence,
                1,
                ofaFence);
        }
    }
}

std::vector<void*> AllocateBuffer(NV_OF_BUFFER_DESCRIPTOR ofDesc, ID3D12Device* pDevice, int iNumBuffer, int iGridSize=1)
{
    uint32_t widthAligned = 0;
    uint32_t heightAligned = 0;
    uint32_t width = ofDesc.width;
    uint32_t height = ofDesc.height;
    uint32_t outputGridSize = iGridSize;
 
    D3D12_HEAP_PROPERTIES heapProps = {};
    D3D12_RESOURCE_DESC desc = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = NvOFBufferFormatToDxgiFormat(ofDesc.bufferFormat);
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    std::vector<void*> resourceList;
    for (int i = 0; i < iNumBuffer; i++)
    {
        ID3D12Resource* pResource;
        D3D_API_CALL(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&pResource)));
        resourceList.emplace_back(pResource);
    }
    return resourceList;
}

void CreateFence(ID3D12Device* pDevice, NV_OF_FENCE_POINT* fence, int count)
{
    for (int i = 0; i < count; i++)
    {
        D3D_API_CALL(pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&(fence[i].fence))));
        fence[i].value = 0;
    }
}

void EstimateFlow(ID3D12Device *pDevice, std::string inputFileName,
    std::string outputFileBaseName,
    NV_OF_PERF_LEVEL perfPreset,
    uint32_t gridSize,
    bool saveFlowAsImage,
    bool measureFPS)
{
    //Create Dataloader 
    std::unique_ptr<NvOFDataLoader> dataLoader = CreateDataloader(inputFileName);
    if (!dataLoader)
    {
        std::ostringstream err;
        err << "Unable to load input data: " << inputFileName << std::endl;
        throw std::invalid_argument(err.str());
    }

    uint32_t width = dataLoader->GetWidth();
    uint32_t height = dataLoader->GetHeight();
    uint32_t nFrameSize = width * height;
    uint32_t nScaleFactor = 1;
    bool dumpOfOutput = (!outputFileBaseName.empty());
    NvOFObj nvOpticalFlow = NvOFD3D12::Create(pDevice, width, height,
        dataLoader->GetBufferFormat(),
        NV_OF_MODE_OPTICALFLOW,
        perfPreset);

    uint32_t hwGridSize;
    if (!nvOpticalFlow->CheckGridSize(gridSize))
    {
        if (!nvOpticalFlow->GetNextMinGridSize(gridSize, hwGridSize))
        {
            throw std::runtime_error("Invalid parameter");
        }
        else
        {
            nScaleFactor = hwGridSize / gridSize;
        }
    }
    else
    {
        hwGridSize = gridSize;
    }

    nvOpticalFlow->Init(hwGridSize);

    const uint32_t NUM_INPUT_BUFFERS = 16;
    const uint32_t NUM_OUTPUT_BUFFERS = NUM_INPUT_BUFFERS - 1;

    std::vector<NvOFBufferObj> inputBuffers;
    std::vector<NvOFBufferObj> outputBuffers;

    NV_OF_FENCE_POINT appFence;
    CreateFence(pDevice, &appFence, 1);
    NV_OF_FENCE_POINT ofaFence;
    CreateFence(pDevice, &ofaFence, 1);

    ID3D12CommandQueue* cmdQ;
    ID3D12CommandAllocator* cmdAlloc;
    ID3D12GraphicsCommandList* cmdList;

    D3D12_COMMAND_QUEUE_DESC cmdQDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    D3D_API_CALL(pDevice->CreateCommandQueue(&cmdQDesc, IID_PPV_ARGS(&cmdQ)));
    D3D_API_CALL(pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc)));
    D3D_API_CALL(pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc, nullptr, IID_PPV_ARGS(&cmdList)));
    cmdList->Close();
    
    NV_OF_BUFFER_DESCRIPTOR inputDesc;
    inputDesc.bufferFormat = dataLoader->GetBufferFormat();
    inputDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    inputDesc.width = width;
    inputDesc.height = height;
    std::vector<void*> inputResourceList = AllocateBuffer(inputDesc, pDevice, NUM_INPUT_BUFFERS);
    
    for (auto pResource : inputResourceList)
    {
        ofaFence.value++;
        inputBuffers.emplace_back(nvOpticalFlow->RegisterPreAllocBuffers(inputDesc, pResource,  &appFence, &ofaFence));
    }

    NV_OF_BUFFER_DESCRIPTOR outputDesc;
    outputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    outputDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    auto nOutWidth = (width + hwGridSize - 1) / hwGridSize;
    auto nOutHeight = (height + hwGridSize - 1) / hwGridSize;
    outputDesc.width = nOutWidth;
    outputDesc.height = nOutHeight;
    std::vector<void*> outputResourceList= AllocateBuffer(outputDesc, pDevice, NUM_OUTPUT_BUFFERS, hwGridSize);

    for (auto pResource : outputResourceList)
    {
        ofaFence.value++;
        outputBuffers.emplace_back(nvOpticalFlow->RegisterPreAllocBuffers(outputDesc, pResource, &appFence, &ofaFence));
    }

    std::unique_ptr<NvOFUtils> nvOFUtils(new NvOFUtilsD3D12(pDevice, cmdList, NV_OF_MODE_OPTICALFLOW));
    std::vector<NvOFBufferObj> upsampleBuffers;
    std::unique_ptr<NV_OF_FLOW_VECTOR[]> pOut;
    std::unique_ptr<NvOFFileWriter> flowFileWriter;

    if (nScaleFactor > 1)
    {
        auto nOutWidth = (width + gridSize - 1) / gridSize;
        auto nOutHeight = (height + gridSize - 1) / gridSize;
        
        NV_OF_BUFFER_DESCRIPTOR upsampleDesc;
        upsampleDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
        upsampleDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
        upsampleDesc.width = nOutWidth;
        upsampleDesc.height = nOutHeight;
        std::vector<void*> upsampleResourceList = AllocateBuffer(upsampleDesc, pDevice, NUM_OUTPUT_BUFFERS);

        for (auto pResource : upsampleResourceList)
        {
            ofaFence.value++;
            upsampleBuffers.emplace_back(nvOpticalFlow->RegisterPreAllocBuffers(upsampleDesc, pResource, &appFence, &ofaFence));
        }

        auto nOutSize = nOutWidth * nOutHeight;
        pOut.reset(new NV_OF_FLOW_VECTOR[nOutSize]);
        if (pOut == nullptr)
        {
            std::ostringstream err;
            err << "Failed to allocate output host memory of size " << nOutSize * sizeof(NV_OF_FLOW_VECTOR) << " bytes" << std::endl;
            throw std::bad_alloc();
        }

        flowFileWriter = NvOFFileWriter::Create(nOutWidth,
            nOutHeight,
            NV_OF_MODE_OPTICALFLOW,
            32.0f);
    }
    else
    {
        uint32_t nOutSize = outputBuffers[0]->getWidth() * outputBuffers[0]->getHeight();
        pOut.reset(new NV_OF_FLOW_VECTOR[nOutSize]);
        if (pOut == nullptr)
        {
            std::ostringstream err;
            err << "Failed to allocate output host memory of size " << nOutSize * sizeof(NV_OF_FLOW_VECTOR) << " bytes" << std::endl;
            throw std::bad_alloc();
        }

        flowFileWriter = NvOFFileWriterFlow::Create(outputBuffers[0]->getWidth(),
            outputBuffers[0]->getHeight(),
            NV_OF_MODE_OPTICALFLOW,
            32.0f);
    }

    uint32_t                        curFrameIdx = 0;
    uint32_t                        frameCount = 0;
    bool                            lastSet = false;
    double                          executionTime = 0;
    

    for (; (!dataLoader->IsDone() || curFrameIdx > 1); dataLoader->Next())
    {
        if (!dataLoader->IsDone())
        {
            appFence.value++;
            inputBuffers[curFrameIdx]->UploadData(dataLoader->CurrentItem(), &ofaFence, &appFence);
            CPUWaitForFencePoint(appFence.fence, appFence.value); 
        }
        else
        {
            // If number of frames is non multiple of NUM_INPUT_BUFFERS then execute will be
            // called for TotalFrames % NUM_INPUT_BUFFERS frames in last set.
            // No uploadData() called for last frame so curFrameIdx is decremented by 1.
            curFrameIdx--;
            lastSet = true;
        }

        if (curFrameIdx == (NUM_INPUT_BUFFERS - 1) || lastSet)
        {
            NvOFBatchExecute(nvOpticalFlow, inputBuffers, outputBuffers, curFrameIdx, executionTime, measureFPS, &appFence, &ofaFence);
            if (dumpOfOutput)
            {
                if (nScaleFactor > 1)
                {
                    cmdQ->Wait(ofaFence.fence, ofaFence.value);
                    for (uint32_t i = 0; i < curFrameIdx; i++)
                    {
                        D3D_API_CALL(cmdList->Reset(cmdAlloc, nullptr));

                        nvOFUtils->Upsample(outputBuffers[i].get(), upsampleBuffers[i].get(), nScaleFactor);
                        appFence.value++;
                        
                        cmdList->Close();
                        ID3D12CommandList * const ppCmdList[] = { cmdList };
                        cmdQ->ExecuteCommandLists(1, ppCmdList);

                        cmdQ->Signal(appFence.fence, appFence.value);
                        upsampleBuffers[i]->DownloadData(pOut.get(), &appFence);
                        flowFileWriter->SaveOutput((void*)pOut.get(),
                            outputFileBaseName, frameCount, saveFlowAsImage);
                        frameCount++;
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < curFrameIdx; i++)
                    {
                        outputBuffers[i]->DownloadData(pOut.get(), &ofaFence);
                        flowFileWriter->SaveOutput((void*)pOut.get(),
                            outputFileBaseName, frameCount, saveFlowAsImage);
                        frameCount++;
                    }
                }
            }
            else
            {
                frameCount += curFrameIdx;
            }

            if (lastSet)
            {
                break;
            }
            // Last frame of previous set of input buffers is reused as first element for next iteration.
            swap(inputBuffers[curFrameIdx], inputBuffers[0]);
            curFrameIdx = 0;
        }
        curFrameIdx++;
    }

    if (measureFPS)
    {
        double fps = (executionTime > 0.0) ? (frameCount / executionTime) : 1.0;
        std::cout << "Total Frames = " << frameCount << "\n";
        std::cout << "Time = " << executionTime << " s, NvOF FPS = " << fps << "\n";
    }
}

int main(int argc, char** argv)
{
    std::string inputFileName;
    std::string outputFileBaseName;
    std::string preset = "medium";
    bool visualFlow = false;
    bool measureFPS = false;
    uint32_t gridSize = 1;

    NV_OF_PERF_LEVEL perfPreset = NV_OF_PERF_LEVEL_MEDIUM;
    int gpuId = 0;
    std::unordered_map<std::string, NV_OF_PERF_LEVEL> presetMap = {
        {"slow", NV_OF_PERF_LEVEL_SLOW},
        { "medium", NV_OF_PERF_LEVEL_MEDIUM },
        { "fast", NV_OF_PERF_LEVEL_FAST } };

    Microsoft::WRL::ComPtr<ID3D12Device> pDevice;
    
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    try
    {
        NvOFCmdParser cmdParser;
        cmdParser.AddOptions("input", inputFileName, "Input filename "
            "[ e.g. inputDir" DIR_SEP "input*.png, "
            "inputDir" DIR_SEP "input%d.png, "
            "inputDir" DIR_SEP "input_wxh.yuv ]");
        cmdParser.AddOptions("output", outputFileBaseName, "Output file base name "
            "[ e.g. outputDir" DIR_SEP "outFilename ]");
        cmdParser.AddOptions("gpuIndex", gpuId, "D3D11 adapter ordinal");
        cmdParser.AddOptions("preset", preset, "perf preset for OF algo [ options : slow, medium, fast ]");
        cmdParser.AddOptions("visualFlow", visualFlow, "save flow vectors as RGB image");
        cmdParser.AddOptions("gridSize", gridSize, "Block size per motion vector");
        cmdParser.AddOptions("measureFPS", measureFPS, "Measure performance(frames per second). When this option is set it is not mandatory to specify --output option,"
            " output is generated only if --output option is specified");
        NVOF_ARGS_PARSE(cmdParser, argc, (const char**)argv);

        if (inputFileName.empty())
        {
            std::cout << "Input file not specified" << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

        if (!measureFPS && outputFileBaseName.empty())
        {
            std::cout << "Output file not specified" << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

        auto search = presetMap.find(preset);
        if (search == presetMap.end())
        {
            std::cout << "Invalid preset level : " << preset << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

        perfPreset = search->second;

        Microsoft::WRL::ComPtr<IDXGIFactory1> pFactory;
        Microsoft::WRL::ComPtr<IDXGIAdapter> pAdapter;
        D3D_API_CALL(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory));
        D3D_API_CALL(pFactory->EnumAdapters(gpuId, &pAdapter));

#if defined(_DEBUG)
        // Enable the debug layer (requires the Graphics Tools "optional feature").
        // NOTE: Enabling the debug layer after device creation will invalidate the active device.
        {
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
            {
                debugController->EnableDebugLayer();
            }
        }
#endif
        D3D_API_CALL(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&pDevice)));
        DXGI_ADAPTER_DESC adapterDesc;
        pAdapter->GetDesc(&adapterDesc);
        size_t descLength = wcslen(adapterDesc.Description) + 1;
        std::unique_ptr<char[]> gpuDesc(new char[descLength]);
        size_t numCharsReturned = 0;
        wcstombs_s(&numCharsReturned, gpuDesc.get(), descLength, adapterDesc.Description, wcslen(adapterDesc.Description));
        std::cout << "GPU in use: " << gpuDesc.get() << std::endl;
        EstimateFlow(pDevice.Get(), inputFileName,
            outputFileBaseName, perfPreset, gridSize, !!visualFlow, !!measureFPS);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
