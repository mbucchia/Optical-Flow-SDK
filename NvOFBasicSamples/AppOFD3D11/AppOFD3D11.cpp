/*
* Copyright (c) 2018-2023 NVIDIA Corporation
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the software, and to permit persons to whom the
* software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/


#include <fstream>
#include <iostream>
#include <memory>
#include <d3d11.h>
#include "NvOFD3D11.h"
#include <Windows.h>
#include <wrl.h>
#include <unordered_map>
#include <memory>
#include <wchar.h>
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFUtilsD3D11.h"
#include "NvOFCmdParser.h"
#include "NvOFD3DCommon.h"

void NvOFBatchExecute(NvOFObj &nvOpticalFlow,
                      std::vector<NvOFBufferObj> &inputBuffers,
                      std::vector<NvOFBufferObj> &outputBuffers,
                      uint32_t batchSize,
                      double &executionTime,
                      bool measureFPS)
{
    //measureFPS makes sure that data upload is finished before kicking off
    //optical flow execute for a batch of frames. It also makes sure that 
    //optical flow batch execute is finished before output download.
    if (measureFPS)
    {
        NvOFStopWatch     nvStopWatch;

        inputBuffers[batchSize]->SyncBuffer();
        nvStopWatch.Start();
        for (uint32_t i = 0; i < batchSize; i++)
        {
            nvOpticalFlow->Execute(inputBuffers[i].get(),
                inputBuffers[i + 1].get(),
                outputBuffers[i].get());
        }
        outputBuffers[batchSize - 1]->SyncBuffer();
        executionTime += nvStopWatch.Stop();
    }
    else
    {
        for (uint32_t i = 0; i < batchSize; i++)
        {
            nvOpticalFlow->Execute(inputBuffers[i].get(),
                inputBuffers[i + 1].get(),
                outputBuffers[i].get());
        }
    }
}

void EstimateFlow(ID3D11Device *pDevice, ID3D11DeviceContext *pContext, std::string inputFileName,
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

    NvOFObj nvOpticalFlow = NvOFD3D11::Create(pDevice, pContext, width, height,
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
    const uint32_t NUM_OUTPUT_BUFFERS = NUM_INPUT_BUFFERS-1;

    std::vector<NvOFBufferObj> inputBuffers;
    std::vector<NvOFBufferObj> outputBuffers;

    inputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, NUM_INPUT_BUFFERS);
    outputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

    std::unique_ptr<NvOFUtils> nvOFUtils(new NvOFUtilsD3D11(pDevice, pContext, NV_OF_MODE_OPTICALFLOW));
    std::vector<NvOFBufferObj> upsampleBuffers;
    std::unique_ptr<NV_OF_FLOW_VECTOR[]> pOut;
    std::unique_ptr<NvOFFileWriter> flowFileWriter;
    if (nScaleFactor > 1)
    {
        auto nOutWidth = (width + gridSize - 1) / gridSize;
        auto nOutHeight = (height + gridSize - 1) / gridSize;

        upsampleBuffers = nvOpticalFlow->CreateBuffers(nOutWidth, nOutHeight, NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

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
            inputBuffers[curFrameIdx]->UploadData(dataLoader->CurrentItem());
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
            NvOFBatchExecute(nvOpticalFlow, inputBuffers, outputBuffers, curFrameIdx, executionTime, measureFPS);
            if (dumpOfOutput)
            {
                for (uint32_t i = 0; i < curFrameIdx; i++)
                {
                    if (nScaleFactor > 1)
                    {
                        nvOFUtils->Upsample(outputBuffers[i].get(), upsampleBuffers[i].get(), nScaleFactor);
                        upsampleBuffers[i]->DownloadData(pOut.get());
                    }
                    else
                    {
                        outputBuffers[i]->DownloadData(pOut.get());
                    }
                    flowFileWriter->SaveOutput((void*)pOut.get(),
                        outputFileBaseName, frameCount, saveFlowAsImage);
                    frameCount++;
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
        { "fast", NV_OF_PERF_LEVEL_FAST }};

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

        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        D3D_API_CALL(D3D11CreateDevice(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
            NULL, 0, D3D11_SDK_VERSION, &pDevice, NULL, &pContext));

        DXGI_ADAPTER_DESC adapterDesc;
        pAdapter->GetDesc(&adapterDesc);
        size_t descLength = wcslen(adapterDesc.Description) + 1;
        std::unique_ptr<char[]> gpuDesc(new char[descLength]);
        size_t numCharsReturned = 0;
        wcstombs_s(&numCharsReturned, gpuDesc.get(), descLength, adapterDesc.Description, wcslen(adapterDesc.Description));
        std::cout << "GPU in use: " << gpuDesc.get() << std::endl;
        EstimateFlow(pDevice.Get(), pContext.Get(), inputFileName,
                    outputFileBaseName, perfPreset, gridSize, !!visualFlow, !!measureFPS);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
