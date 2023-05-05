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
#include <unordered_map>
#include <d3d11.h>
#include <Windows.h>
#include <wrl.h>
#include "NvOFD3D11.h"
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFUtilsD3D11.h"
#include "NvOFCmdParser.h"

void EstimateStereoDisparity(ID3D11Device *pDevice,
    ID3D11DeviceContext *pContext,
    std::string inputFileNameL,
    std::string inputFileNameR,
    std::string outputFileBaseName,
    NV_OF_PERF_LEVEL perfPreset,
    uint32_t gridSize,
    bool saveFlowAsImage)
{

    //Create Dataloader 
    std::unique_ptr<NvOFDataLoader> dataLoaderL = CreateDataloader(inputFileNameL);
    if (!dataLoaderL)
    {
        std::ostringstream err;
        err << "Unable to load left view input data: " << inputFileNameL << std::endl;
        throw std::invalid_argument(err.str());
    }

    std::unique_ptr<NvOFDataLoader> dataLoaderR = CreateDataloader(inputFileNameR);
    if (!dataLoaderL)
    {
        std::ostringstream err;
        err << "Unable to load right view input data: " << inputFileNameR << std::endl;
        throw std::invalid_argument(err.str());
    }

    uint32_t width = dataLoaderL->GetWidth();
    uint32_t height = dataLoaderL->GetHeight();
    uint32_t nFrameSize = width * height;
    uint32_t nScaleFactor = 1;

     auto nvOpticalFlow = NvOFD3D11::Create(pDevice, pContext, width, height,
        dataLoaderL->GetBufferFormat(), NV_OF_MODE_STEREODISPARITY, perfPreset);

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

    const uint32_t NUM_INPUT_BUFFERS = 2;
    const uint32_t NUM_OUTPUT_BUFFERS = 1;

    std::vector<NvOFBufferObj> inputBuffers;
    std::vector<NvOFBufferObj> outputBuffers;
    inputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, NUM_INPUT_BUFFERS);
    outputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

    std::unique_ptr<NvOFUtils> nvOFUtils(new NvOFUtilsD3D11(pDevice, pContext, NV_OF_MODE_STEREODISPARITY));
    std::vector<NvOFBufferObj> upsampleBuffers;
    std::unique_ptr<NV_OF_STEREO_DISPARITY[]> pOut;
    std::unique_ptr<NvOFFileWriter> stereoFileWriter;
    if (nScaleFactor > 1)
    {
        auto nOutWidth = (width + gridSize - 1) / gridSize;
        auto nOutHeight = (height + gridSize - 1) / gridSize;

        upsampleBuffers = nvOpticalFlow->CreateBuffers(nOutWidth, nOutHeight, NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

        auto nOutSize = nOutWidth * nOutHeight;
        pOut.reset(new NV_OF_STEREO_DISPARITY[nOutSize]);
        if (pOut == nullptr)
        {
            std::ostringstream err;
            err << "Failed to allocate output host memory of size " << nOutSize * sizeof(NV_OF_STEREO_DISPARITY) << " bytes" << std::endl;
            throw std::bad_alloc();
        }

        stereoFileWriter = NvOFFileWriter::Create(nOutWidth,
            nOutHeight,
            NV_OF_MODE_STEREODISPARITY,
            32.0f);
    }
    else
    {
        uint32_t nOutSize = outputBuffers[0]->getWidth() * outputBuffers[0]->getHeight();
        pOut.reset(new NV_OF_STEREO_DISPARITY[nOutSize]);
        if (pOut == nullptr)
        {
            std::ostringstream err;
            err << "Failed to allocate output host memory of size " << nOutSize * sizeof(NV_OF_STEREO_DISPARITY) << " bytes" << std::endl;
            throw std::bad_alloc();
        }

        stereoFileWriter = NvOFFileWriter::Create(outputBuffers[0]->getWidth(),
            outputBuffers[0]->getHeight(),
            NV_OF_MODE_STEREODISPARITY,
            32.0f);
    }

    uint32_t frameCount = 0;
    const uint32_t LEFT_VIEW = 0;
    const uint32_t RIGHT_VIEW = 1;
    while (!dataLoaderL->IsDone() && !dataLoaderR->IsDone())
    {
        inputBuffers[LEFT_VIEW]->UploadData(dataLoaderL->CurrentItem());
        inputBuffers[RIGHT_VIEW]->UploadData(dataLoaderR->CurrentItem());

        nvOpticalFlow->Execute(inputBuffers[LEFT_VIEW].get(), inputBuffers[RIGHT_VIEW].get(), outputBuffers[0].get());

        if (nScaleFactor > 1)
        {
            nvOFUtils->Upsample(outputBuffers[0].get(), upsampleBuffers[0].get(), nScaleFactor);
            upsampleBuffers[0]->DownloadData(pOut.get());
        }
        else
        {
            outputBuffers[0]->DownloadData(pOut.get());
        }

        stereoFileWriter->SaveOutput((void*)pOut.get(),
            outputFileBaseName, frameCount, saveFlowAsImage);

        dataLoaderL->Next();
        dataLoaderR->Next();
        frameCount++;
    }
}


int main(int argc, char **argv)
{
    std::string inputFileNameL;
    std::string inputFileNameR;
    std::string outputFileBaseName = "./out";
    std::string preset = "medium";
    bool visualFlow = 0;
    uint32_t gridSize = 1;

    NV_OF_PERF_LEVEL perfPreset = NV_OF_PERF_LEVEL_MEDIUM;
    int gpuId = 0;
    int showHelp = 0;
    std::unordered_map<std::string, NV_OF_PERF_LEVEL> presetMap = {
        { "slow", NV_OF_PERF_LEVEL_SLOW },
        { "medium", NV_OF_PERF_LEVEL_MEDIUM },
        { "fast", NV_OF_PERF_LEVEL_FAST } };

    try
    {
        NvOFCmdParser cmdParser;
        cmdParser.AddOptions("inputL", inputFileNameL, "Input filename for left view "
            "[ e.g. inputDir" DIR_SEP "inputLeft*.png, "
            "inputDir" DIR_SEP "inputLeft_%d.png, "
            "inputDir" DIR_SEP "inputLeft_wxh.yuv ]");
        cmdParser.AddOptions("inputR", inputFileNameR, "Input filename for right view "
            "[ e.g. inputDir" DIR_SEP "inputRight*.png, "
            "inputDir" DIR_SEP "inputRight_%d.png, "
            "inputDir" DIR_SEP "inputRight_wxh.yuv ]");
        cmdParser.AddOptions("output", outputFileBaseName, "Output file base name "
            "[ e.g. outputDir" DIR_SEP "outFilename ]");
        cmdParser.AddOptions("gpuIndex", gpuId, "D3D11 adapter ordinal");
        cmdParser.AddOptions("preset", preset, "perf preset for stereo disparity algo [ options : slow, medium, fast ]");
        cmdParser.AddOptions("visualFlow", visualFlow, "save flow vectors as RGB image");
        cmdParser.AddOptions("gridSize", gridSize, "Block size per motion vector");

        NVOF_ARGS_PARSE(cmdParser, argc, (const char**)argv);

        if (inputFileNameL.empty())
        {
            std::cout << "Invalid input filename for left view" << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

        if (inputFileNameR.empty())
        {
            std::cout << "Invalid input filename for right view" << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

        auto presetIt = presetMap.find(preset);
        if (presetIt == presetMap.end())
        {
            std::cout << "Invalid preset level : " << preset << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }

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

        EstimateStereoDisparity(pDevice.Get(),
            pContext.Get(),
            inputFileNameL, 
            inputFileNameR,
            outputFileBaseName,
            presetIt->second,
            gridSize,
            !!visualFlow);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
