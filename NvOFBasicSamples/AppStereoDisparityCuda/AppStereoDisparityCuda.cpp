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
#include <cuda.h>
#include <unordered_map>
#include "NvOFCuda.h"
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFCmdParser.h"

void EstimateStereoDisparity(CUcontext cuContext,
    CUstream inputStream,
    CUstream outputStream,
    std::string inputFileNameL,
    std::string inputFileNameR,
    std::string outputFileBaseName,
    NV_OF_CUDA_BUFFER_TYPE inputBufferType,
    NV_OF_CUDA_BUFFER_TYPE outputBufferType,
    NV_OF_PERF_LEVEL perfPreset,
    uint32_t gridSize,
    bool saveFlowAsImage)
{
    std::unique_ptr<NvOFDataLoader> dataLoaderL = CreateDataloader(inputFileNameL);
    std::unique_ptr<NvOFDataLoader> dataLoaderR = CreateDataloader(inputFileNameR);

    uint32_t width = dataLoaderL->GetWidth();
    uint32_t height = dataLoaderL->GetHeight();
    uint32_t nFrameSize = width * height;
    uint32_t nScaleFactor = 1;

    NvOFObj nvOpticalFlow = NvOFCuda::Create(cuContext, width, height, dataLoaderL->GetBufferFormat(),
        inputBufferType, outputBufferType, NV_OF_MODE_STEREODISPARITY, perfPreset, inputStream, outputStream);

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

    std::unique_ptr<NvOFUtils> nvOFUtils(new NvOFUtilsCuda(NV_OF_MODE_STEREODISPARITY));
    std::vector<NvOFBufferObj> upsampleBuffers;
    std::unique_ptr<NV_OF_STEREO_DISPARITY[]> pOut;
    std::unique_ptr<NvOFFileWriter> stereoFileWriter;
    if (nScaleFactor > 1)
    {
        auto nOutWidth = (width + gridSize - 1) / gridSize;
        auto nOutHeight = (height + gridSize - 1) / gridSize;

        upsampleBuffers = nvOpticalFlow->CreateBuffers(nOutWidth, nOutHeight, NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

        uint32_t nOutSize = nOutWidth * nOutHeight;
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

    CUDA_DRVAPI_CALL(cuCtxSynchronize());
}


int main(int argc, char **argv)
{
    std::string inputFileNameL;
    std::string inputFileNameR;
    std::string outputFileBaseName = "./out";
    std::string preset = "medium";
    std::string inputBufferType = "cudaArray";
    std::string outputBufferType = "cudaArray";
    int gridSize = 1;
    bool visualFlow = false;
    int gpuId = 0;
    bool useCudaStream = false;

    CUcontext cuContext = nullptr;
    CUstream   inputStream = nullptr;
    CUstream   outputStream = nullptr;

    std::unordered_map<std::string, NV_OF_PERF_LEVEL> presetMap = {
        { "slow", NV_OF_PERF_LEVEL_SLOW },
        { "medium", NV_OF_PERF_LEVEL_MEDIUM },
        { "fast", NV_OF_PERF_LEVEL_FAST } };

    std::unordered_map<std::string, NV_OF_CUDA_BUFFER_TYPE> bufferTypeMap = {
        { "cudaArray", NV_OF_CUDA_BUFFER_TYPE_CUARRAY },
        { "cudaDevicePtr", NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR } };

    NV_OF_PERF_LEVEL perfPreset = NV_OF_PERF_LEVEL_MEDIUM;
    NV_OF_CUDA_BUFFER_TYPE inputBufferTypeEnum = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;
    NV_OF_CUDA_BUFFER_TYPE outputBufferTypeEnum = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;

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
        cmdParser.AddOptions("gpuIndex", gpuId, "cuda device index");
        cmdParser.AddOptions("preset", preset, "perf preset for stereo disparity algo [ options : slow, medium, fast ]");
        cmdParser.AddOptions("visualFlow", visualFlow, "save flow vectors as RGB image");
        cmdParser.AddOptions("inputBufferType", inputBufferType, "input cuda buffer type [options : cudaArray cudaDevicePtr]");
        cmdParser.AddOptions("outputBufferType", outputBufferType, "output cuda buffer type [options : cudaArray cudaDevicePtr]");
        cmdParser.AddOptions("useCudaStream", useCudaStream, "Use cuda stream for input and output processing");
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

        auto search = presetMap.find(preset);
        if (search == presetMap.end())
        {
            std::cout << "Invalid preset level : " << preset << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }
        perfPreset = search->second;

        auto inputbufferTypeIt = bufferTypeMap.find(inputBufferType);
        if (inputbufferTypeIt == bufferTypeMap.end())
        {
            std::cout << "Invalid input buffer type : " << inputBufferType << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }
        inputBufferTypeEnum = inputbufferTypeIt->second;

        auto outputBufferTypeIt = bufferTypeMap.find(outputBufferType);
        if (outputBufferTypeIt == bufferTypeMap.end())
        {
            std::cout << "Invalid output buffer type : " << outputBufferType << std::endl;
            std::cout << cmdParser.help(argv[0]) << std::endl;
            return 1;
        }
        outputBufferTypeEnum = outputBufferTypeIt->second;

        int nGpu = 0;
        CUDA_DRVAPI_CALL(cuInit(0));
        CUDA_DRVAPI_CALL(cuDeviceGetCount(&nGpu));
        if (gpuId < 0 || gpuId >= nGpu)
        {
            std::cout << "GPU ordinal out of range. Should be with in [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            return 1;
        }
        CUdevice cuDevice = 0;
        CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, gpuId));
        char szDeviceName[80];
        CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;
        CUDA_DRVAPI_CALL(cuCtxCreate(&cuContext, 0, cuDevice));


        if (useCudaStream)
        {
            CUDA_DRVAPI_CALL(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
            CUDA_DRVAPI_CALL(cuStreamCreate(&outputStream, CU_STREAM_DEFAULT));
        }

        EstimateStereoDisparity(cuContext,
            inputStream,
            outputStream,
            inputFileNameL,
            inputFileNameR,
            outputFileBaseName,
            inputBufferTypeEnum,
            outputBufferTypeEnum,
            perfPreset,
            gridSize,
            !!visualFlow);


        if (useCudaStream)
        {
            CUDA_DRVAPI_CALL(cuStreamDestroy(outputStream));
            outputStream = nullptr;
            CUDA_DRVAPI_CALL(cuStreamDestroy(inputStream));
            inputStream = nullptr;
        }
        CUDA_DRVAPI_CALL(cuCtxDestroy(cuContext));
        cuContext = nullptr;
    }
    catch (const std::exception &ex)
    {
        if (outputStream)
        {
            CUDA_DRVAPI_CALL(cuStreamDestroy(outputStream));
        }

        if (inputStream)
        {
            CUDA_DRVAPI_CALL(cuStreamDestroy(inputStream));
        }

        if (cuContext)
        {
            CUDA_DRVAPI_CALL(cuCtxDestroy(cuContext));
        }

        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
