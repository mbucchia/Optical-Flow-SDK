/*
* Copyright (c) 2022-2023 NVIDIA Corporation
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


#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <wchar.h>
#include "NvOFVulkan.h"
#include "NvOFDataLoader.h"
#include "NvOFUtilsVulkan.h"
#include "NvOFUtils.h"
#include "NvOFCmdParser.h"

using NV_OF_RES_VK = std::unique_ptr<NV_OF_RESOURCE_VK, const std::function<void(NV_OF_RESOURCE_VK*)>>;

void ClientWaitSync(VkDevice device, VkSemaphore semaphore, uint64_t waitValue)
{
    uint64_t completedValue;
    vkGetSemaphoreCounterValue(device, semaphore, &completedValue);
    if (completedValue < waitValue)
    {
        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &semaphore;
        waitInfo.pValues = &waitValue;
        VK_API_CALL(vkWaitSemaphores(device, &waitInfo, UINT64_MAX));
    }
}

std::vector<NV_OF_RES_VK> AllocateVkResources(NvOFObj& nvOpticalFlow, NV_OF_BUFFER_DESCRIPTOR ofDesc, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, int iNumBuffer, 
    const std::function<void(NV_OF_RESOURCE_VK*)> lamdhaDeAllocateVkRes, uint32_t iGridSize = 1)
{
    uint32_t widthAligned = 0;
    uint32_t heightAligned = 0;
    uint32_t width = ofDesc.width;
    uint32_t height = ofDesc.height;
    uint32_t outputGridSize = iGridSize;

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkImageCreateInfo imageCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = GNvOFUtilsVulkan::NvOFBufferFormatToVkFormat(ofDesc.bufferFormat);
    imageCreateInfo.extent = { width, height, 1 };
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    std::vector<NV_OF_RES_VK> resourceList;
    for (int i = 0; i < iNumBuffer; i++)
    {
        NV_OF_RES_VK pResource(new NV_OF_RESOURCE_VK(), lamdhaDeAllocateVkRes);
        size_t size = 0;

        NvOFUtilsVulkan::EnableResourceOpticalFlow(instance, physicalDevice, ofDesc.bufferUsage, NV_OF_MODE_OPTICALFLOW, &imageCreateInfo, &size, NULL);
        std::unique_ptr<uint8_t[]> pImageCreateInfoOF(new uint8_t[size]);
        NvOFUtilsVulkan::EnableResourceOpticalFlow(instance, physicalDevice, ofDesc.bufferUsage, NV_OF_MODE_OPTICALFLOW, &imageCreateInfo, &size, (VkImageCreateInfo*)pImageCreateInfoOF.get());

        VK_API_CALL(vkCreateImage(device, (VkImageCreateInfo*)pImageCreateInfoOF.get(), NULL, &pResource->image));

        VkMemoryRequirements mem_reqs;
        vkGetImageMemoryRequirements(device, pResource->image, &mem_reqs);

        pResource->allocationSize = mem_reqs.size;

        VkMemoryAllocateInfo memAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        memAllocInfo.allocationSize = mem_reqs.size;
        for (memAllocInfo.memoryTypeIndex = 0; memAllocInfo.memoryTypeIndex < VK_MAX_MEMORY_TYPES; memAllocInfo.memoryTypeIndex++)
        {
            if ((mem_reqs.memoryTypeBits & (1 << memAllocInfo.memoryTypeIndex)) &&
                (memoryProperties.memoryTypes[memAllocInfo.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                break;
            }
        }

        if (memAllocInfo.memoryTypeIndex == VK_MAX_MEMORY_TYPES)
        {
            throw std::runtime_error("No suitable device memory available");
        }

        size = 0;
        NvOFUtilsVulkan::EnableResourceOpticalFlow(instance, physicalDevice, ofDesc.bufferUsage, NV_OF_MODE_OPTICALFLOW, &memAllocInfo, &size, NULL);
        std::unique_ptr<uint8_t[]> pMemAllocInfoOF(new uint8_t[size]);
        NvOFUtilsVulkan::EnableResourceOpticalFlow(instance, physicalDevice, ofDesc.bufferUsage, NV_OF_MODE_OPTICALFLOW, &memAllocInfo, &size, (VkMemoryAllocateInfo*)pMemAllocInfoOF.get());

        VK_API_CALL(vkAllocateMemory(device, (VkMemoryAllocateInfo*)pMemAllocInfoOF.get(), NULL, &pResource->memory));
        VK_API_CALL(vkBindImageMemory(device, pResource->image, pResource->memory, 0));
        resourceList.emplace_back(std::move(pResource));
    }
    return resourceList;
}

void NvOFBatchExecute(NvOFObj& nvOpticalFlow,
    std::vector<NvOFBufferObj>& inputBuffers,
    std::vector<NvOFBufferObj>& outputBuffers,
    uint32_t batchSize,
    double& executionTime,
    bool measureFPS,
    VkDevice device,
    NV_SYNC_VK appSync,
    NV_SYNC_VK ofaSync)
{
    NvOFStopWatch     nvStopWatch;

    if (measureFPS)
    {
        nvStopWatch.Start();
    }

    for (uint32_t i = 0; i < batchSize; i++)
    {
        ofaSync->value++;
        nvOpticalFlow->Execute(inputBuffers[i].get(),
            inputBuffers[i + 1].get(),
            outputBuffers[i].get(),
            nullptr,
            nullptr,
            0,
            nullptr,
            appSync.get(), /* Pointer to an array of fence points to reach or exceed before GPU operation starts. */
            (uint32_t)1, 
            ofaSync.get());
    }

    if (measureFPS)
    {
        ClientWaitSync(device, ofaSync->semaphore, ofaSync->value);
        executionTime += nvStopWatch.Stop();
    }
}

void EstimateFlow(VkInstance instance,
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    uint32_t queueFamilyIndex,
    uint32_t queueIndex,
    std::string inputFileName,
    std::string outputFileBaseName,
    NV_OF_PERF_LEVEL perfPreset,
    uint32_t gridSize,
    bool saveFlowAsImage,
    bool measureFPS
)
{
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
    bool     dumpOfOutput = (!outputFileBaseName.empty());
    std::unique_ptr<NvOFSyncVk> appSyncObj(new NvOFSyncVk(device));
    std::unique_ptr<NvOFSyncVk> ofaSyncObj(new NvOFSyncVk(device));
    NV_SYNC_VK appSync = appSyncObj->getVkSyncObject();
    NV_SYNC_VK ofaSync = ofaSyncObj->getVkSyncObject();

    NvOFObj nvOpticalFlow = NvOFVk::Create(instance, physicalDevice, device,
        width, height,
        dataLoader->GetBufferFormat(),
        NV_OF_MODE_OPTICALFLOW,
        perfPreset);

    uint32_t hwGridSize = gridSize;
    nvOpticalFlow->Init(hwGridSize, NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED, false, false);

    const uint32_t NUM_INPUT_BUFFERS = 2;
    const uint32_t NUM_OUTPUT_BUFFERS = NUM_INPUT_BUFFERS - 1;

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, &queue);

    VkCommandPoolCreateInfo cmdPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cmdPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

    VkCommandPool cmdPool;
    VK_API_CALL(vkCreateCommandPool(device, &cmdPoolCreateInfo, NULL, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    cmdAllocInfo.commandPool = cmdPool;

    VkCommandBuffer cmdBuffer;
    VK_API_CALL(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));

    VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    VK_API_CALL(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));
    std::unique_ptr<NvOFUtilsVulkan> nvOFUtils(new NvOFUtilsVulkan());

    auto lamdhaAllocateNvOFBuffers = [&](std::vector<NV_OF_RES_VK>& inputResourceList, NV_OF_BUFFER_DESCRIPTOR inputDesc, uint32_t hwGridSize = 1)
    {
        std::vector<NvOFBufferObj> buffers;
        for (auto it = inputResourceList.begin(); it != inputResourceList.end(); it++)
        {
            ofaSync->value++;

            VK_API_CALL(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
            VK_API_CALL(vkResetFences(device, 1, &fence));

            VK_API_CALL(vkResetCommandBuffer(cmdBuffer, 0));
            VkCommandBufferBeginInfo cmdBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            VK_API_CALL(vkBeginCommandBuffer(cmdBuffer, &cmdBeginInfo));

            nvOFUtils->TransferLayout(cmdBuffer, it->get()->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, queueFamilyIndex);

            VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            VkTimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
            VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, &timelineSemaphoreSubmitInfo };

            if (appSync->semaphore && appSync->value)
            {
                timelineSemaphoreSubmitInfo.waitSemaphoreValueCount = 1;
                timelineSemaphoreSubmitInfo.pWaitSemaphoreValues = &appSync->value;
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &appSync->semaphore;
                submitInfo.pWaitDstStageMask = &waitDstStageMask;
            }

            timelineSemaphoreSubmitInfo.signalSemaphoreValueCount = 1;
            timelineSemaphoreSubmitInfo.pSignalSemaphoreValues = &ofaSync->value;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &ofaSync->semaphore;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffer;

            VK_API_CALL(vkEndCommandBuffer(cmdBuffer));
            VK_API_CALL(vkQueueSubmit(queue, 1, &submitInfo, fence));
            buffers.emplace_back(std::move(nvOpticalFlow->RegisterPreAllocBuffers(inputDesc, it->get())));
        }
        return buffers;
    };

    auto lamdhaUploadData = [&](NvOFBufferObj& buffer, uint8_t* pOfData)
    {
        appSync->value++;
        buffer->UploadData(pOfData, ofaSync.get(), appSync.get());
    };

    auto lamdhaDeAllocateVkResources = [&](NV_OF_RESOURCE_VK* pResource)
    {
        NV_OF_RESOURCE_VK* pVkResource = pResource;
        vkDestroyImage(device, pVkResource->image, NULL);
        vkFreeMemory(device, pVkResource->memory, NULL);
        delete pVkResource;
        pVkResource = NULL;
    };

    NV_OF_BUFFER_DESCRIPTOR inputDesc;
    inputDesc.bufferFormat = dataLoader->GetBufferFormat();
    inputDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    inputDesc.width = width;
    inputDesc.height = height;

    std::vector<NV_OF_RES_VK> inputResourceList = AllocateVkResources(nvOpticalFlow, inputDesc, instance, physicalDevice, device, NUM_INPUT_BUFFERS, lamdhaDeAllocateVkResources, hwGridSize);
    std::vector<NvOFBufferObj> inputBuffers = lamdhaAllocateNvOFBuffers(inputResourceList, inputDesc);
    NV_OF_BUFFER_DESCRIPTOR outputDesc;
    auto nOutWidth = (width + hwGridSize - 1) / hwGridSize;
    auto nOutHeight = (height + hwGridSize - 1) / hwGridSize;

    outputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    outputDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    outputDesc.width = nOutWidth;
    outputDesc.height = nOutHeight;
    std::vector<NV_OF_RES_VK> outputResourceList = AllocateVkResources(nvOpticalFlow, outputDesc, instance, physicalDevice, device, NUM_OUTPUT_BUFFERS, lamdhaDeAllocateVkResources, hwGridSize);
    std::vector<NvOFBufferObj> outputBuffers = lamdhaAllocateNvOFBuffers(outputResourceList, outputDesc);

    std::unique_ptr<NV_OF_FLOW_VECTOR[]> pOut;
    std::unique_ptr<NvOFFileWriter> flowFileWriter;

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

    uint32_t                        curFrameIdx = 0;
    uint32_t                        frameCount = 0;
    bool                            lastSet = false;
    double                          executionTime = 0;

    for (; (!dataLoader->IsDone() || curFrameIdx > 1); dataLoader->Next())
    {
        if (!dataLoader->IsDone())
        {
            lamdhaUploadData(inputBuffers[curFrameIdx], dataLoader->CurrentItem());
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
            NvOFBatchExecute(nvOpticalFlow, inputBuffers, outputBuffers, curFrameIdx, executionTime, measureFPS, device, appSync, ofaSync);
            if (dumpOfOutput)
            {
                for (uint32_t i = 0; i < curFrameIdx; i++)
                {
                    outputBuffers[i]->DownloadData(pOut.get(), ofaSync.get());
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

    vkDeviceWaitIdle(device);

    vkDestroyFence(device, fence, NULL);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
    vkDestroyCommandPool(device, cmdPool, NULL);
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

    try
    {
        NvOFCmdParser cmdParser;
        cmdParser.AddOptions("input", inputFileName, "Input filename "
            "[ e.g. inputDir" DIR_SEP "input*.png, "
            "inputDir" DIR_SEP "input%d.png, "
            "inputDir" DIR_SEP "input_wxh.yuv ]");
        cmdParser.AddOptions("output", outputFileBaseName, "Output file base name "
            "[ e.g. outputDir" DIR_SEP "outFilename ]");
        cmdParser.AddOptions("gpuIndex", gpuId, "Vulkan adapter ordinal");
        cmdParser.AddOptions("preset", preset, "perf preset for OF algo [ options : slow, medium, fast]");
        cmdParser.AddOptions("gridSize", gridSize, "Block size per motion vector");
        cmdParser.AddOptions("visualFlow", visualFlow, "save flow vectors as RGB image");
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

        const char* required_instance_layers[] = {
#if defined(_DEBUG)
           "VK_LAYER_KHRONOS_validation",
#endif
            NULL
        };

        uint32_t required_instance_layer_count = 0;
        uint32_t enabled_instance_layer_count = 0;
        uint32_t instance_layer_count = 0;
        VK_API_CALL(vkEnumerateInstanceLayerProperties(&instance_layer_count, NULL));

        std::unique_ptr<VkLayerProperties[]> instance_layers(new VkLayerProperties[instance_layer_count]);

        VK_API_CALL(vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layers.get()));

        for (uint32_t i = 0; required_instance_layers[i]; i++) {
            for (uint32_t j = 0; j < instance_layer_count; j++) {
                if (!strcmp(required_instance_layers[i], instance_layers[j].layerName)) {
                    enabled_instance_layer_count++;
                    break;
                }
            }
            required_instance_layer_count++;
        }

        if (enabled_instance_layer_count < required_instance_layer_count) {
            std::cerr << "Some instance layers are missing." << std::endl;
            return 1;
        }

        const char* required_instance_extensions[] = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            NULL
        };

        uint32_t required_instance_extension_count = 0;
        uint32_t enabled_instance_extension_count = 0;
        uint32_t instance_extension_count = 0;
        VK_API_CALL(vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, NULL));

        std::unique_ptr<VkExtensionProperties[]> instance_extensions(new VkExtensionProperties[instance_extension_count]);

        VK_API_CALL(vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, instance_extensions.get()));

        for (uint32_t i = 0; required_instance_extensions[i]; i++) {
            for (uint32_t j = 0; j < instance_extension_count; j++) {
                if (!strcmp(required_instance_extensions[i], instance_extensions[j].extensionName)) {
                    enabled_instance_extension_count++;
                    break;
                }
            }
            required_instance_extension_count++;
        }

        if (enabled_instance_extension_count < required_instance_extension_count) {
            std::cerr << "Some instance extensions are missing." << std::endl;
            return 1;
        }

        VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        applicationInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        instanceCreateInfo.enabledLayerCount = enabled_instance_layer_count;
        instanceCreateInfo.ppEnabledLayerNames = enabled_instance_layer_count ? required_instance_layers : NULL;
        instanceCreateInfo.enabledExtensionCount = enabled_instance_extension_count;
        instanceCreateInfo.ppEnabledExtensionNames = enabled_instance_extension_count ? required_instance_extensions : NULL;

        size_t size = 0;
        NvOFUtilsVulkan::EnableInstanceOpticalFlow(&instanceCreateInfo, &size, NULL);
        std::unique_ptr<uint8_t[]> pInstanceCreateInfoOF(new uint8_t[size]);
        NvOFUtilsVulkan::EnableInstanceOpticalFlow(&instanceCreateInfo, &size, reinterpret_cast<VkInstanceCreateInfo*>(pInstanceCreateInfoOF.get()));

        VkInstance instance = VK_NULL_HANDLE;
        VK_API_CALL(vkCreateInstance(reinterpret_cast<VkInstanceCreateInfo*>(pInstanceCreateInfoOF.get()), NULL, &instance));

        uint32_t deviceCount = 0;
        VK_API_CALL(vkEnumeratePhysicalDevices(instance, &deviceCount, NULL));

        std::unique_ptr<VkPhysicalDevice[]> physicalDevices(new VkPhysicalDevice[deviceCount]);

        if (uint32_t(gpuId) >= deviceCount) {
            std::cerr << "Adapter " << gpuId << " not found." << std::endl;
            return 1;
        }

        VK_API_CALL(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.get()));

        VkPhysicalDevice physicalDevice = physicalDevices[gpuId];

        VkPhysicalDeviceProperties2 physicalDeviceProperties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties2);

        std::cout << "GPU in use: " << physicalDeviceProperties2.properties.deviceName << std::endl;

        const char* required_device_extensions[] = {
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            NULL
        };

        uint32_t required_device_extension_count = 0;
        uint32_t enabled_device_extension_count = 0;
        uint32_t device_extension_count = 0;
        VK_API_CALL(vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &device_extension_count, NULL));

        std::unique_ptr<VkExtensionProperties[]> device_extensions(new VkExtensionProperties[device_extension_count]);

        VK_API_CALL(vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &device_extension_count, device_extensions.get()));

        for (uint32_t i = 0; required_device_extensions[i]; i++) {
            for (uint32_t j = 0; j < device_extension_count; j++) {
                if (!strcmp(required_device_extensions[i], device_extensions[j].extensionName)) {
                    enabled_device_extension_count++;
                    break;
                }
            }
            required_device_extension_count++;
        }

        if (enabled_device_extension_count < required_device_extension_count) {
            std::cerr << "Some device extensions are missing." << std::endl;
            return 1;
        }

        float queuePriorities[] = { 1.f };
        uint32_t queueIndex = 0;
        VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueCreateInfo.pQueuePriorities = queuePriorities;
        queueCreateInfo.queueCount = 1;

        if (!NvOFUtilsVulkan::GetAppQueue(physicalDevice, queueCreateInfo.queueFamilyIndex, queueIndex, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) {
            std::cerr << "No suitable queue available." << std::endl;
        }
        VkPhysicalDeviceTimelineSemaphoreFeatures physicalDeviceTimelineSemaphoreFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
        VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &physicalDeviceTimelineSemaphoreFeatures };

        vkGetPhysicalDeviceFeatures2(physicalDevice, &physicalDeviceFeatures2);

        if (!physicalDeviceTimelineSemaphoreFeatures.timelineSemaphore) {
            std::cerr << "No timeline semaphore support." << std::endl;
            return 1;
        }

        VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &physicalDeviceTimelineSemaphoreFeatures };
        deviceCreateInfo.enabledExtensionCount = enabled_device_extension_count;
        deviceCreateInfo.ppEnabledExtensionNames = required_device_extensions;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        size = 0;
        NvOFUtilsVulkan::EnableDeviceOpticalFlow(instance, physicalDevice, &deviceCreateInfo, &size, NULL);
        std::unique_ptr<uint8_t[]> pDeviceCreateInfoOF(new uint8_t[size]);
        NvOFUtilsVulkan::EnableDeviceOpticalFlow(instance, physicalDevice, &deviceCreateInfo, &size, reinterpret_cast<VkDeviceCreateInfo*>(pDeviceCreateInfoOF.get()));

        VkDevice device = VK_NULL_HANDLE;
        VK_API_CALL(vkCreateDevice(physicalDevice, reinterpret_cast<const VkDeviceCreateInfo*>(pDeviceCreateInfoOF.get()), NULL, &device));

        EstimateFlow(instance, physicalDevice, device, queueCreateInfo.queueFamilyIndex, queueIndex,
                     inputFileName, outputFileBaseName, perfPreset, gridSize, !!visualFlow, !!measureFPS);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
