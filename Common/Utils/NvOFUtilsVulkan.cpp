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


#include <assert.h> // TBD VK OFA (remove if not required anymore)
#include "NvOFVulkan.h"
#include "NvOFUtilsVulkan.h"

using namespace GNvOFUtilsVulkan;

NV_OF_BUFFER_FORMAT GNvOFUtilsVulkan::VkFormatToNvOFBufferFormat(VkFormat format)
{
    NV_OF_BUFFER_FORMAT ofBufFormat;
    switch (format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
        ofBufFormat = NV_OF_BUFFER_FORMAT_ABGR8;
        break;
    case VK_FORMAT_R16G16_S10_5_NV:
        ofBufFormat = NV_OF_BUFFER_FORMAT_SHORT2;
        break;
    case VK_FORMAT_R32_UINT:
        ofBufFormat = NV_OF_BUFFER_FORMAT_UINT;
        break;
    case VK_FORMAT_R16_UINT:
        ofBufFormat = NV_OF_BUFFER_FORMAT_SHORT;
        break;
    case VK_FORMAT_R8_UINT:
        ofBufFormat = NV_OF_BUFFER_FORMAT_UINT8;
        break;
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
        ofBufFormat = NV_OF_BUFFER_FORMAT_NV12;
        break;
    default:
        ofBufFormat = NV_OF_BUFFER_FORMAT_UNDEFINED;
    }
    return ofBufFormat;
}


VkFormat GNvOFUtilsVulkan::NvOFBufferFormatToVkFormat(NV_OF_BUFFER_FORMAT  ofBufFormat)
{
    VkFormat format;
    switch (ofBufFormat)
    {
    case NV_OF_BUFFER_FORMAT_ABGR8:
        format = VK_FORMAT_B8G8R8A8_UNORM;
        break;
    case NV_OF_BUFFER_FORMAT_SHORT2:
        format =  VK_FORMAT_R16G16_S10_5_NV;
        break;
    case NV_OF_BUFFER_FORMAT_UINT:
        format = VK_FORMAT_R32_UINT;
        break;
    case NV_OF_BUFFER_FORMAT_UINT8:
        format = VK_FORMAT_R8_UINT;
        break;
    case NV_OF_BUFFER_FORMAT_GRAYSCALE8:
        format = VK_FORMAT_R8_UNORM;
        break;
    case NV_OF_BUFFER_FORMAT_NV12:
        format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        break;
    default:
        format = VK_FORMAT_UNDEFINED;
    }
    return format;
}

uint32_t GNvOFUtilsVulkan::GetNumberOfPlanes(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_R16G16_S10_5_NV:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
        return 1;
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
        return 2;
    default:
        NVOF_THROW_ERROR("Invalid buffer format", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }

    return 0;
}

VkOpticalFlowUsageFlagsNV GNvOFUtilsVulkan::GetVkOpticalFlowUsageFlag(const NV_OF_BUFFER_USAGE bufUsage)
{
    VkOpticalFlowUsageFlagsNV usage = VK_OPTICAL_FLOW_USAGE_UNKNOWN_NV;

    switch (bufUsage) {
    case NV_OF_BUFFER_USAGE_INPUT:
        usage = VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_OUTPUT:
        usage = VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_HINT:
        usage = VK_OPTICAL_FLOW_USAGE_HINT_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_COST:
        usage = VK_OPTICAL_FLOW_USAGE_COST_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_GLOBAL_FLOW:
        usage = VK_OPTICAL_FLOW_USAGE_GLOBAL_FLOW_BIT_NV;
        break;
    default:
        NVOF_THROW_ERROR("Invalid value for bufUsage.", NV_OF_ERR_INVALID_PARAM);
    }
    return usage;
}

VkFormatFeatureFlags2 GNvOFUtilsVulkan::GetVkFormatFeatureFlags(const NV_OF_BUFFER_USAGE bufUsage)
{
    VkFormatFeatureFlags2 feature = 0;

    switch (bufUsage) {
    case NV_OF_BUFFER_USAGE_INPUT:
        feature = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_IMAGE_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_OUTPUT:
        feature = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_HINT:
        feature = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_COST:
        feature = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_COST_BIT_NV;
        break;
    case NV_OF_BUFFER_USAGE_GLOBAL_FLOW:
        feature = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV;
        break;
    default:
        NVOF_THROW_ERROR("Invalid value for bufUsage.", NV_OF_ERR_INVALID_PARAM);
    }
    return feature;
}

void NvOFUtilsVulkan::TransferLayout(VkCommandBuffer cmdBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t queueFamilyIndex)
{
    if (oldLayout != newLayout) {
        VkImageSubresourceRange subresourceRange = { };
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        VkImageMemoryBarrier imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        imageBarrier.srcAccessMask = getAccessMask(oldLayout);
        imageBarrier.dstAccessMask = getAccessMask(newLayout);
        imageBarrier.oldLayout = oldLayout;
        imageBarrier.newLayout = newLayout;
        imageBarrier.srcQueueFamilyIndex = queueFamilyIndex;
        imageBarrier.dstQueueFamilyIndex = queueFamilyIndex;
        imageBarrier.image = image;
        imageBarrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(cmdBuffer, getStageMask(oldLayout), getStageMask(newLayout), 0, 0, 0, 0, 0, 1, &imageBarrier);
    }
}

VkPipelineStageFlags NvOFUtilsVulkan::getStageMask(VkImageLayout layout)
{

    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        return VK_PIPELINE_STAGE_NONE;
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        return VK_PIPELINE_STAGE_HOST_BIT;
    case VK_IMAGE_LAYOUT_GENERAL:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    default:
        NVOF_THROW_ERROR("Unhandled vulkan image layout", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }
    return (VkPipelineStageFlags)0;
}

VkAccessFlags
NvOFUtilsVulkan::getAccessMask(VkImageLayout layout)
{
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        return 0;
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        return VK_ACCESS_HOST_WRITE_BIT;
    case VK_IMAGE_LAYOUT_GENERAL:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return VK_ACCESS_TRANSFER_READ_BIT;
    default:
        NVOF_THROW_ERROR("Unhandled vulkan image layout", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }
    return 0;
}

void NvOFUtilsVulkan::EnableInstanceOpticalFlow(VkInstanceCreateInfo* const pInstanceCreateInfo, size_t* pSize, VkInstanceCreateInfo* pInstanceCreateInfoOF)
{
    if (!pSize)
    {
        NVOF_THROW_ERROR("pSize must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (!pInstanceCreateInfo || (pInstanceCreateInfo->sType != VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO))
    {
        NVOF_THROW_ERROR("pInstanceCreateInfo must be pointer to valid VkInstanceCreateInfo chain", NV_OF_ERR_INVALID_PARAM);
    }

    size_t requiredSize = CREATE_INFO_ALIGN(sizeof(VkInstanceCreateInfo));
    if (!pInstanceCreateInfo->pApplicationInfo || (pInstanceCreateInfo->pApplicationInfo->apiVersion < VK_API_VERSION_1_3)) {
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkApplicationInfo));
    }

    if (!pInstanceCreateInfoOF)
    {
        *pSize = requiredSize;
        return;
    }

    if (*pSize < requiredSize)
    {
        NVOF_THROW_ERROR("Size of output buffer too small.", NV_OF_ERR_INVALID_PARAM);
    }

    *pInstanceCreateInfoOF = *pInstanceCreateInfo;

    size_t size = CREATE_INFO_ALIGN(sizeof(VkInstanceCreateInfo));
    void* pNextFree = (void*)((uintptr_t)pInstanceCreateInfoOF + size);

    if (!pInstanceCreateInfo->pApplicationInfo || (pInstanceCreateInfo->pApplicationInfo->apiVersion < VK_API_VERSION_1_3)) {
        VkApplicationInfo* pApplicationInfo = (VkApplicationInfo*)pNextFree;

        if (pInstanceCreateInfo->pApplicationInfo) {
            *pApplicationInfo = *pInstanceCreateInfo->pApplicationInfo;
        }
        else {
            memset(pApplicationInfo, 0, sizeof(VkApplicationInfo));
            pApplicationInfo->sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        }
        pApplicationInfo->apiVersion = VK_API_VERSION_1_3;

        pInstanceCreateInfoOF->pApplicationInfo = pApplicationInfo;

        pNextFree = (void*)((uintptr_t)pApplicationInfo + CREATE_INFO_ALIGN(sizeof(VkApplicationInfo)));
    }
}

void NvOFUtilsVulkan::EnableDeviceOpticalFlow(VkInstance instance, VkPhysicalDevice physicalDevice, VkDeviceCreateInfo* const pDeviceCreateInfo, size_t* pSize, VkDeviceCreateInfo* pDeviceCreateInfoOF)
{
    if (!pSize)
    {
        NVOF_THROW_ERROR("pSize must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (instance == VK_NULL_HANDLE)
    {
        NVOF_THROW_ERROR("instance must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        NVOF_THROW_ERROR("physicalDevice must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (!pDeviceCreateInfo || (pDeviceCreateInfo->sType != VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO))
    {
        NVOF_THROW_ERROR("pDeviceCreateInfo must be pointer to valid VkDeviceCreateInfo chain", NV_OF_ERR_INVALID_PARAM);
    }

    struct VkExtension {
        const char* name;
        bool enabled;
    } requiredExtensions[] = {
        { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, false },
        { VK_NV_OPTICAL_FLOW_EXTENSION_NAME, false },
        {VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME , false},
    };

    // check if all required extension are supported
    uint32_t enabledExtensions = 0;
    uint32_t numExtensions = 0;
    VK_API_CALL(vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &numExtensions, NULL));

    std::unique_ptr<VkExtensionProperties[]> extensions(new VkExtensionProperties[numExtensions]);

    VK_API_CALL(vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &numExtensions, extensions.get()));

    for (uint32_t i = 0; i < ARRAYSIZE(requiredExtensions); i++) {
        for (uint32_t j = 0; j < numExtensions; j++) {
            if (!strcmp(requiredExtensions[i].name, extensions[j].extensionName)) {
                enabledExtensions++;
                break;
            }
        }
    }

    if (enabledExtensions < ARRAYSIZE(requiredExtensions)) {
        NVOF_THROW_ERROR("Some required vulkan device extensions are missing.", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }

    // check if any of the required extensions is already on the list
    uint32_t missingExtensions = ARRAYSIZE(requiredExtensions);
    for (uint32_t i = 0; i < ARRAYSIZE(requiredExtensions); i++) {
        for (uint32_t j = 0; j < pDeviceCreateInfo->enabledExtensionCount; j++) {
            if (!strcmp(requiredExtensions[i].name, pDeviceCreateInfo->ppEnabledExtensionNames[j])) {
                requiredExtensions[i].enabled = true;
                missingExtensions--;
                break;
            }
        }
    }

    size_t requiredSize = CREATE_INFO_ALIGN(sizeof(VkDeviceCreateInfo));
    if (missingExtensions) {
        requiredSize += (pDeviceCreateInfo->enabledExtensionCount + missingExtensions) * sizeof(void*);
    }

    VkPhysicalDeviceOpticalFlowFeaturesNV physicalDeviceOpticalFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV };
    VkPhysicalDeviceSynchronization2Features physicalDeviceSynchronization2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, &physicalDeviceOpticalFlowFeatures };
    VkPhysicalDeviceTimelineSemaphoreFeatures physicalDeviceTimelineSemaphoreFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, &physicalDeviceSynchronization2Features };
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &physicalDeviceTimelineSemaphoreFeatures };

    vkGetPhysicalDeviceFeatures2(physicalDevice, &physicalDeviceFeatures2);

    if (!physicalDeviceTimelineSemaphoreFeatures.timelineSemaphore) {
        NVOF_THROW_ERROR("Vulkan timeline semaphores are not supported.", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }

    if (!physicalDeviceSynchronization2Features.synchronization2 ||
        !physicalDeviceOpticalFlowFeatures.opticalFlow) {
        NVOF_THROW_ERROR("Not all required features are supported.", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }
    bool synchronization2Feature = false;
    bool opticalFlowFeature = false;
    bool timelineSemaphoreFeature = false;
    const VkBaseInStructure* pCreateInfo = (const VkBaseInStructure*)pDeviceCreateInfo;
    while (pCreateInfo->pNext) {
        pCreateInfo = (const VkBaseInStructure*)pCreateInfo->pNext;
        switch (pCreateInfo->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:
            if (!(((VkPhysicalDeviceTimelineSemaphoreFeatures*)pCreateInfo)->timelineSemaphore)) {
                NVOF_THROW_ERROR("Vulkan timeline semaphores feature not enabled in VkDeviceCreateInfo chain.", NV_OF_ERR_INVALID_PARAM);
            }
            timelineSemaphoreFeature = true;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES:
            if (!(((VkPhysicalDeviceSynchronization2Features*)pCreateInfo)->synchronization2)) {
                NVOF_THROW_ERROR("Vulkan synchronization2 feature not supported.", NV_OF_ERR_UNSUPPORTED_FEATURE);
            }
            synchronization2Feature = true;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV:
            if (!(((VkPhysicalDeviceOpticalFlowFeaturesNV*)pCreateInfo)->opticalFlow)) {
                NVOF_THROW_ERROR("Vulkan optical flow not supported.", NV_OF_ERR_UNSUPPORTED_FEATURE);
            }
            opticalFlowFeature = true;
            break;
        default:
            ;
        }
    }

    if (!timelineSemaphoreFeature) {
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceTimelineSemaphoreFeatures));
    }
    if (!synchronization2Feature) {
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceSynchronization2Features));
    }
    if (!opticalFlowFeature) {
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceOpticalFlowFeaturesNV));
    }

    // check for queues
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    std::unique_ptr<VkQueueFamilyProperties[]> queueProps(new VkQueueFamilyProperties[queueFamilyCount]);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueProps.get());

    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
    if (!NvOFUtilsVulkan::GetOpticalFlowQueue(physicalDevice, queueFamilyIndex, queueIndex)) {
        NVOF_THROW_ERROR("No suitable vulkan queue available.", NV_OF_ERR_UNSUPPORTED_FEATURE);
    }

    const VkDeviceQueueCreateInfo* pQueueCreateInfo = NULL;
    for (uint32_t i = 0; i < pDeviceCreateInfo->queueCreateInfoCount; i++) {
        if (pDeviceCreateInfo->pQueueCreateInfos[i].queueFamilyIndex == queueFamilyIndex) {
            pQueueCreateInfo = &pDeviceCreateInfo->pQueueCreateInfos[i];
            break;
        }
    }

    if (!pQueueCreateInfo)
    {
        requiredSize += CREATE_INFO_ALIGN((pDeviceCreateInfo->queueCreateInfoCount + 1) * sizeof(VkDeviceQueueCreateInfo));
        requiredSize += CREATE_INFO_ALIGN(sizeof(float));
    }

    if (!pDeviceCreateInfoOF)
    {
        *pSize = requiredSize;
        return;
    }

    if (*pSize < requiredSize)
    {
        NVOF_THROW_ERROR("Size of output buffer too small.", NV_OF_ERR_INVALID_PARAM);
    }

    *pDeviceCreateInfoOF = *pDeviceCreateInfo;

    VkBaseInStructure* pCreateInfoOF = (VkBaseInStructure*)pDeviceCreateInfoOF;
    while (pCreateInfoOF->pNext) {
        pCreateInfoOF = (VkBaseInStructure*)pCreateInfoOF->pNext;
    }

    size_t size = CREATE_INFO_ALIGN(sizeof(VkDeviceCreateInfo));
    void* pNextFree = (void*)((uintptr_t)pDeviceCreateInfoOF + size);

    if (missingExtensions) {
        const char** ppEnabledExtensionNames = (const char**)pNextFree;

        pDeviceCreateInfoOF->enabledExtensionCount += missingExtensions;
        pDeviceCreateInfoOF->ppEnabledExtensionNames = ppEnabledExtensionNames;

        memcpy(ppEnabledExtensionNames, pDeviceCreateInfo->ppEnabledExtensionNames, pDeviceCreateInfo->enabledExtensionCount * sizeof(const char*));

        ppEnabledExtensionNames += pDeviceCreateInfo->enabledExtensionCount;
        for (uint32_t i = 0; i < ARRAYSIZE(requiredExtensions); i++) {
            if (!requiredExtensions[i].enabled) {
                *ppEnabledExtensionNames++ = requiredExtensions[i].name;
            }
        }

        pNextFree = ppEnabledExtensionNames;
    }

    if (!timelineSemaphoreFeature)
    {
        VkPhysicalDeviceTimelineSemaphoreFeatures* pPhysicalDeviceTimelineSemaphoreFeatures = (VkPhysicalDeviceTimelineSemaphoreFeatures*)pNextFree;

        pPhysicalDeviceTimelineSemaphoreFeatures->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        pPhysicalDeviceTimelineSemaphoreFeatures->pNext = NULL;
        pPhysicalDeviceTimelineSemaphoreFeatures->timelineSemaphore = VK_TRUE;

        pCreateInfoOF->pNext = (const VkBaseInStructure*)pPhysicalDeviceTimelineSemaphoreFeatures;
        pCreateInfoOF = (VkBaseInStructure*)pPhysicalDeviceTimelineSemaphoreFeatures;

        pNextFree = (void*)((uintptr_t)pPhysicalDeviceTimelineSemaphoreFeatures + CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceTimelineSemaphoreFeatures)));
    }

    if (!synchronization2Feature)
    {
        VkPhysicalDeviceSynchronization2Features* pPhysicalDeviceSynchronization2Features = (VkPhysicalDeviceSynchronization2Features*)pNextFree;

        pPhysicalDeviceSynchronization2Features->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        pPhysicalDeviceSynchronization2Features->pNext = NULL;
        pPhysicalDeviceSynchronization2Features->synchronization2 = VK_TRUE;

        pCreateInfoOF->pNext = (const VkBaseInStructure*)pPhysicalDeviceSynchronization2Features;
        pCreateInfoOF = (VkBaseInStructure*)pPhysicalDeviceSynchronization2Features;

        pNextFree = (void*)((uintptr_t)pPhysicalDeviceSynchronization2Features + CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceSynchronization2Features)));
    }

    if (!opticalFlowFeature)
    {
        VkPhysicalDeviceOpticalFlowFeaturesNV* pPhysicalDeviceOpticalFlowFeatures = (VkPhysicalDeviceOpticalFlowFeaturesNV*)pNextFree;

        pPhysicalDeviceOpticalFlowFeatures->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
        pPhysicalDeviceOpticalFlowFeatures->pNext = NULL;
        pPhysicalDeviceOpticalFlowFeatures->opticalFlow = physicalDeviceOpticalFlowFeatures.opticalFlow;

        pCreateInfoOF->pNext = (const VkBaseInStructure*)pPhysicalDeviceOpticalFlowFeatures;
        pCreateInfoOF = (VkBaseInStructure*)pPhysicalDeviceOpticalFlowFeatures;

        pNextFree = (void*)((uintptr_t)pPhysicalDeviceOpticalFlowFeatures + CREATE_INFO_ALIGN(sizeof(VkPhysicalDeviceOpticalFlowFeaturesNV)));
    }

    if (!pQueueCreateInfo)
    {
        VkDeviceQueueCreateInfo* pQueueCreateInfos = (VkDeviceQueueCreateInfo*)pNextFree;
        pNextFree = (void*)((uintptr_t)pQueueCreateInfos + CREATE_INFO_ALIGN((pDeviceCreateInfo->queueCreateInfoCount + 1) * sizeof(VkDeviceQueueCreateInfo)));

        memcpy(pQueueCreateInfos, pDeviceCreateInfo->pQueueCreateInfos, pDeviceCreateInfo->queueCreateInfoCount * sizeof(VkDeviceQueueCreateInfo));

        pDeviceCreateInfoOF->queueCreateInfoCount++;
        pDeviceCreateInfoOF->pQueueCreateInfos = pQueueCreateInfos;

        pQueueCreateInfos += pDeviceCreateInfo->queueCreateInfoCount;

        float* pQueuePriorities = (float*)pNextFree;
        pNextFree = (void*)((uintptr_t)pQueuePriorities + CREATE_INFO_ALIGN(sizeof(float)));

        *pQueuePriorities = 1.f;

        pQueueCreateInfos->sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        pQueueCreateInfos->pNext = NULL;
        pQueueCreateInfos->flags = 0;
        pQueueCreateInfos->queueFamilyIndex = queueFamilyIndex;
        pQueueCreateInfos->queueCount = 1;
        pQueueCreateInfos->pQueuePriorities = pQueuePriorities;
    }
}

void NvOFUtilsVulkan::EnableResourceOpticalFlow(VkInstance instance, VkPhysicalDevice physicalDevice, const NV_OF_BUFFER_USAGE bufUsage, const NV_OF_MODE ofMode, void* const pCreateInfo, size_t* pSize, void* pCreateInfoOF)
{
    if (!pSize)
    {
        NVOF_THROW_ERROR("pSize must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (instance == VK_NULL_HANDLE)
    {
        NVOF_THROW_ERROR("instance must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        NVOF_THROW_ERROR("physicalDevice must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    if (!pCreateInfo)
    {
        NVOF_THROW_ERROR("pDeviceCreateInfo must not be NULL", NV_OF_ERR_INVALID_PARAM);
    }

    VkOpticalFlowUsageFlagsNV usage = GetVkOpticalFlowUsageFlag(bufUsage);
    VkFormatFeatureFlags2 feature = GetVkFormatFeatureFlags(bufUsage);

    const VkBaseInStructure* pBaseCreateInfo = (const VkBaseInStructure*)pCreateInfo;

    size_t requiredSize = 0;
    if (pBaseCreateInfo->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
    {
        const VkImageCreateInfo* pImageCreateInfo = (const VkImageCreateInfo*)pCreateInfo;
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkImageCreateInfo));

        VkOpticalFlowImageFormatInfoNV opticalFlowImageFormatInfo = { VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV };
        opticalFlowImageFormatInfo.usage = usage;

        VkFormatProperties3 formatProperties3 = { VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
        VkFormatProperties2 formatProperties2 = { VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &formatProperties3 };
        vkGetPhysicalDeviceFormatProperties2(physicalDevice, pImageCreateInfo->format, &formatProperties2);
        if ((pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL) && !(formatProperties3.optimalTilingFeatures & feature)) {
            NVOF_THROW_ERROR("Optimal tiling not supported for requested format.", NV_OF_ERR_INVALID_PARAM);
        }
        if ((pImageCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) && !(formatProperties3.linearTilingFeatures & feature)) {
            NVOF_THROW_ERROR("Optimal tiling not supported for requested format.", NV_OF_ERR_INVALID_PARAM);
        }

        VkPhysicalDeviceImageFormatInfo2 imageFormatInfo2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &opticalFlowImageFormatInfo };
        imageFormatInfo2.format = pImageCreateInfo->format;
        imageFormatInfo2.type = pImageCreateInfo->imageType;
        imageFormatInfo2.tiling = pImageCreateInfo->tiling;
        imageFormatInfo2.usage = pImageCreateInfo->usage | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageFormatInfo2.flags = pImageCreateInfo->flags;

        VkImageFormatProperties2 imageFormatProperties2 = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };

        if (vkGetPhysicalDeviceImageFormatProperties2(physicalDevice, &imageFormatInfo2, &imageFormatProperties2) != VK_SUCCESS) {
            NVOF_THROW_ERROR("Optical flow not supported for image create parameters.", NV_OF_ERR_INVALID_PARAM);
        }

        requiredSize += CREATE_INFO_ALIGN(sizeof(VkOpticalFlowImageFormatInfoNV));
    }
    else if (pBaseCreateInfo->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
    {
        requiredSize += CREATE_INFO_ALIGN(sizeof(VkMemoryAllocateInfo));
    }
    else
    {
        NVOF_THROW_ERROR("Invalid CreateInfo chain.", NV_OF_ERR_INVALID_PARAM);
    }

    if (!pCreateInfoOF)
    {
        *pSize = requiredSize;
        return;
    }

    if (*pSize < requiredSize)
    {
        NVOF_THROW_ERROR("Size of output buffer too small.", NV_OF_ERR_INVALID_PARAM);
    }

    if (pBaseCreateInfo->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
    {
        const VkImageCreateInfo* pImageCreateInfo = (const VkImageCreateInfo*)pCreateInfo;
        VkImageCreateInfo* pImageCreateInfoOF = (VkImageCreateInfo*)pCreateInfoOF;
        *pImageCreateInfoOF = *pImageCreateInfo;

        VkBaseInStructure* pBaseInfoOF = (VkBaseInStructure*)pImageCreateInfoOF;
        while (pBaseInfoOF->pNext) {
            pBaseInfoOF = (VkBaseInStructure*)pBaseInfoOF->pNext;
        }

        size_t size = CREATE_INFO_ALIGN(sizeof(VkImageCreateInfo));
        void* pNextFree = (void*)((uintptr_t)pImageCreateInfoOF + size);

        pImageCreateInfoOF->usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (pImageCreateInfoOF->format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
            pImageCreateInfoOF->flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        }

        VkOpticalFlowImageFormatInfoNV* pOpticalFlowImageFormatInfo = (VkOpticalFlowImageFormatInfoNV*)pNextFree;
        pNextFree = (void*)((uintptr_t)pOpticalFlowImageFormatInfo + CREATE_INFO_ALIGN(sizeof(VkOpticalFlowImageFormatInfoNV)));

        pOpticalFlowImageFormatInfo->sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV;
        pOpticalFlowImageFormatInfo->pNext = NULL;
        pOpticalFlowImageFormatInfo->usage = GetVkOpticalFlowUsageFlag(bufUsage);
        pBaseInfoOF->pNext = (const VkBaseInStructure*)pOpticalFlowImageFormatInfo;
        pBaseInfoOF = (VkBaseInStructure*)pOpticalFlowImageFormatInfo;
    }
    else
    {
        const VkMemoryAllocateInfo* pMemoryAllocateInfo = (const VkMemoryAllocateInfo*)pCreateInfo;
        VkMemoryAllocateInfo* pMemoryAllocateInfoOF = (VkMemoryAllocateInfo*)pCreateInfoOF;
        *pMemoryAllocateInfoOF = *pMemoryAllocateInfo;
    }
}

bool NvOFUtilsVulkan::GetOpticalFlowQueue(VkPhysicalDevice physicalDevice, uint32_t& queueFamilyIndex, uint32_t& queueIndex)
{
    queueFamilyIndex = 0;
    queueIndex = 0;

    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    std::unique_ptr<VkQueueFamilyProperties[]> queueProps(new VkQueueFamilyProperties[queueFamilyCount]);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueProps.get());

    // OFA is running on first queue family which supports optical flow with queue index 0
    // (in case optical flow is not supported on vulkan, OFA is running on graphics queue with queue index 0)
    VkQueueFlags requiredCaps = VK_QUEUE_OPTICAL_FLOW_BIT_NV;

    for (queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++) {
        if ((queueProps[queueFamilyIndex].queueFlags & requiredCaps) == requiredCaps) {
            break;
        }
    }

    if (queueFamilyIndex == queueFamilyCount) {
        return false;
    }

    return true;
}

bool NvOFUtilsVulkan::GetAppQueue(VkPhysicalDevice physicalDevice, uint32_t& queueFamilyIndex, uint32_t& queueIndex, VkQueueFlags requiredCaps, VkQueueFlags disallowCaps)
{
    uint32_t opticalFlowQueueFamilyIndex;
    uint32_t opticalFlowQueueIndex;
    if (!NvOFUtilsVulkan::GetOpticalFlowQueue(physicalDevice, opticalFlowQueueFamilyIndex, opticalFlowQueueIndex)) {
        return false;
    }

    queueIndex = 0;

    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    std::unique_ptr<VkQueueFamilyProperties[]> queueProps(new VkQueueFamilyProperties[queueFamilyCount]);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueProps.get());

    for (queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++) {
        if (((queueProps[queueFamilyIndex].queueFlags & requiredCaps) == requiredCaps) &&
            !(queueProps[queueFamilyIndex].queueFlags & disallowCaps)) {
            break;
        }
    }

    if (queueFamilyIndex == queueFamilyCount) {
       return false;
    }

    //Native OFA always runs on the very first queue of the very first optical flow-capable queue family.
    if (queueFamilyIndex == opticalFlowQueueFamilyIndex) {
        queueIndex = opticalFlowQueueIndex ? 0 : 1;
    } else {
        queueIndex = 0;
    }

    return true;
}

