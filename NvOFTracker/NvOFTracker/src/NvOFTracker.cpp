/*
* Copyright 2019-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "NvOFTracker.h"
#include "CNvOFTracker.h"
#include "CNvOFTrackerException.h"

#include "opencv2/core.hpp"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

NvOFT_STATUS NvOFTCreate(const NvOFT_CREATE_PARAMS* createParams, NvOFTrackerHandle* hOFT)
{
    *hOFT = NULL;
    NvOFT_STATUS status = NvOFT_SUCCESS;
    try
    {
        if (createParams == NULL)
        {
            NVOFTRACKER_THROW_ERROR("Invalid NvOFT_CREATE_PARAMS pointer", NvOFT_ERR_INVALID_PTR);
        }
        std::unique_ptr<CNvOFTracker> pTracker(new CNvOFTracker(createParams));
        *hOFT = (NvOFTrackerHandle)pTracker.release();
    }
    catch(const CNvOFTrackerException& e)
    {
        std::cerr << e.what() << '\n';
        status = e.GetErrorCode();
    }
    catch(const cv::Exception& e)
    {
        // TODO: We need a cleaner solution. But, this is the best we can do now.
        std::cerr << e.what() << "\n";
        std::string errorString(e.what());
        status = NvOFT_ERR_GENERIC;
        std::vector<std::string> subStrings = { "NV_OF_ERR_OF_NOT_AVAILABLE", "NV_OF_ERR_UNSUPPORTED_DEVICE" };
        auto it = std::find_if(std::begin(subStrings), std::end(subStrings), [&](const std::string& s) { return errorString.find(s) != std::string::npos; });
        if (it != std::end(subStrings))
        {
            status = NvOFT_ERR_NvOF_NOT_SUPPORTED;
        }
    }

    return status;
}

NvOFT_STATUS NvOFTProcess(NvOFTrackerHandle hOFT, const NvOFT_PROCESS_IN_PARAMS* inParams, NvOFT_PROCESS_OUT_PARAMS* outParams)
{
    NvOFT_STATUS status = NvOFT_SUCCESS;
    try
    {
        if (hOFT == NULL)
        {
            NVOFTRACKER_THROW_ERROR("Invalid handle to NvOFTracker", NvOFT_ERR_INVALID_HANDLE);
        }
        if (inParams == NULL)
        {
           NVOFTRACKER_THROW_ERROR("Invalid NvOFT_PROCESS_IN_PARAMS pointer", NvOFT_ERR_INVALID_PTR);
        }
        if (outParams == NULL)
        {
            NVOFTRACKER_THROW_ERROR("Invalid NvOFT_PROCESS_OUT_PARAMS pointer", NvOFT_ERR_INVALID_PTR);
        }
        CNvOFTracker* pTracker = (CNvOFTracker* )hOFT;
        pTracker->TrackObjects(inParams, outParams);
    }
    catch(const CNvOFTrackerException& e)
    {
        std::cerr << e.what() << '\n';
        status = e.GetErrorCode();
    }
    catch (const cv::Exception& e)
    {
        std::cerr << e.what() << "\n";
        status = NvOFT_ERR_GENERIC;
    }

    return status;
}

NvOFT_STATUS NvOFTDestroy(NvOFTrackerHandle hOFT)
{
    CNvOFTracker* pTracker = (CNvOFTracker*)hOFT;
    delete pTracker;

    return NvOFT_SUCCESS;
}