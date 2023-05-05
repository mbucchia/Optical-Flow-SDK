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

#pragma once
#include "NvOFTracker.h"

#include <exception>
#include <string>
#include <sstream>

class CNvOFTrackerException : public std::exception
{
public:
    CNvOFTrackerException(const std::string& errorString, NvOFT_STATUS status) :
    m_ErrorString(errorString),
    m_Status(status){}
    virtual ~CNvOFTrackerException() {}
    virtual const char* what() const noexcept { return m_ErrorString.c_str(); }
    NvOFT_STATUS GetErrorCode() const { return m_Status; }
    static CNvOFTrackerException MakeCNvOFTrackerException(const std::string& erroString, NvOFT_STATUS status, const std::string& functionName, const std::string& fileName, int lineNo);
private:
    std::string m_ErrorString;
    NvOFT_STATUS m_Status;
};

inline CNvOFTrackerException CNvOFTrackerException::MakeCNvOFTrackerException(const std::string& errorString, NvOFT_STATUS status, const std::string& functionName, const std::string& fileName, int lineNo)
{
    std::ostringstream oss;
    oss << functionName << " : " << errorString << " at " << fileName << ":" << lineNo << std::endl
        << "Status: " << status << std::endl;
    CNvOFTrackerException exception(oss.str(), status);

    return exception;
}

#define NVOFTRACKER_THROW_ERROR(errorString, status)                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        throw CNvOFTrackerException::MakeCNvOFTrackerException(errorString, status, __FUNCTION__, __FILE__, __LINE__); \
    } while(0)

#define CK_CUDA(func) do {                                             \
    cudaError_t status = (func);                                       \
    if (status != 0) {                                                 \
        std::ostringstream cudaErr;                                    \
        cudaErr << "Cuda Runtime Failure: " << status;                 \
        NVOFTRACKER_THROW_ERROR(cudaErr.str(), NvOFT_ERR_GENERIC);     \
    }                                                                  \
} while(0);
