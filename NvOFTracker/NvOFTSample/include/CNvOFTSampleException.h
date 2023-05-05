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
#include "NvOFTracker.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include <exception>
#include <string>
#include <sstream>

#define NVOFTSAMPLE_THROW_ERROR(errorString) do {                      \
    std::ostringstream _error;                                         \
    _error << errorString << " "                                       \
    << " In function, " << __FUNCTION__ << ", In file, " << __FILE__   \
    << ", at line, " << __LINE__ << std::endl;                         \
    throw std::runtime_error(_error.str());                            \
} while(0);

#define CK_CUDA(func) do {                                             \
    cudaError_t status = (func);                                       \
    if (status != 0) {                                                 \
        std::ostringstream cudaErr;                                    \
        cudaErr << "Cuda Runtime Failure: " << status;                 \
        NVOFTSAMPLE_THROW_ERROR(cudaErr.str());                        \
    }                                                                  \
} while(0);

#define CK_CU(func) do {                                               \
    CUresult result = (func);                                          \
    if (result != 0) {                                                 \
        std::ostringstream cudaResult;                                 \
        cudaResult << "Cuda Driver Failure: " << result;               \
        NVOFTSAMPLE_THROW_ERROR(cudaResult.str());                     \
    }                                                                  \
} while(0);

#define CK_NVOFTRACKER(func) do {                                      \
    NvOFT_STATUS status = (func);                                      \
    if (status != 0) {                                                 \
        std::ostringstream nvoftrackerErr;                             \
        nvoftrackerErr << "NvOFTracker Failure: " << status;           \
        NVOFTSAMPLE_THROW_ERROR(nvoftrackerErr.str());                 \
    }                                                                  \
} while(0);
