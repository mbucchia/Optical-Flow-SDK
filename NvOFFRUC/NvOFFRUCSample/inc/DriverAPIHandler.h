/*
* Copyright 2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once

#ifndef DRIVER_API_HANDLE_H
#define DRIVER_API_HANDLE_H

#include <iostream>
#include <vector>
#include "../cudart_header/cuda.h"
#include <stdint.h>
#include <mutex>
#include <cassert>
#include <../../Interface/NvOFFRUC.h>

#if defined _MSC_VER
#include <Windows.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <dlfcn.h>
#include <memory>
#define IUnknown void
#define GETPROCEDUREADDRESS dlsym
#endif

#ifndef GETPROCEDUREADDRESS
#define GETPROCEDUREADDRESS GetProcAddress
#endif

#define CUDA_VERSION_NUMBER 11020

typedef CUresult(CALLBACK* PFNCUINIT)(unsigned int); //Function pointer to cuInit() cuda driver api
typedef CUresult(CALLBACK* PFNCUDEVICEGETCOUNT)(int *);//Function pointer to cuDeviceCount() cuda driver api
typedef CUresult(CALLBACK* PFNCUDEVICEGET)(CUdevice *, int);//Function pointer to cuDeviceGet() cuda driver api
typedef CUresult(CALLBACK* PFNCUCTXCREATE)(CUcontext *, unsigned int,CUdevice);//Function pointer to cuCtxCreate() cuda driver api
typedef CUresult(CALLBACK* PFNCUCTXDESTROY)(CUcontext);//Function pointer to cuCtxDestroy() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMALLOC)(CUdeviceptr *,size_t);//Function pointer to cuMemAlloc() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMFREE)(CUdeviceptr);//Function pointer to cuMemFree() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMCPYHTOD)(CUdeviceptr,const void *, size_t);//Function pointer to cuMemCpyHtoD() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMCPYDTOH)(void *, CUdeviceptr,size_t ByteCount);//Function pointer to cuMemCpyDtoH() cuda driver api
typedef CUresult(CALLBACK* PFNCTXCUCTXSETCURRENT)(CUcontext);//Function pointer to cuCtxSetCurrent() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMALLOCPITCH)(CUdeviceptr *dptr,size_t *pPitch,size_t WidthInBytes,size_t Height,unsigned int ElementSizeBytes);//Function pointer to cuMemAllocPitch() cuda driver api
typedef CUresult(CALLBACK* PFNCUMEMCPY2D)(const CUDA_MEMCPY2D *pCopy);//Function pointer to cuMemCpy2D() cuda driver api
typedef CUresult(CALLBACK* PFNCUARRAYCREATE)(CUarray* pHandle,const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);//Function pointer to cuArrayCreate() cuda driver api
typedef CUresult(CALLBACK* PFNCUARRAYDESTROY)(CUarray hArray);//Function pointer to cuArrayDestroy() cuda driver api
typedef CUresult(CALLBACK* PFNCUCTXPUSHCURRENT)(CUcontext ctx);//Function pointer to cuCtxCreate() cuda driver api
typedef CUresult(CALLBACK* PFNCUCTXPOPCURRENT)(CUcontext* ctx);//Function pointer to cuCtxPopCurrent() cuda driver api
typedef CUresult(CALLBACK* PFNCUGETPROCADDRESS)(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags);//Function pointer to cuGetProcAddress() cuda driver api

/*
* This structure holds the function pointers to cuda driver api's, we load nvcuda.dll at runtime, and find the
* function pointers for each structure entry below, this structure is then shared to client application which can call
* the driver api without having to install cuda runtime
*/
typedef struct _CUDA_DRIVER_API_FUNCTION_LIST
{
    PFNCUINIT cuInitPFN;
    PFNCUGETPROCADDRESS cuGetProcAddressPFN;
    PFNCUDEVICEGETCOUNT cuDeviceGetCountPFN;
    PFNCUDEVICEGET cuDeviceGetPFN;
    PFNCUCTXCREATE cuCtxCreatePFN;
    PFNCUCTXDESTROY cuCtxDestroyPFN;
    PFNCUMEMALLOC cuMemAllocPFN;
    PFNCUMEMFREE cuMemFreePFN;
    PFNCUMEMCPYHTOD cuMemcpyHtoDPFN;
    PFNCUMEMCPYDTOH cuMemcpyDtoHPFN;
    PFNCTXCUCTXSETCURRENT cuCtxSetCurrentPFN;
    PFNCUMEMALLOCPITCH cuMemAllocPitchPFN;
    PFNCUMEMCPY2D cuMemcpy2DPFN;
    PFNCUARRAYCREATE cuArrayCreatePFN;
    PFNCUARRAYDESTROY cuArrayDesstroyPFN;
    PFNCUCTXPUSHCURRENT cuCtxPushCurrentPFN;
    PFNCUCTXPOPCURRENT cuCtxPopCurrentPFN;
} CUDA_DRIVER_API_FUNCTION_LIST;

class CUDADriverAPIHandler
{
public:
    CUDADriverAPIHandler() 
    {

#if defined _MSC_VER	
        m_pCUDALibHandle = LoadLibrary(TEXT("nvcuda.dll"));
#elif defined __GNUC__
        m_pCUDALibHandle = dlopen("libcuda.so", RTLD_LAZY); //todo : reinstate this check later
#endif
        m_cudaDriverAPI.reset(new CUDA_DRIVER_API_FUNCTION_LIST());
        CreatPFNDrvAPI();
    };

    virtual ~CUDADriverAPIHandler() 
    {
        DestroyPFNDrvAPI();
    };

    CUDA_DRIVER_API_FUNCTION_LIST* GetAPI() 
    {
        return  m_cudaDriverAPI.get();
    }

    HRESULT CreatPFNDrvAPI() 
    {

        HRESULT hr = S_OK;
        
        m_cudaDriverAPI.get()->cuGetProcAddressPFN = (PFNCUGETPROCADDRESS)GETPROCEDUREADDRESS(m_pCUDALibHandle, "cuGetProcAddress");

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuInit", (void**)&m_cudaDriverAPI.get()->cuInitPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuDeviceGetCount", (void**)&m_cudaDriverAPI.get()->cuDeviceGetCountPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuDeviceGet", (void**)&m_cudaDriverAPI.get()->cuDeviceGetPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuCtxCreate", (void**)&m_cudaDriverAPI.get()->cuCtxCreatePFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuCtxDestroy", (void**)&m_cudaDriverAPI.get()->cuCtxDestroyPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemAlloc", (void**)&m_cudaDriverAPI.get()->cuMemAllocPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemcpyDtoH", (void**)&m_cudaDriverAPI.get()->cuMemcpyDtoHPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemcpyHtoD", (void**)&m_cudaDriverAPI.get()->cuMemcpyHtoDPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemFree", (void**)&m_cudaDriverAPI.get()->cuMemFreePFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuCtxSetCurrent", (void**)&m_cudaDriverAPI.get()->cuCtxSetCurrentPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemAllocPitch", (void**)&m_cudaDriverAPI.get()->cuMemAllocPitchPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuMemcpy2D", (void**)&m_cudaDriverAPI.get()->cuMemcpy2DPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuArrayCreate", (void**)&m_cudaDriverAPI.get()->cuArrayCreatePFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuArrayDestroy", (void**)&m_cudaDriverAPI.get()->cuArrayDesstroyPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuCtxPushCurrent", (void**)&m_cudaDriverAPI.get()->cuCtxPushCurrentPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        m_cudaDriverAPI.get()->cuGetProcAddressPFN("cuCtxPopCurrent", (void**)&m_cudaDriverAPI.get()->cuCtxPopCurrentPFN, CUDA_VERSION_NUMBER, CU_GET_PROC_ADDRESS_DEFAULT);

        assert(m_cudaDriverAPI.get()->cuInitPFN != NULL || m_cudaDriverAPI.get()->cuDeviceGetCountPFN != NULL || m_cudaDriverAPI.get()->cuDeviceGetPFN != NULL ||
            m_cudaDriverAPI.get()->cuCtxCreatePFN != NULL || m_cudaDriverAPI.get()->cuCtxDestroyPFN != NULL || m_cudaDriverAPI.get()->cuMemAllocPFN != NULL ||
            m_cudaDriverAPI.get()->cuMemcpyDtoHPFN != NULL || m_cudaDriverAPI.get()->cuMemcpyHtoDPFN != NULL || m_cudaDriverAPI.get()->cuMemFreePFN != NULL ||
            m_cudaDriverAPI.get()->cuCtxSetCurrentPFN != NULL || m_cudaDriverAPI.get()->cuMemAllocPitchPFN != NULL || m_cudaDriverAPI.get()->cuMemcpy2DPFN != NULL ||
            m_cudaDriverAPI.get()->cuArrayCreatePFN != NULL || m_cudaDriverAPI.get()->cuArrayDesstroyPFN != NULL
        );
        return hr;
    }

    HRESULT DestroyPFNDrvAPI() 
    {
        m_cudaDriverAPI.release();
#if defined _MSC_VER
        bool bResult = FreeLibrary(m_pCUDALibHandle);
        return bResult ? S_OK : S_FALSE;
#elif defined __GNUC__		
        bool bResult = dlclose(m_pCUDALibHandle);
        return bResult ? S_OK : S_FALSE;
#endif
    }


protected:

#if defined _MSC_VER
    HMODULE m_pCUDALibHandle;
#elif defined __GNUC__
    void* m_pCUDALibHandle;
#endif
    std::unique_ptr<CUDA_DRIVER_API_FUNCTION_LIST> m_cudaDriverAPI;

};

#endif //DRIVER_API_HANDLE_H