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
#include "../../Interface/NvOFFRUC.h"
#include "../inc/FrameGenerator.h"
#include "../inc/Arguments.h"
#include <fstream>

#if defined _MSC_VER
#include "SecureLibraryLoader.h"
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <dlfcn.h>
#define IUnknown void
#define GETPROCEDUREADDRESS dlsym
#endif

#ifndef GETPROCEDUREADDRESS
#define GETPROCEDUREADDRESS GetProcAddress
#endif
#define PERF_LEVEL_MULTIPLY_FACTOR 5

double  g_dPCFreq                   = 0.0;
__int64 g_iCounterStart             = 0;
double  g_doubleCounterStart        = 0.0;
double  g_dTotalTime                = 0;
double  g_dRepeatFrameExecutionTime = 0;
int     g_iProcessedFrameCount      = 0;
int     g_iRepeatedFrameCount       = 0;


#if defined __GNUC__
int QueryPerformanceCounter(double & val)
{
    struct timespec ts;
    ts.tv_nsec = 0;
    ts.tv_sec = 0;
    double theTick = 0.0;
    clock_gettime(CLOCK_REALTIME, &ts); //LINUX
    theTick = ts.tv_nsec / 1000000.0;
    theTick += ts.tv_sec * 1000.0;
    val = theTick;
    return 0;
}
void StartCounter()
{
    double li;
    QueryPerformanceCounter(li);
    g_doubleCounterStart = li;
}
double GetCounter()
{
    double li;
    QueryPerformanceCounter(li);
    double timeTaken = double(li - g_doubleCounterStart);// / g_dPCFreq;
    g_dTotalTime += timeTaken;
    g_iProcessedFrameCount++;
    return timeTaken;
}

#elif defined _MSC_VER
/**
 * Sets the QueryPerformanceCounter clicks to 0
 * This function is used to measure execution duration of function in CPU
 *
 * @param none
 * @return none
 */
void StartCounter()
{
    LARGE_INTEGER li = { 0 };

    if (!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";

    g_dPCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    g_iCounterStart = li.QuadPart;
}

/**
 * Sets QueryPerformanceCounter to CPU ticks elapsed from the time StartCounter() was called
 * This function is used to measure execution duration of function in CPU
 *
 * @param none
 * @return none
 */
double GetCounter()
{
    LARGE_INTEGER li = { 0 };

    QueryPerformanceCounter(&li);
    double timeTaken = double(li.QuadPart - g_iCounterStart) / g_dPCFreq;
    g_dTotalTime += timeTaken;
    g_iProcessedFrameCount++;
    return timeTaken;
}
#endif

/**
*Displays NvOFFRUC error strings to terminal
**/
void DisplayError(NvOFFRUC_STATUS *message)
{
    if (message && *message > NvOFFRUC_SUCCESS && *message < NvOFFRUC_ERR_MAX_ERROR) {
        std::cerr << NvOFFRUCErrorString[*message] << std::endl;
    }
}

int main(int argc, char* argv[])
{
    FrameGenerator*     objFrameGenerator;
    Arguments           stArgs;
    NvOFFRUC_STATUS       status = NvOFFRUC_SUCCESS;

    //Declare handles and function pointers
#if defined _MSC_VER
    HINSTANCE hDLL = NULL;
#else
    void* hDLL = NULL;
#endif

    PtrToFuncNvOFFRUCCreate               NvOFFRUCCreate = NULL;
    PtrToFuncNvOFFRUCRegisterResource     NvOFFRUCRegisterResource = NULL;
    PtrToFuncNvOFFRUCUnregisterResource   NvOFFRUCUnregisterResource = NULL;
    PtrToFuncNvOFFRUCProcess              NvOFFRUCProcess = NULL;
    PtrToFuncNvOFFRUCDestroy              NvOFFRUCDestroy = NULL;

    //Parse Arguments
    if (!stArgs.ParseArguments(argc, (const char**)argv))
    {
        std::cerr << "Failed to parse input argument correctly" << std::endl;
        return -1;
    }

    if (stArgs.m_ResourceType == DirectX11Resource)
    {
#if defined _MSC_VER
        objFrameGenerator = new FrameGeneratorD3D11;
#elif defined __GNUC__
            std::cerr << "DirectX is not supported allocation type for Linux OS" << std::endl;
            return -1;
#endif
    }
    else if (stArgs.m_ResourceType == CudaResource)
    {
        objFrameGenerator = new FrameGeneratorCUDA;
    }

    if (objFrameGenerator == NULL)
    {
        std::cerr << "Failed to create object of FrameGenerator" << std::endl;
        return -1;
    }

    if (!objFrameGenerator->Init(stArgs))
    {
        std::cout << "Failed to initialize frameGenerator" << std::endl;
        if (objFrameGenerator != NULL)
        {
            objFrameGenerator->Destroy();
            delete objFrameGenerator;
            objFrameGenerator = NULL;
        }
        return -1;
    }
   
#if defined _MSC_VER
    //Load dll and populate function pointers
    SecureLoadLibrary(L"NvOFFRUC.dll", &hDLL);
#elif defined __GNUC__
    hDLL = dlopen("libNvOFFRUC.so", RTLD_LAZY);
#endif
    if (hDLL != NULL)
    {
        NvOFFRUCCreate                = (PtrToFuncNvOFFRUCCreate)GETPROCEDUREADDRESS(hDLL, CreateProcName);
        NvOFFRUCRegisterResource      = (PtrToFuncNvOFFRUCRegisterResource)GETPROCEDUREADDRESS(hDLL, RegisterResourceProcName);
        NvOFFRUCUnregisterResource    = (PtrToFuncNvOFFRUCUnregisterResource)GETPROCEDUREADDRESS(hDLL, UnregisterResourceProcName);
        NvOFFRUCProcess               = (PtrToFuncNvOFFRUCProcess)GETPROCEDUREADDRESS(hDLL, ProcessProcName);
        NvOFFRUCDestroy               = (PtrToFuncNvOFFRUCDestroy)GETPROCEDUREADDRESS(hDLL, DestroyProcName);

        if (
            !NvOFFRUCCreate
            || !NvOFFRUCRegisterResource
            || !NvOFFRUCUnregisterResource
            || !NvOFFRUCProcess
            || !NvOFFRUCDestroy)
        {
            std::cerr << "DLL/so exports not found, exiting!" << std::endl;
#if defined _MSC_VER
            FreeLibrary(hDLL);
#elif defined __GNUC__
            dlclose(hDLL);
#endif            
            if (objFrameGenerator != NULL)
            {
                objFrameGenerator->Destroy();
                delete objFrameGenerator;
                objFrameGenerator = NULL;
            }
            return -1;
        }
    }
    else
    {
        std::cerr << "DLL handle is NULL, exiting!" << std::endl;
        if (objFrameGenerator != NULL)
        {
            objFrameGenerator->Destroy();
            delete objFrameGenerator;
            objFrameGenerator = NULL;
        }
        return -1;
    }

    NvOFFRUC_CREATE_PARAM             createParams = { 0 };
    NvOFFRUCHandle                    hFRUC;

    createParams.pDevice            = objFrameGenerator->GetDevice();
    createParams.uiHeight           = stArgs.m_Height;
    createParams.uiWidth            = stArgs.m_Width;
    createParams.eResourceType      = (NvOFFRUCResourceType)stArgs.m_ResourceType;
    createParams.eSurfaceFormat     = (NvOFFRUCSurfaceFormat)stArgs.m_InputSurfaceFormat;
    createParams.eCUDAResourceType  = (NvOFFRUCCUDAResourceType)stArgs.m_CudaResourceType;
    
    //Initialize FRUC pipeline which internally initializes Optical flow engine
    status = NvOFFRUCCreate(
                &createParams,
                &hFRUC);
    if (status != NvOFFRUC_SUCCESS)
    {
        std::cerr << "NvOFFRUCCreate failed with status " << status << std::endl;
        delete objFrameGenerator;
        objFrameGenerator = NULL;
        return -1;
    }

    //If allocationtype is DX then dx11 textures are created and cuda-dx interop is used to share texture handles with FRUC pipeline 
    //which internally calls OF API's, this requires synchronization.
    //If allocationtype is CUDA then we either create input memory having type cuDevicePtr or cuArray 
    //and share their pointers to FRUC pipeline, this does not require synchronization.
    //For dx-cuda interop synchronization IDXGIKeyedMutex and ID3D11Fence are supported,
    //We first check is OS build version is less than 1703, if it is then we force
    //IDXGIKeyedMutex to be synchronization mechanism else we default back to ID3D11Fence,.
    NvOFFRUC_REGISTER_RESOURCE_PARAM regOutParam = { 0 };
    objFrameGenerator->GetResource(
                        regOutParam.pArrResource,
                        regOutParam.uiCount);
    regOutParam.pD3D11FenceObj = objFrameGenerator->GetFenceObj();//passing NULL here to indicate that you cannot create fence
 
    status = NvOFFRUCRegisterResource(
                hFRUC,
                &regOutParam);
    if (status != NvOFFRUC_SUCCESS)
    {
        std::cerr << "NvOFFRUCRegisterResource failed with status " << status << std::endl;
        delete objFrameGenerator;
        objFrameGenerator = NULL;
        NvOFFRUCDestroy(hFRUC);
        return -1;
    }

    bool bHasFrameRepetitionOccured = false;

    for (int iFrameIndex = stArgs.m_StartFrame; iFrameIndex <= stArgs.m_EndFrame; iFrameIndex++)
    {
        NvOFFRUC_PROCESS_IN_PARAMS    stInParams      = { 0 };
        NvOFFRUC_PROCESS_OUT_PARAMS   stOutParams     = { 0 };
        void                        *pRenderTexture = NULL;
        double                      renderTime      = 0;

        bool bReturn = objFrameGenerator->Render(
                            &pRenderTexture,
                            renderTime);
        if (!bReturn)
        {
            break;
        }

        //Wait for GPU to complete the dx rendering work so that cuda can proceed
        bReturn = objFrameGenerator->WaitForUploadToComplete();
        if (!bReturn)
        {
            std::cout << "Unable to wait for upload to complete" << std::endl;
        }

        stInParams.stFrameDataInput.pFrame          = pRenderTexture;
        stInParams.stFrameDataInput.nTimeStamp      = renderTime;
        stInParams.stFrameDataInput.nCuSurfacePitch = objFrameGenerator->getRenderAllocPitchSize();

        //Checks if OS build version >= 1703 and sets FenceSupported flag to true
        //interfaces ID3D11Device5, ID3D11DeviceContext4 & ID3D11Fence are available on Windows 10 v1703 and later
        if (objFrameGenerator->IsFenceSupported())
        {
            stInParams.uSyncWait.FenceWaitValue.uiFenceValueToWaitOn = objFrameGenerator->m_uiFenceValue;//dx will signal on this value once rendering is complete
        }
        else
        {
            stInParams.uSyncWait.MutexAcquireKey.uiKeyForRenderTextureAcquire = objFrameGenerator->GetActiveRenderTextureKey();
            stInParams.uSyncWait.MutexAcquireKey.uiKeyForInterpTextureAcquire = objFrameGenerator->GetActiveInterpolateTextureKey();
        }

        void*   pInterpolateTexture = NULL;
        double  interpolateTime = 0;

        bReturn = objFrameGenerator->GetResourceForInterpolation(
                                        &pInterpolateTexture,
                                        interpolateTime);
        if (!bReturn)
        {
            std::cout << "Unable to find a free DX resource to share with NvOFFRUCProcess" << std::endl;
            return -1;
        }
        stOutParams.stFrameDataOutput.pFrame            = pInterpolateTexture;
        stOutParams.stFrameDataOutput.nTimeStamp        = interpolateTime;
        stOutParams.stFrameDataOutput.nCuSurfacePitch   = objFrameGenerator->getInterpolateAllocPitchSize();
        stOutParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &bHasFrameRepetitionOccured;
        if (objFrameGenerator->IsFenceSupported())
        {
            stOutParams.uSyncSignal.FenceSignalValue.uiFenceValueToSignalOn = ++objFrameGenerator->m_uiFenceValue; //cuda will signal on this fence value once interpolated frame is ready
        }
        else
        {
            objFrameGenerator->SetActiveRenderTextureKey(objFrameGenerator->GetActiveRenderTextureKey() + 1);
            objFrameGenerator->SetActiveInterpolateTextureKey(objFrameGenerator->GetActiveInterpolateTextureKey() + 1);
            stOutParams.uSyncSignal.MutexReleaseKey.uiKeyForRenderTextureRelease = objFrameGenerator->GetActiveRenderTextureKey();
            stOutParams.uSyncSignal.MutexReleaseKey.uiKeyForInterpolateRelease = objFrameGenerator->GetActiveInterpolateTextureKey();
        }

        //Record the number of ticks the performance counter has in the g_iCounterStart variable
        StartCounter();

        //Take two input surfaces (for e.g. Input_0 and Input_1) and 
        //produce a temporal equidistant output surface (for e.g. Interpolated_0.5)
        status = NvOFFRUCProcess(
                    hFRUC,
                    &stInParams,
                    &stOutParams);
        if (status == NvOFFRUC_SUCCESS)
        {
            bReturn = objFrameGenerator->WaitForInterpolationToComplete();

            //Get number of milliseconds since StartCounter() was last called as a double
            double dFRUCExecutionTimeMS = GetCounter();
            
            if (!bReturn)
            {
                status = NvOFFRUC_ERR_SYNC_WRITE_FAILED;
            }
            else
            {
                bReturn = objFrameGenerator->WriteOutput(
                                                pInterpolateTexture,
                                                TEXTURE_TYPE_INTERPOLATE);
                if (!bReturn)
                {
                    status = NvOFFRUC_ERR_WRITE_TODISK_FAILED;
                }
                else
                {
                    //When the Interpolated frame has very bad quality, we copy the previous frames data to interpolated frame and send it, this is done to save cycles for next execution
                    //We store execution duration of pipeline for each run in dFRUCExecutionTimeMS and add it to g_dTotalTime to get execution duration of all frames on which pipeline was run
                    //When interpolate frame repeats, entire pipeline is not executed hence substract this execution duration from g_dTotalTime and increment the g_iRepeatedFrameCount.
                    if (bHasFrameRepetitionOccured)
                    {
                        g_iRepeatedFrameCount++;
                        g_dRepeatFrameExecutionTime += dFRUCExecutionTimeMS;
                        bHasFrameRepetitionOccured = false;
                    }

                    bReturn = objFrameGenerator->WriteOutput(
                                                    pRenderTexture,
                                                    TEXTURE_TYPE_RENDER);
                    if (!bReturn)
                    {
                        status = NvOFFRUC_ERR_WRITE_TODISK_FAILED;
                    }
                }
            }
        }
        else
        {
            status = NvOFFRUC_ERR_PIPELINE_EXECUTION_FAILURE;
        }
        DisplayError(&status);
    }

    if (status == NvOFFRUC_SUCCESS)
    {
        if (g_iProcessedFrameCount == 0)
        {
            g_iProcessedFrameCount = stArgs.m_EndFrame;
        }
        //Number of Input frames to Process
        std::cout << "Total Number of Frames : " << g_iProcessedFrameCount << std::endl;
        //When the Interpolated frame has very bad quality, we copy the previous frames data to interpolated frame and send it, this is done to save cycles for next execution
        std::cout << "Number of Frames repeated : " << g_iRepeatedFrameCount << std::endl;
        //Summing up execution duration of all calls to NvOFFRUCProcess(), this includes repeated interpolated frames as well
        std::cout << "Average NvOFFRUCProcess duration of Total Frames in milliseconds: " << (g_dTotalTime / g_iProcessedFrameCount) << std::endl;
        //now get the processed frames after substracting repeating frames and also substracting the execution duration of repeated frames from total duration
        std::cout << "Adjusted Average NvOFFRUCProcess duration of All Frames (Total - Repeated) in milliseconds : " << (max(g_dTotalTime - g_dRepeatFrameExecutionTime, 0.0) / max(1, max(g_iProcessedFrameCount - g_iRepeatedFrameCount, 0))) << std::endl;
    }
    
    //Remove the mapping created by client side resources and destroy any allocation in the pipeline
    NvOFFRUC_UNREGISTER_RESOURCE_PARAM stUnregisterResourceParam = { 0 };
    memcpy(
        stUnregisterResourceParam.pArrResource,
        regOutParam.pArrResource,
        regOutParam.uiCount * sizeof(IUnknown*));
    stUnregisterResourceParam.uiCount = regOutParam.uiCount;

    status = NvOFFRUCUnregisterResource(
                hFRUC, 
                &stUnregisterResourceParam);
    if (status != NvOFFRUC_SUCCESS)
    {
        std::cerr << "NvOFFRUCUnregisterResource failed with error code :" << status << std::endl;
        return -1;
    }

    //Destroy FRUC instance
    status = NvOFFRUCDestroy(hFRUC);
    if (status != NvOFFRUC_SUCCESS)
    {
        std::cerr << "NvOFFRUCDestroy failed with error code :" << status << std::endl;
    }

    objFrameGenerator->Destroy();
    delete objFrameGenerator;
    objFrameGenerator = NULL;

    return 0;
}
