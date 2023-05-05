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

#include "DriverAPIHandler.h"
#include <iostream>
#include <vector>

#if defined _MSC_VER
#include <d3d11.h>
#include <d3d9.h>
#include "d3d9.h"
#include "d3d12.h"
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <d3d11_3.h>
#include <d3d11_4.h>
#include <Dxva2api.h>
#endif

#include "Arguments.h"
#include "Common.h"
#include "../../Interface/NvOFFRUC.h"


const int NUM_RENDER_TEXTURE = 2;
const int NUM_INTERPOLATE_TEXTURE = 1;
const float DEFAULT_FRUC_SPEED = 0.5f;

#define RETURN_ON_D3D_ERROR(errorCode) if(FAILED(errorCode)) return 0;

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err) 
    {
        fprintf(stderr,
            "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err, file, line);
        exit(-1);
    }
}

//Texture Type 
enum TextureType {
    TEXTURE_TYPE_RENDER = 0,
    TEXTURE_TYPE_INTERPOLATE
};

//Render Frame information
struct FrameInfo {
    int iRenderWidth;
    int iRenderHeight;
    int iPitch;
};

enum BufferType
{
    RENDER_HOST = 0,
    INTERPOLATED_HOST,
    RENDER_DEVPTR,
    INTERPOLATED_DEVPTR,
    RENDER_CUARRAY,
    INTERPOLATED_CUARRAY,
};

/*
* This class parses input yuv/png, fills the structure which handles per frame data
* also handles sending/receiving data to/from pipeline 
*/
class FrameGenerator {
public:
    virtual ~FrameGenerator() {};
    virtual bool  Render(
                        void** ppTexture,
                        double& dMillisec) = 0;
    virtual bool  WriteOutput(  
                        void* pTexture,
                        TextureType eTextureType) = 0;
    virtual bool  CreateTextureBuffer(
                        NvOFFRUCSurfaceFormat d3dSurfaceFormat) = 0;
    virtual void  Destroy() = 0;
    virtual bool  GetResource(
                        void** ppTexture,
                        uint32_t& uiCount) = 0;
    virtual HRESULT CreateDevice(void** pDevice) = 0;
    virtual bool GetResourceForInterpolation(
                        void** ppTexture,
                        double& dMillisec) = 0;
    bool Init(Arguments args);
    bool CreateSource(
                        std::string strInputFileName,
                        int iStartFrom);
    bool CreateSink(
                        std::string strOutputFileName,
                        std::string strInputFileName);
    bool AllocateInputCPUBuffer();
    void GetLastRenderedFileName(std::string strFilePath);
    void* GetDevice() { return m_pDevice; };

    virtual void        *GetFenceObj() { return NULL; };
    virtual bool        WaitForUploadToComplete() { return false; };
    virtual bool        WaitForInterpolationToComplete() { return false; };
    virtual bool        IsFenceSupported() { return m_bFenceSupported; };
    virtual uint64_t    GetActiveRenderTextureIndex() { return 0; }
    virtual uint64_t    GetActiveInterpolateTextureIndex() { return 0; }
    virtual void        SetActiveRenderTextureKey(uint64_t m_newKeyValue) {}
    virtual void        SetActiveInterpolateTextureKey(uint64_t m_newKeyValue) {}
    virtual uint64_t    GetActiveRenderTextureKey() { return 0; }
    virtual uint64_t    GetActiveInterpolateTextureKey() { return 0; }
    virtual bool        WaitForSyncObject(TextureType eTextureType) { return 0; };
    virtual bool        SignalSyncObject(TextureType eTextureType) { return 0; };
    virtual size_t      getRenderAllocPitchSize() { return 0; }
    virtual size_t      getInterpolateAllocPitchSize() { return 0; }

    __inline void FLAG_D3D_ERROR(HRESULT hr)
    {
        if (hr != S_OK)
        {
            std::cout << "Error : D3D call failed with " << hr;
        }
    }
    size_t GetInputBufferSize() { return m_InputBufferSize; }
public:
    uint64_t                        m_uiFenceValue;
protected:
    void*                           m_pDevice;
    std::ifstream                   m_Source;               // Required for read YUV file
    std::ofstream                   m_Output;               // Required for read YUV file
    std::string                     m_strOutputDirectory;
    std::vector<std::string>        m_vecFileNames; // Required for PNG files
    int                             m_iPNGVectorIndex;                 // Required for PNG files
    int                             m_iRenderIndex;
    double                          m_dLastRenderTime;
    std::string                     m_strLastRenderFileName;     // Required to save PNG files
    const double                    m_constdRenderInterval = 1;    //dMillisec
    int                             m_iFrameCnt;
    FrameInfo                       m_FrameInfo;
    std::vector<char>               m_InputCPUBuffer;
    size_t                          m_InputBufferSize = 0;
    NvOFFRUCInputFileType           m_InputFileType;
    NvOFFRUCSurfaceFormat           m_eSurfaceFormat;
    NvOFFRUCCUDAResourceType        m_cudaResourceType;
    NvOFFRUCResourceType            m_eResourceType;
    uint64_t                        m_uiRenderKey[NUM_RENDER_TEXTURE];
    uint64_t                        m_uiInterpolateKey[NUM_INTERPOLATE_TEXTURE];
    bool                            m_bFenceSupported;
};

#if defined _MSC_VER
class FrameGeneratorD3D11 : public FrameGenerator
{
public:
    virtual             ~FrameGeneratorD3D11();
    virtual bool        WaitForUploadToComplete();
    virtual bool        WaitForInterpolationToComplete();
    virtual void        SetActiveRenderTextureKey(uint64_t uiNewKeyValue);
    virtual void        SetActiveInterpolateTextureKey(uint64_t uiNewKeyValue);
    virtual uint64_t    GetActiveRenderTextureKey();
    virtual uint64_t    GetActiveInterpolateTextureKey();
    virtual bool        WaitForSyncObject(TextureType eTextureType);
    virtual bool        SignalSyncObject(TextureType eTextureType);

    FrameGeneratorD3D11::FrameGeneratorD3D11()
    {
        m_pDevice = NULL;
        m_pContext = NULL;
        for (int idx = 0; idx < NUM_INTERPOLATE_TEXTURE; idx++)
        {
            m_pInterpolateTexture2D[idx] = NULL;
            m_pInterpolateTextureKeyedMutex[idx] = NULL;
        }

        for (int idx = 0; idx < _countof(m_pRenderTexture2D); idx++)
        {
            m_pRenderTexture2D[idx] = NULL;
            m_pRenderTextureKeyedMutex[idx] = NULL;
        }

        m_pStagingTexture2D     = NULL;
        m_uiFenceValue          = 0;
        m_bFenceSupported       = false;
        m_pDevice5              = NULL;
        m_pDeviceContext4       = NULL;
        m_hFenceEvent           = NULL;
    };

    void* GetFenceObj() { return m_pFence; };
    bool  CreateTextureBuffer(NvOFFRUCSurfaceFormat d3dSurfaceFormat);
    void  Destroy();
    bool  Render(
                void** ppTexture,
                double& dMillisec);
    bool  WriteOutput(
                void* pTexture,
                TextureType eTextureType);
    bool GetResourceForInterpolation(
                void** ppTexture,
                double& dMillisec);
    bool GetResource(
                void** ppTexture,
                uint32_t& uiCount);
    HRESULT CreateDevice(void** pDevice);

private:
    ID3D11Device*           m_pDevice;
    ID3D11DeviceContext*    m_pContext;
    ID3D11Texture2D*        m_pInterpolateTexture2D[NUM_INTERPOLATE_TEXTURE];
    ID3D11Texture2D*        m_pRenderTexture2D[NUM_RENDER_TEXTURE];
    ID3D11Texture2D*        m_pStagingTexture2D;
    ID3D11Fence*            m_pFence;
    ID3D11Device5*          m_pDevice5;
    ID3D11DeviceContext4*   m_pDeviceContext4;
    IDXGIKeyedMutex*        m_pInterpolateTextureKeyedMutex[NUM_INTERPOLATE_TEXTURE];
    IDXGIKeyedMutex*        m_pRenderTextureKeyedMutex[NUM_RENDER_TEXTURE];
    HANDLE                  m_hFenceEvent;
};
#endif

class BufferManagerBase
{
public:
    virtual             ~BufferManagerBase() {}
    template<class T> T& GetCudaMemPtr(
                                    BufferType eBufferType,
                                    int index);
    virtual CUDADriverAPIHandler* getCUDADrvAPIHandle() = 0;
    virtual bool        CreateTextureBuffer() = 0;
    virtual bool        DestroyBuffers() = 0;
    virtual bool        Render(
                                    int iRenderIndexDst, 
                                    BufferType eBufferType) = 0;
    virtual bool        WriteOutput(
                                    int iRenderIndexSrc,
                                    BufferType eBufferType) = 0;
    virtual void*       GetHostPointer(
                                    BufferType eBufferType,
                                    int index) = 0;
};

template <typename T>
class BufferManager : public BufferManagerBase
{
public:
    BufferManager() {}
    BufferManager(
                int iWidth, 
                int iHeight,
                int iPitch,
                NvOFFRUCSurfaceFormat eSurfaceFormat) :
                    m_iBufferWidth(iWidth),
                    m_iBufferHeight(iHeight),
                    m_iBufferPitch(iPitch),
                    m_eSurfaceFormat(eSurfaceFormat)
    {
        m_pCUDADriverAPIHandler.reset(new CUDADriverAPIHandler());
    }
    T& GetCudaMemPtr(
                BufferType eBufferType,
                int index);
    void* GetHostPointer(
                BufferType eBufferType,
                int index) override;
    bool CreateTextureBuffer() override;
    bool DestroyBuffers() override;
    bool Render(
                int iRenderIndexDst, 
                BufferType eBufferType) override;
    bool WriteOutput(
                int iRenderIndexSrc,
                BufferType eBufferType) override;
    CUDADriverAPIHandler* getCUDADrvAPIHandle() override { return m_pCUDADriverAPIHandler.get(); }

private:
    std::unique_ptr<CUDADriverAPIHandler>   m_pCUDADriverAPIHandler;
    T                                       m_pIntermediateFrameCudaMemPtr[NUM_INTERPOLATE_TEXTURE];
    T                                       m_pRenderFrameCudaMemPtr[NUM_RENDER_TEXTURE];
    int                                     m_iBufferWidth;
    int                                     m_iBufferHeight;
    int                                     m_iBufferPitch;
    void*                                   m_pIntermediateFrameHostPtr[NUM_INTERPOLATE_TEXTURE];
    void*                                   m_pRenderFrameHostPtr[NUM_RENDER_TEXTURE];
    BufferType                              m_eBufferType;
    NvOFFRUCSurfaceFormat                     m_eSurfaceFormat;
};

template<class T>
inline void * BufferManager<T>::GetHostPointer(BufferType eBufferType, int index)
{
    switch (eBufferType)
    {
        case RENDER_HOST:
        {
            return m_pRenderFrameHostPtr[index];
        }
        break;
        case INTERPOLATED_HOST:
        {
            return m_pIntermediateFrameHostPtr[index];
        }
        break;
        default:
        {
            return NULL;
        }
        break;
    }
}

template<>
inline CUdeviceptr& BufferManager<CUdeviceptr>::GetCudaMemPtr(BufferType eBufferType, int index)
{
    switch (eBufferType)
    {
        case RENDER_CUARRAY:
        case RENDER_DEVPTR:
        {
            return m_pRenderFrameCudaMemPtr[index];
        }
        break;
        case INTERPOLATED_CUARRAY:
        case INTERPOLATED_DEVPTR:
        {
            return m_pIntermediateFrameCudaMemPtr[index];
        }
        break;
        default:
        {
            throw std::invalid_argument("received invalid cudasurfacetype, exiting");
            exit(0);
        }
        break;
    }
}

template<>
inline bool BufferManager<CUdeviceptr>::CreateTextureBuffer()
{
    int uiCount = NUM_INTERPOLATE_TEXTURE + NUM_RENDER_TEXTURE;

    for (uint32_t idx = 0; idx < NUM_INTERPOLATE_TEXTURE; idx++)
    {
        m_pIntermediateFrameHostPtr[idx] = malloc(sizeof(char) * m_iBufferPitch * m_iBufferHeight);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemAllocPFN(&m_pIntermediateFrameCudaMemPtr[idx], sizeof(char) * m_iBufferPitch * m_iBufferHeight));
    }

    for (uint32_t idx = 0; idx < NUM_RENDER_TEXTURE; idx++)
    {
        m_pRenderFrameHostPtr[idx] = malloc(sizeof(char) * m_iBufferPitch * m_iBufferHeight);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemAllocPFN(&m_pRenderFrameCudaMemPtr[idx], sizeof(char) * m_iBufferPitch * m_iBufferHeight));
    }
    return true;
}

template<>
inline bool BufferManager<CUdeviceptr>::DestroyBuffers()
{
    for (size_t szIndex = 0; szIndex < NUM_INTERPOLATE_TEXTURE; szIndex++)
    {
        free(m_pIntermediateFrameHostPtr[szIndex]);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemFreePFN(m_pIntermediateFrameCudaMemPtr[szIndex]));
        m_pIntermediateFrameHostPtr[szIndex] = NULL;
    }
    for (size_t szIndex = 0; szIndex < NUM_RENDER_TEXTURE; szIndex++)
    {
        free(m_pRenderFrameHostPtr[szIndex]);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemFreePFN(m_pRenderFrameCudaMemPtr[szIndex]));
        m_pRenderFrameHostPtr[szIndex] = NULL;
    }
    return true;
}

template<>
inline bool BufferManager<CUdeviceptr>::Render(int iRenderIndexDst, BufferType eBufferType)
{
    CUDA_MEMCPY2D           renderStr;
    renderStr.dstDevice     = m_pRenderFrameCudaMemPtr[iRenderIndexDst];
    renderStr.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    renderStr.dstPitch      = m_iBufferPitch;
    renderStr.dstXInBytes   = 0;
    renderStr.dstY          = 0;
    renderStr.Height        = m_iBufferHeight;
    renderStr.srcHost       = m_pRenderFrameHostPtr[iRenderIndexDst];
    renderStr.srcMemoryType = CU_MEMORYTYPE_HOST;
    renderStr.srcXInBytes   = 0;
    renderStr.srcY          = 0;
    renderStr.srcPitch      = m_iBufferPitch;
    renderStr.WidthInBytes  = m_iBufferPitch;
    checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemcpy2DPFN(&renderStr));
    return true;
}

template<>
inline bool BufferManager<CUdeviceptr>::WriteOutput(int iRenderIndexSrc, BufferType eBufferType)
{
    CUDA_MEMCPY2D               objInterpStr;
    objInterpStr.dstHost        = m_pIntermediateFrameHostPtr[0];
    objInterpStr.dstMemoryType  = CU_MEMORYTYPE_HOST;
    objInterpStr.dstXInBytes    = 0;
    objInterpStr.dstY           = 0;
    objInterpStr.Height         = m_iBufferHeight;

    if (eBufferType == RENDER_DEVPTR)
    {
        objInterpStr.srcDevice = m_pRenderFrameCudaMemPtr[iRenderIndexSrc];
    }
    else
    {
        objInterpStr.srcDevice = m_pIntermediateFrameCudaMemPtr[0];
    }

    objInterpStr.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    objInterpStr.srcXInBytes    = 0;
    objInterpStr.srcY           = 0;
    objInterpStr.srcPitch       = m_iBufferPitch;
    objInterpStr.dstPitch       = m_iBufferPitch;
    objInterpStr.WidthInBytes   = m_iBufferPitch;

    checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemcpy2DPFN(&objInterpStr));
    return true;
}

template<>
inline CUarray& BufferManager<CUarray>::GetCudaMemPtr(BufferType eBufferType, int iIndex)
{
    switch (eBufferType)
    {
        case RENDER_CUARRAY:
        case RENDER_DEVPTR:
        {
            return m_pRenderFrameCudaMemPtr[iIndex];
        }
        break;
        case INTERPOLATED_CUARRAY:
        case INTERPOLATED_DEVPTR:
        {
            return m_pIntermediateFrameCudaMemPtr[iIndex];
        }
        break;
        default:
        {
            return m_pRenderFrameCudaMemPtr[iIndex];
        }
        break;
    }
}

template<>
inline bool BufferManager<CUarray>::CreateTextureBuffer()
{
    CUDA_ARRAY_DESCRIPTOR cuArrayDesc = {0};
    switch (m_eSurfaceFormat)
    {
        case NV12Surface:
        {
            cuArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            cuArrayDesc.Height = m_iBufferHeight ;
            cuArrayDesc.NumChannels = 1;
            cuArrayDesc.Width = m_iBufferWidth;
        }
        break;
        case ARGBSurface:
        {
            cuArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            cuArrayDesc.Height = m_iBufferHeight;
            cuArrayDesc.NumChannels = 4;
            cuArrayDesc.Width = m_iBufferWidth;
        }
        break;
    }

    for (uint32_t uIndex = 0; uIndex < NUM_INTERPOLATE_TEXTURE; uIndex++)
    {
        m_pIntermediateFrameHostPtr[uIndex] = malloc(sizeof(char) * m_iBufferPitch * m_iBufferHeight);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuArrayCreatePFN(&m_pIntermediateFrameCudaMemPtr[0], &cuArrayDesc));
    }
    for (uint32_t uIndex = 0; uIndex < NUM_RENDER_TEXTURE; uIndex++)
    {
        m_pRenderFrameHostPtr[uIndex] = malloc(sizeof(char) * m_iBufferPitch * m_iBufferHeight);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuArrayCreatePFN(&m_pRenderFrameCudaMemPtr[uIndex], &cuArrayDesc));
    }

    return true;
}

template<>
inline bool BufferManager<CUarray>::DestroyBuffers()
{
    for (size_t szIndex = 0; szIndex < NUM_INTERPOLATE_TEXTURE; szIndex++)
    {
        free(m_pIntermediateFrameHostPtr[szIndex]);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuArrayDesstroyPFN(m_pIntermediateFrameCudaMemPtr[szIndex]));
        m_pIntermediateFrameHostPtr[szIndex] = NULL;
    }
    for (size_t szIndex = 0; szIndex < NUM_RENDER_TEXTURE; szIndex++)
    {
        free(m_pRenderFrameHostPtr[szIndex]);
        checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuArrayDesstroyPFN(m_pRenderFrameCudaMemPtr[szIndex]));
        m_pRenderFrameHostPtr[szIndex] = NULL;
    }
    return true;
}

template<>
inline bool BufferManager<CUarray>::Render(int iRenderIndexDst, BufferType eBufferType)
{
    CUDA_MEMCPY2D           renderStr;
    renderStr.dstArray      = m_pRenderFrameCudaMemPtr[iRenderIndexDst];
    renderStr.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    renderStr.dstPitch      = m_iBufferPitch;
    renderStr.dstXInBytes   = 0;
    renderStr.dstY          = 0;
    renderStr.Height        = m_iBufferHeight;
    renderStr.srcHost       = m_pRenderFrameHostPtr[iRenderIndexDst];
    renderStr.srcMemoryType = CU_MEMORYTYPE_HOST;
    renderStr.srcXInBytes   = 0;
    renderStr.srcY          = 0;
    renderStr.srcPitch      = m_iBufferPitch;
    renderStr.WidthInBytes  = m_iBufferPitch;

    checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemcpy2DPFN(&renderStr));
    return true;
}

template<>
inline bool BufferManager<CUarray>::WriteOutput(int iRenderIndexSrc, BufferType eBufferType)
{
    CUDA_MEMCPY2D               objInterpStr;
    objInterpStr.dstHost        = (char*)m_pIntermediateFrameHostPtr[0];
    objInterpStr.dstMemoryType  = CU_MEMORYTYPE_HOST;
    objInterpStr.dstXInBytes    = 0;
    objInterpStr.dstY           = 0;
    objInterpStr.Height         = m_iBufferHeight;
    objInterpStr.srcMemoryType  = CU_MEMORYTYPE_ARRAY;
    objInterpStr.srcXInBytes    = 0;
    objInterpStr.srcY           = 0;
    objInterpStr.srcPitch       = m_iBufferPitch;
    objInterpStr.dstPitch       = m_iBufferPitch;
    objInterpStr.WidthInBytes   = m_iBufferPitch;

    if (eBufferType == RENDER_DEVPTR)
    {
        objInterpStr.srcArray = m_pRenderFrameCudaMemPtr[iRenderIndexSrc];
    }
    else
    {
        objInterpStr.srcArray = m_pIntermediateFrameCudaMemPtr[0];
    }

    checkCudaErrors(getCUDADrvAPIHandle()->GetAPI()->cuMemcpy2DPFN(&objInterpStr));
    return true;
}

template<class T> 
T& BufferManagerBase::GetCudaMemPtr(BufferType eBufferType, int index)
{
    return dynamic_cast<BufferManager<T>&>(*this).GetCudaMemPtr(eBufferType, index);
}

class FrameGeneratorCUDA : public FrameGenerator
{
public:
    virtual         ~FrameGeneratorCUDA();
    virtual bool    WaitForUploadToComplete();
    virtual bool    WaitForInterpolationToComplete();
    virtual void    SetActiveRenderTextureKey(uint64_t uiNewKeyValue);
    virtual void    SetActiveInterpolateTextureKey(uint64_t uiNewKeyValue);
    virtual uint64_t GetActiveRenderTextureKey();
    virtual uint64_t GetActiveInterpolateTextureKey();
    virtual bool    WaitForSyncObject(TextureType eTextureType);
    virtual bool    SignalSyncObject(TextureType eTextureType);
    void*           GetFenceObj() { return NULL; };
    bool            CreateTextureBuffer(  NvOFFRUCSurfaceFormat d3dSurfaceFormat);
    void            Destroy();
    bool            Render(
                        void** ppTexture,
                        double& dMillisec);
    bool            WriteOutput(
                        void* pTexture,
                        TextureType eTextureType);
    bool            GetResourceForInterpolation(
                        void** ppTexture,
                        double& dMillisec);
    bool            GetResource(
                        void** ppTexture,
                        uint32_t& uiCount);
    HRESULT CreateDevice(void** ppDevice);

protected:
    std::unique_ptr<BufferManagerBase> m_pBufferManagerBase;

};
