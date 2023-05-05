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
#include "FrameGenerator.h"
#include "Common.h"
#include "FreeImage.h"
#include "Bitmap.h"

#if defined _MSC_VER
#include <direct.h>
#include <Windows.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#define GETPROCEDUREADDRESS dlsym

bool IsPathExist(const std::string &s)
{
    struct stat buffer;
    return stat(s.c_str(), &buffer) == 0;
}
#endif

#ifndef GETPROCEDUREADDRESS
#define GETPROCEDUREADDRESS GetProcAddress
#endif

/**
 * Allocate host memory for a frame in YUV or PNG
 *
 * @param none
 * @return boolean
 */
bool FrameGenerator::AllocateInputCPUBuffer()
{
    if (m_InputFileType == InputFileTypeYUV)
    {
        m_InputCPUBuffer.reserve(size_t(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5));
        m_InputBufferSize = size_t(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5);
        return true;
    }
    else if (m_InputFileType == InputFileTypePNG)
    {
        m_InputCPUBuffer.reserve(size_t(m_FrameInfo.iRenderHeight * m_FrameInfo.iPitch));
        m_InputBufferSize = size_t(m_FrameInfo.iRenderHeight * m_FrameInfo.iPitch);
        return true;
    }
    return false;
}

/**
 * Gets the list of png's from strFilePath, sorts them alphabetically and returns the name of last file in the list
 *
 * @param path to the png files in string format
 * @return none
 */
void FrameGenerator::GetLastRenderedFileName(std::string strFilePath)
{
#if defined __GNUC__    
    const char dir_separators[] = "/";
#else
    const char dir_separators[] = "/\\";
#endif
    std::string strFileNameWithExt;
    std::string strFileName;
    std::string strInterpolated = "_interpolated.png";

    size_t pos = strFilePath.find_last_of(dir_separators);
    if (pos != std::string::npos)
    {
        strFileNameWithExt = strFilePath.substr(pos + 1);
        pos = strFileNameWithExt.find_last_of(".");
        if (pos != std::string::npos)
        {
            strFileName = strFileNameWithExt.substr(0, pos);
        }
        else
        {
            strFileName = strFileNameWithExt;
        }
    }
    else
    {
        strFileName = strFilePath;
    }
    m_strLastRenderFileName = strFileName + strInterpolated;
}

/**
 * Initializes input data structures which holds YUV data per frame
 *
 * @param path to the input png/yuv files in string format and start index in the file list
 * @return true if initialization was successfull else false
 */
bool FrameGenerator::CreateSource(
                        std::string strFileName, 
                        int iStartFrom)
{
    if (m_InputFileType == InputFileTypeYUV)
    {
        m_Source.open(strFileName, std::ios_base::binary);
        if (!m_Source.is_open())
        {
            std::cout << "Unable to open input file to simulate game. Please provide a valid YUV/ PNG file." << std::endl;
            return false;
        }
        size_t szYUVSize = (size_t)(m_FrameInfo.iRenderHeight * m_FrameInfo.iRenderWidth * 1.5);
        m_Source.seekg(szYUVSize * iStartFrom, m_Source.beg);
    }
    else if (m_InputFileType == InputFileTypePNG)
    {
        glob(strFileName, m_vecFileNames);
        if (m_vecFileNames.empty())
        {
            std::cout << "Invalid input File name/pattern: " << std::endl;
            return false;
        }
        m_FrameInfo.iPitch = GetPitch(m_vecFileNames[0]);
    }
    return true;
}

/**
 * Initializes output data structures used to write to disk
 *
 * @param path to the output png/yuv files in string format and input png/yuv files
 * @return true if initialization was successfull else false
 */
bool FrameGenerator::CreateSink(
                        std::string strOutputFileName, 
                        std::string strInputFileName)
{
#if defined _MSC_VER
    int mkdirsuccess = CreateDirectory(strOutputFileName.c_str(), NULL);

    if (GetLastError() == ERROR_PATH_NOT_FOUND)
    {
        std::cout << "Folder could not be created \n" << std::endl;
        return false;
    }
    else if (GetLastError() != ERROR_ALREADY_EXISTS)
    {
        std::cout << "Folder created \n" << std::endl;
    }

    if (m_InputFileType == InputFileTypeYUV)
    {
        std::string strOutFileName = strInputFileName.substr(strInputFileName.find_last_of("/\\") + 1);
        m_Output.open(strOutputFileName + "\\FRUC_" + strOutFileName, std::ofstream::out | std::ofstream::binary);
    }
#elif defined __GNUC__
    int mkdirsuccess = -1;
    if (IsPathExist(strOutputFileName) == false)
    {
        mkdirsuccess = mkdir(strOutputFileName.c_str(), 0777);
        if (mkdirsuccess != 0)
        {
            std::cout << "Folder could not be created \n" << std::endl;
            return false;
        }
        else
        {
            std::cout << "Folder created \n" << std::endl;
        }
    }
    else
    {
        std::cout << "Folder already exists\n" << std::endl;
    }

    if (m_InputFileType == InputFileTypeYUV)
    {
        std::string strOutFileName = strInputFileName.substr(strInputFileName.find_last_of("//") + 1);
        m_Output.open(strOutputFileName + "//FRUC_" + strOutFileName, std::ofstream::out | std::ofstream::binary);
    }
#endif
    m_strOutputDirectory = strOutputFileName;
    return true;
}

/**
 * Initializes framegenerator class, this class handles input and output data structures being sent to FRUC pipeline
 *
 * @param arguments class objects, this has members such as width, height, surfacetype etc.
 * @return true if initialization was successfull else false
 */
bool FrameGenerator::Init(Arguments args)
{
#define RETURN_ON_ERROR(bStatus) if (!bStatus) return bStatus;
    m_FrameInfo.iRenderWidth    = args.m_Width;
    m_FrameInfo.iRenderHeight   = args.m_Height;
    m_FrameInfo.iPitch          = m_FrameInfo.iRenderWidth;
    m_InputFileType             = args.m_InputFileType;
    m_iRenderIndex              = 0;
    m_iPNGVectorIndex           = args.m_StartFrame;
    m_dLastRenderTime           = 0;
    m_iFrameCnt                 = args.m_EndFrame;
    m_eSurfaceFormat            = (NvOFFRUCSurfaceFormat)args.m_InputSurfaceFormat;
    m_eResourceType             = (NvOFFRUCResourceType)args.m_ResourceType;
    m_cudaResourceType          = (NvOFFRUCCUDAResourceType)args.m_CudaResourceType;

#ifndef _MSC_VER
    if (m_eResourceType == DirectX11Resource)
    {
        std::cout << "Allocation Type DX is not supported on Linux OS" << std::endl;
        return false;
    }
#endif

    /*Some assumptions on window params*/
    const bool  FULL_SCREEN = false;
    const bool  VSYNC_ENABLED = false;
    const float SCREEN_DEPTH = 1000.0f;
    const float SCREEN_NEAR = 0.1f;

    HRESULT hr = CreateDevice(&m_pDevice);
    if (FAILED(hr))
    {
        return false;
    }

    FreeImage_Initialise();
    RETURN_ON_ERROR(CreateSource(args.m_InputName, args.m_StartFrame));
    RETURN_ON_ERROR(CreateSink(args.m_OutputName, args.m_InputName));
    RETURN_ON_ERROR(CreateTextureBuffer(m_eSurfaceFormat));
    RETURN_ON_ERROR(AllocateInputCPUBuffer());
    return true;
}

#if defined _MSC_VER

FrameGeneratorD3D11::~FrameGeneratorD3D11()
{
    Destroy();
}

/**
 * Destroys framegenerator class object
 *
 * @param none
 * @return none
 */
void FrameGeneratorD3D11::Destroy()
{
    FreeImage_DeInitialise();
    m_Source.close();
    m_Output.close();
}

/**
 * Returns the pointer to input resource, input resource is a DX texture and could be of type RENDER or INTERPOLATE
 *
 * @param none
 * @return none
 */
bool FrameGeneratorD3D11::GetResource(
                            void** ppTexture,
                            uint32_t& uiCount)
{
    uiCount = NUM_INTERPOLATE_TEXTURE + NUM_RENDER_TEXTURE;
    for (uint32_t i = 0; i < NUM_INTERPOLATE_TEXTURE; i++)
    {
        if (m_pInterpolateTexture2D[i])
        {
            ppTexture[i] = m_pInterpolateTexture2D[i];
        }
    }
    ppTexture = ppTexture + NUM_INTERPOLATE_TEXTURE;
    for (uint32_t i = 0; i < NUM_RENDER_TEXTURE; i++)
    {
        if (m_pRenderTexture2D[i])
        {
            ppTexture[i] = m_pRenderTexture2D[i];
        }
    }
    return true;
}

/**
 * Returns the direct3d object and swapchain
 *
 * @params pointer to array of input resources
 * @return S_OK if successfull
 */
HRESULT FrameGeneratorD3D11::CreateDevice(void** ppDevice)
{
    HRESULT hr = S_OK;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    UINT numFeatureLevels = ARRAYSIZE(featureLevels);
    hr = D3D11CreateDevice( 
                    NULL,
                    D3D_DRIVER_TYPE_HARDWARE,
                    NULL, 
                    0, 
                    featureLevels,
                    numFeatureLevels,
                    D3D11_SDK_VERSION,
                    &m_pDevice,
                    &featureLevel,
                    &m_pContext);
    if (FAILED(hr))
    {
        return hr;
    }

    *ppDevice = m_pDevice;

    ID3D11DeviceContext* pDeviceContext;
    m_pDevice->GetImmediateContext(&pDeviceContext);

    //Check if fence creation is supported else fallback to keyed mutex
    ID3D11Device5* m_pDevice5;
    hr = m_pDevice->QueryInterface<ID3D11Device5>(&m_pDevice5);
    if (FAILED(hr))
    {
        m_bFenceSupported = false;
        std::cout << "fence is not supported, falling back to keyed mutex" << std::endl;
        m_pFence = NULL;
        for (size_t i = 0; i < NUM_RENDER_TEXTURE; i++)
        {
            m_uiRenderKey[i] = 0;
        }
        for (size_t i = 0; i < NUM_INTERPOLATE_TEXTURE; i++)
        {
            m_uiInterpolateKey[i] = 0;
        }
    }
    else
    {
        m_bFenceSupported = true;
        hr = m_pDevice->QueryInterface<ID3D11Device5>(&m_pDevice5);
        if (FAILED(hr))
        {
            return hr;
        }

        hr = pDeviceContext->QueryInterface<ID3D11DeviceContext4>(&m_pDeviceContext4);
        if (FAILED(hr))
        {
            return hr;
        }

        hr = m_pDevice5->CreateFence(
                                0,
                                D3D11_FENCE_FLAG_SHARED,
                                IID_PPV_ARGS(&m_pFence));
        if (FAILED(hr))
        {
            return hr;
        }

        *ppDevice = m_pDevice5;

        m_hFenceEvent = CreateEvent(
                                nullptr,
                                FALSE,
                                FALSE,
                                nullptr);
        if (m_hFenceEvent == nullptr)
        {
            return E_APPLICATION_EXITING;
        }
    }
    return S_OK;
}

/**
 * Sets the current render texture as active and ready for interop
 *
 * @params key on which interop FRUC pipeline would wait for interop to finish
 * @return none
 */
void FrameGeneratorD3D11::SetActiveRenderTextureKey(uint64_t uiNewKeyValue)
{
    m_uiRenderKey[m_iRenderIndex] = uiNewKeyValue;
}

/**
 * Sets the current interpolate texture as active and ready for interop
 *
 * @params key on which interop FRUC pipeline would wait for interop to finish
 * @return none
 */
void FrameGeneratorD3D11::SetActiveInterpolateTextureKey(uint64_t uiNewKeyValue)
{
    m_uiInterpolateKey[0] = uiNewKeyValue;
}

/**
 * Gets the current render texture which was active and interop completed
 *
 * @params key on which interop FRUC pipeline waited for interop to finish
 * @return none
 */
uint64_t FrameGeneratorD3D11::GetActiveRenderTextureKey()
{
    return m_uiRenderKey[m_iRenderIndex];
}

/**
 * Gets the current interpolate texture which was active and interop completed
 *
 * @params key on which interop FRUC pipeline waited for interop to finish
 * @return none
 */
uint64_t FrameGeneratorD3D11::GetActiveInterpolateTextureKey()
{
    return m_uiInterpolateKey[0];
}

/**
 * Sample app waits on cpu thread while the pipeline is completing write to interpolate texture or pipeline is reading render texture 
 *
 * @params whether the type of texture is RENDER or INTERPOLATE
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::WaitForSyncObject(TextureType eTextureType)
{
    if (IsFenceSupported())
    {
        RETURN_ON_D3D_ERROR(m_pDeviceContext4->Wait(m_pFence, m_uiFenceValue++));
    }
    else
    {
        bool bIsRenderTexture = (eTextureType == TEXTURE_TYPE_RENDER) ? true : false;
        ID3D11Texture2D* pTexture = bIsRenderTexture ? m_pRenderTexture2D[m_iRenderIndex] : m_pInterpolateTexture2D[0];
        IDXGIKeyedMutex* pKeyedMutex = bIsRenderTexture ? m_pRenderTextureKeyedMutex[m_iRenderIndex] : m_pInterpolateTextureKeyedMutex[0];
        uint64_t m_key = bIsRenderTexture ? GetActiveRenderTextureKey() : GetActiveInterpolateTextureKey();

        HRESULT hr = (pKeyedMutex->AcquireSync(m_key++, INFINITE));
        if (hr == WAIT_ABANDONED || hr == WAIT_TIMEOUT)
        {
            std::cout << "Sync Object wait failed" << std::endl;
            return false;
        }
        bIsRenderTexture ? SetActiveRenderTextureKey(m_key) : SetActiveInterpolateTextureKey(m_key);
    }
    return true;
}

/**
 * Sample app signals the pipeline that RENDER or INTEROLATE texture is free to write
 *
 * @params whether the type of texture is RENDER or INTERPOLATE
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::SignalSyncObject(TextureType eTextureType)
{
    HRESULT hr = S_OK;
    if (IsFenceSupported())
    {
        hr = m_pDeviceContext4->Signal(m_pFence, m_uiFenceValue);
        if (FAILED(hr))
        {
            std::cout << "Release Sync failed during Signal " << std::endl;
            return false;
        }
    }
    else
    {
        bool bIsRenderTexture = (eTextureType == TEXTURE_TYPE_RENDER) ? true : false;
        ID3D11Texture2D* pTexture = bIsRenderTexture ? m_pRenderTexture2D[m_iRenderIndex] : m_pInterpolateTexture2D[0];
        IDXGIKeyedMutex* pKeyedMutex = bIsRenderTexture ? m_pRenderTextureKeyedMutex[m_iRenderIndex] : m_pInterpolateTextureKeyedMutex[0];
        uint64_t m_key = bIsRenderTexture ? GetActiveRenderTextureKey() : GetActiveInterpolateTextureKey();

        hr = (pKeyedMutex->ReleaseSync(m_key));
        if (FAILED(hr))
        {
            std::cout << "Release Sync failed during Object Signal Sync " << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * Copies data from CPU buffer to active render texture, once the copy is complete signal the pipeline that render texture is ready for interop 
 *
 * @params pointer to RENDER or INTERPOLATE texture and temporal distance to input frame
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::Render(
                            void** ppTexture, 
                            double& dRenderTime)
{
    m_iRenderIndex = (m_iRenderIndex + 1) % NUM_RENDER_TEXTURE;

    //get the key for the frame to be rendered
    uint64_t j = GetActiveRenderTextureIndex();
    D3D11_MAPPED_SUBRESOURCE  MappedResource;
    size_t InputBufferSize = GetInputBufferSize();

    RETURN_ON_D3D_ERROR(m_pContext->Map(m_pStagingTexture2D, 0, D3D11_MAP_WRITE, NULL, &MappedResource));

    if (m_InputFileType == InputFileTypeYUV)
    {
        if (!m_Source.read(m_InputCPUBuffer.data(), InputBufferSize))
        {
            if (m_iFrameCnt == MAX_FRAMECOUNT)
            {
                std::cout << "End of the input reached." << std::endl;
            }
            else
            {
                std::cout << "More frames has been requested than total frames in the file." << std::endl;
            }
            return false;
        }
        if (m_eSurfaceFormat == NV12Surface)
        {
            ConvertYUVPlanarToNV12( m_InputCPUBuffer.data(),
                                    (char*)MappedResource.pData,
                                    m_FrameInfo.iRenderWidth,
                                    m_FrameInfo.iRenderHeight,
                                    MappedResource.RowPitch);
        }
        else if (m_eSurfaceFormat == ARGBSurface)
        {
            ConvertYUVPlanarToBGR(  
                                m_InputCPUBuffer.data(),
                                (char*)MappedResource.pData,
                                m_FrameInfo.iRenderWidth,
                                m_FrameInfo.iRenderHeight,
                                MappedResource.RowPitch);
        }
    }
    else if (m_InputFileType == InputFileTypePNG)
    {
        if (m_iPNGVectorIndex >= m_vecFileNames.size())
        {
            return false;
        }

        FREE_IMAGE_FORMAT ImageFormat = FreeImage_GetFileType(m_vecFileNames[m_iPNGVectorIndex].c_str());
        if (ImageFormat != FIF_PNG)
        {
            return false;
        }

        FIBITMAP* pBitmap = FreeImage_Load(ImageFormat, m_vecFileNames[m_iPNGVectorIndex].c_str());
        if (pBitmap != NULL)
        {
            if (m_FrameInfo.iRenderWidth != FreeImage_GetWidth(pBitmap))
            {
                return false;
            }
            if (m_FrameInfo.iRenderHeight != FreeImage_GetHeight(pBitmap))
            {
                return false;
            }
            auto bpp = FreeImage_GetBPP(pBitmap);
            if (bpp == 24)
            {
                FIBITMAP* newBitmap = FreeImage_ConvertTo32Bits(pBitmap);
                FreeImage_Unload(pBitmap);
                pBitmap = newBitmap;
            }
            else if ((bpp != 32) && (bpp != 8))
            {
                return false;
            }
            void* pSysMem = FreeImage_GetBits(pBitmap);
            auto srcPitch = FreeImage_GetPitch(pBitmap);
            auto destPitch = m_FrameInfo.iPitch;
            for (int iRowIndex = 0; iRowIndex < m_FrameInfo.iRenderHeight; ++iRowIndex)
            {
                uint8_t* pSrcLine = FreeImage_GetScanLine(pBitmap, m_FrameInfo.iRenderHeight - iRowIndex - 1);
                memcpy((uint8_t*)m_InputCPUBuffer.data() + (iRowIndex * destPitch), pSrcLine, destPitch);
            }
            ConvertARGBToABGR(  
                        m_InputCPUBuffer.data(),
                        (char*)MappedResource.pData,
                        m_FrameInfo.iRenderWidth,
                        m_FrameInfo.iRenderHeight,
                        MappedResource.RowPitch);

            FreeImage_Unload(pBitmap);
            GetLastRenderedFileName(m_vecFileNames[m_iPNGVectorIndex]);
            m_iPNGVectorIndex++;
        }
        else
        {
            return false;
        }
    }

    m_pContext->Unmap(m_pStagingTexture2D, 0);

    D3D11_BOX       srcBox;
    srcBox.back     = 1;
    srcBox.front    = 0;
    srcBox.left     = 0;
    srcBox.top      = 0;
    srcBox.right    = m_FrameInfo.iRenderWidth;
    srcBox.bottom   = m_FrameInfo.iRenderHeight;

    WaitForSyncObject(TEXTURE_TYPE_RENDER);
    m_pContext->CopySubresourceRegion(
                    m_pRenderTexture2D[m_iRenderIndex],
                    0,
                    0,
                    0,
                    0,
                    m_pStagingTexture2D,
                    0,
                    &srcBox);
    SignalSyncObject(TEXTURE_TYPE_RENDER);

    *ppTexture = m_pRenderTexture2D[m_iRenderIndex];
    dRenderTime = m_dLastRenderTime + m_constdRenderInterval;
    m_dLastRenderTime = dRenderTime;
    return true;
}

/**
 * Copies data from CPU buffer to active render texture, once the copy is complete signal the pipeline that render texture is ready for interop
 *
 * @params pointer to RENDER or INTERPOLATE texture and temporal distance to input frame
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::GetResourceForInterpolation(
                            void** ppTexture,
                            double& interMilliSec)
{
    float fTimeOffset = 0;
    *ppTexture = m_pInterpolateTexture2D[0];
    interMilliSec = m_dLastRenderTime + (fTimeOffset - float(m_constdRenderInterval)) * DEFAULT_FRUC_SPEED;
    return true;
}

bool FrameGeneratorD3D11::WaitForUploadToComplete()
{
    //Map and Unmap is required to make sure upload of YUV data to the DX Surface is completed before calling FRUC Process
    //This is not required when we don't have to read from file
    D3D11_MAPPED_SUBRESOURCE  MappedResource;
    RETURN_ON_D3D_ERROR(m_pContext->Map(
                            m_pStagingTexture2D,
                            0,
                            D3D11_MAP_WRITE,
                            NULL,
                            &MappedResource));
    m_pContext->Unmap(m_pStagingTexture2D, 0);

    return true;
}

bool FrameGeneratorD3D11::WaitForInterpolationToComplete()
{
    //This is a dummy copy on GPU, purpose of it is to schedule a small copy of Interpolate Texture and have DX11 Map block on CPU for it to complete
    //This is added only to measure execution time in milliseconds of NvOFFRUCProcess() call and not required in normal operation
    D3D11_MAPPED_SUBRESOURCE  MappedResource;

    D3D11_BOX       srcBox;
    srcBox.back     = 1;
    srcBox.front    = 0;
    srcBox.left     = 0;
    srcBox.top      = 0;
    srcBox.right    = 4;
    srcBox.bottom   = 4;

    WaitForSyncObject(TEXTURE_TYPE_INTERPOLATE);
    m_pContext->CopySubresourceRegion(
                    m_pStagingTexture2D, 
                    0,
                    0,
                    0,
                    0,
                    (ID3D11Texture2D*)m_pInterpolateTexture2D[0],
                    0,
                    &srcBox);

    SignalSyncObject(TEXTURE_TYPE_INTERPOLATE);

    RETURN_ON_D3D_ERROR(m_pContext->Map(
                            m_pStagingTexture2D,
                            0,
                            D3D11_MAP_READ, 
                            NULL,
                            &MappedResource));
    m_pContext->Unmap(m_pStagingTexture2D, 0);

    return true;
}

/**
 * Create input DX texture, texture could be of type RENDER or INTERPOLATE
 *
 * @params whether texture type is RENDER or INTERPOLATE
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::CreateTextureBuffer(NvOFFRUCSurfaceFormat eSurfaceFormat)
{
    D3D11_TEXTURE2D_DESC desc   = { 0 };
    ZeroMemory(&desc, sizeof(desc));
    desc.Width                  = m_FrameInfo.iRenderWidth;
    desc.Height                 = m_FrameInfo.iRenderHeight;
    desc.MipLevels              = desc.ArraySize = 1;

    if (eSurfaceFormat == NV12Surface)
    {
        desc.Format = DXGI_FORMAT_NV12;
    }
    else
    {
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    desc.SampleDesc.Count = 1;

    if (m_bFenceSupported)
    {
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    }
    else
    {
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    }

    for (int iRenderTextureIndex = 0; iRenderTextureIndex < NUM_RENDER_TEXTURE; iRenderTextureIndex++)
    {
        RETURN_ON_D3D_ERROR(m_pDevice->CreateTexture2D(&desc, NULL, &m_pRenderTexture2D[iRenderTextureIndex]));
        if (!m_bFenceSupported)
        {
            IDXGIResource *dxgiResource = 0;
            RETURN_ON_D3D_ERROR(m_pRenderTexture2D[iRenderTextureIndex]->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&dxgiResource)));
            RETURN_ON_D3D_ERROR(dxgiResource->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&m_pRenderTextureKeyedMutex[iRenderTextureIndex])));
            RETURN_ON_D3D_ERROR(dxgiResource->Release());
        }
    }
    for (int interpolateTextureIndex = 0; interpolateTextureIndex < NUM_INTERPOLATE_TEXTURE; interpolateTextureIndex++)
    {
        RETURN_ON_D3D_ERROR(m_pDevice->CreateTexture2D(&desc, NULL, &m_pInterpolateTexture2D[interpolateTextureIndex]));
        if (!m_bFenceSupported)
        {
            IDXGIResource *dxgiResource = 0;
            RETURN_ON_D3D_ERROR(m_pInterpolateTexture2D[interpolateTextureIndex]->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&dxgiResource)));
            RETURN_ON_D3D_ERROR(dxgiResource->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&m_pInterpolateTextureKeyedMutex[interpolateTextureIndex])));
            RETURN_ON_D3D_ERROR(dxgiResource->Release());
        }
    }
    desc.MiscFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    RETURN_ON_D3D_ERROR(m_pDevice->CreateTexture2D(&desc, NULL, &m_pStagingTexture2D));

    return true;
}

/**
 * Copies the output from pipeline (i.e. interpolated frame) to CPU buffer and writes the CPU buffer to file
 *
 * @params pointer to RENDER or INTERPOLATE texture and whether the type of texture is RENDER or INTERPOLATE
 * @return returns true is successfull
 */
bool FrameGeneratorD3D11::WriteOutput(void* pTexture, TextureType eTextureType)
{
    D3D11_MAPPED_SUBRESOURCE  MappedResource;
    std::fill(m_InputCPUBuffer.begin(), m_InputCPUBuffer.end(), 0);
    bool bIsRenderTexture = (eTextureType == TEXTURE_TYPE_RENDER) ? true : false;

    D3D11_BOX       srcBox;
    srcBox.back     = 1;
    srcBox.front    = 0;
    srcBox.left     = 0;
    srcBox.top      = 0;
    srcBox.right    = m_FrameInfo.iRenderWidth;
    srcBox.bottom   = m_FrameInfo.iRenderHeight;

    WaitForSyncObject(eTextureType);
    m_pContext->CopySubresourceRegion(
                    m_pStagingTexture2D,
                    0,
                    0,
                    0,
                    0,
                    (ID3D11Texture2D*)pTexture,
                    0,
                    &srcBox);
    SignalSyncObject(eTextureType);

    HRESULT hr = S_OK;
    RETURN_ON_D3D_ERROR(m_pContext->Map(
                            m_pStagingTexture2D,
                            0,
                            D3D11_MAP_READ,
                            NULL,
                            &MappedResource));

    if (m_eSurfaceFormat == NV12Surface)
    {
        ConvertNV12ToYUVPlanar(
                    (char*)MappedResource.pData,
                    m_InputCPUBuffer.data(),
                    m_FrameInfo.iRenderWidth,
                    m_FrameInfo.iRenderHeight,
                    MappedResource.RowPitch);

        m_Output.write(
                m_InputCPUBuffer.data(),
                std::streamsize(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5));
    }
    else if (m_eSurfaceFormat == ARGBSurface)
    {
        if (m_InputFileType == InputFileTypeYUV)
        {
            ConvertBGRToYUVPlanar(
                (char*)MappedResource.pData,
                m_InputCPUBuffer.data(),
                m_FrameInfo.iRenderWidth,
                m_FrameInfo.iRenderHeight, 
                MappedResource.RowPitch);

            m_Output.write(
                m_InputCPUBuffer.data(),
                std::streamsize(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5));
        }
        else if (!bIsRenderTexture)
        {
            const int BPP = 32;
            FreeImage_Initialise();
            FIBITMAP* bitmap = FreeImage_Allocate(
                                    m_FrameInfo.iRenderWidth,
                                    m_FrameInfo.iRenderHeight,
                                    BPP);

            ConvertABGRToARGB(
                (char*)MappedResource.pData,
                m_InputCPUBuffer.data(),
                m_FrameInfo.iRenderWidth, 
                m_FrameInfo.iRenderHeight, 
                MappedResource.RowPitch);

            struct ARGBPixel
            {
                unsigned char ucBlue;
                unsigned char ucGreen;
                unsigned char ucRed;
                unsigned char ucAlpha;
            };

            ARGBPixel* pARGB = (ARGBPixel*)m_InputCPUBuffer.data();
            for (int32_t iFrameHeight = 0; iFrameHeight < m_FrameInfo.iRenderHeight; ++iFrameHeight)
            {
                for (int32_t iFrameWidth = 0; iFrameWidth < m_FrameInfo.iRenderWidth; ++iFrameWidth)
                {
                    RGBQUAD objARB;
                    objARB.rgbReserved = 0xFF;
                    int outputIndex = iFrameWidth + iFrameHeight * m_FrameInfo.iRenderWidth;
                    objARB.rgbBlue = pARGB[outputIndex].ucBlue;
                    objARB.rgbGreen = pARGB[outputIndex].ucGreen;
                    objARB.rgbRed = pARGB[outputIndex].ucRed;
                    FreeImage_SetPixelColor(bitmap, iFrameWidth, m_FrameInfo.iRenderHeight - 1 - iFrameHeight, &objARB);
                }
            }

            std::string strOutputPath = m_strOutputDirectory + "\\" + m_strLastRenderFileName;
            FreeImage_Save(FIF_PNG, bitmap, strOutputPath.c_str(), PNG_DEFAULT);
            FreeImage_Unload(bitmap);
            FreeImage_DeInitialise();
        }
    }
    m_pContext->Unmap(m_pStagingTexture2D, 0);
    return true;
}

#endif

FrameGeneratorCUDA::~FrameGeneratorCUDA()
{
    m_pBufferManagerBase->DestroyBuffers();
    Destroy();
}

void FrameGeneratorCUDA::Destroy()
{
    FreeImage_DeInitialise();
    m_Source.close();
    m_Output.close();
}

/**
 * Gets the pointer to array of cuda resources, cuda resource array could be of type RENDER or INTERPOLATE
 *
 * @params pointer to RENDER or INTERPOLATE texture and temporal distance to input frame
 * @return returns true is successfull
 */
bool FrameGeneratorCUDA::GetResource(void** ppTexture, uint32_t& uiCount)
{
    uiCount = NUM_INTERPOLATE_TEXTURE + NUM_RENDER_TEXTURE;
    for (uint32_t i = 0; i < NUM_INTERPOLATE_TEXTURE; i++)
    {
        switch (m_cudaResourceType)
        {
            case CudaResourceTypeUndefined:
            {
                assert(0);
            }
            break;
            case CudaResourceCuDevicePtr:
            {
                ppTexture[i] = &m_pBufferManagerBase->GetCudaMemPtr<CUdeviceptr>(INTERPOLATED_DEVPTR, i);
            }
            break;
            case CudaResourceCuArray:
            {
                ppTexture[i] = m_pBufferManagerBase->GetCudaMemPtr<CUarray>(INTERPOLATED_DEVPTR, i);
            }
            break;
            default:
            {
                break;
            }
        }        
    }

    ppTexture = ppTexture + NUM_INTERPOLATE_TEXTURE;
    for (uint32_t i = 0; i < NUM_RENDER_TEXTURE; i++)
    {
        switch (m_cudaResourceType)
        {
            case CudaResourceTypeUndefined:
                break;
            case CudaResourceCuDevicePtr:
            {
                ppTexture[i] = &m_pBufferManagerBase->GetCudaMemPtr<CUdeviceptr>(RENDER_DEVPTR, i);
            }
            break;
            case CudaResourceCuArray:
            {
                ppTexture[i] = m_pBufferManagerBase->GetCudaMemPtr<CUarray>(RENDER_DEVPTR, i);
            }
            break;
            default:
                break;
        }
    }
    return true;
}

/**
 * Creates object of class buffermanager which handles creation, read-write and destruction of cuda input surfaces which could be of type cuDevicePtr or cuArray
 *
 * @params pointer to array of cuda input memory
 * @return returns true is successfull
 */
HRESULT FrameGeneratorCUDA::CreateDevice(void** ppDevice)
{
    *ppDevice = NULL;
    CUdevice   objCuDevice;
    CUcontext  objCuContext;
    int iDeviceCount = 0;

    float fFactor = (float)((m_eSurfaceFormat == NV12Surface) ? 1.5 : 4);
    
    switch (m_cudaResourceType)
    {
        
        case CudaResourceCuDevicePtr:
        {
            m_pBufferManagerBase.reset(
                                    new BufferManager<CUdeviceptr>(
                                        m_FrameInfo.iRenderWidth,
                                        m_FrameInfo.iRenderHeight,
                                        (int)(m_FrameInfo.iRenderWidth * fFactor),
                                        m_eSurfaceFormat));
        }
        break;
        case CudaResourceCuArray:
        {
            if (m_eSurfaceFormat == NV12Surface)
            {
                m_pBufferManagerBase.reset(
                                new BufferManager<CUarray>(  
                                    m_FrameInfo.iRenderWidth,
                                    (int)(m_FrameInfo.iRenderHeight * fFactor),
                                    m_FrameInfo.iRenderWidth,
                                    m_eSurfaceFormat));
            }
            else if (m_eSurfaceFormat == ARGBSurface)
            {
                m_pBufferManagerBase.reset(
                            new BufferManager<CUarray>(
                                    m_FrameInfo.iRenderWidth,
                                    m_FrameInfo.iRenderHeight,
                                    (int)(m_FrameInfo.iRenderWidth * fFactor),
                                    m_eSurfaceFormat));
            }
            break;
        }
        default:
        {
            break;
        }
    }
         
    CUresult err = m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuInitPFN(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
    {
        checkCudaErrors(m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuDeviceGetCountPFN(&iDeviceCount));
    }

    if (iDeviceCount == 0)
    {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA objCuDevice
    checkCudaErrors(m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuDeviceGetPFN(&objCuDevice, 0));

    //set the objCuContext to NULL
    err = m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuCtxCreatePFN(&objCuContext, 0, objCuDevice);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA objCuContext.\n");
        m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuCtxDestroyPFN(objCuContext);
        exit(-1);
    }

    checkCudaErrors(m_pBufferManagerBase->getCUDADrvAPIHandle()->GetAPI()->cuCtxSetCurrentPFN(objCuContext));

    return S_OK;
}

void FrameGeneratorCUDA::SetActiveRenderTextureKey(uint64_t uiNewKeyValue)
{

}

void FrameGeneratorCUDA::SetActiveInterpolateTextureKey(uint64_t uiNewKeyValue)
{

}

uint64_t FrameGeneratorCUDA::GetActiveRenderTextureKey()
{
    return 0;
}

uint64_t FrameGeneratorCUDA::GetActiveInterpolateTextureKey()
{
    return 0;
}

bool FrameGeneratorCUDA::WaitForSyncObject(TextureType eTextureType)
{

    return true;
}

bool FrameGeneratorCUDA::SignalSyncObject(TextureType eTextureType)
{
    HRESULT hr = S_OK;

    return true;
}


/**
 * Copies data from host memory (input CPU buffer) to device memory in Client
 *
 * @params pointer to RENDER or INTERPOLATE texture and temporal distance to input frame
 * @return returns true is successfull
 */
bool FrameGeneratorCUDA::Render(void** ppTexture, double& dRenderTime)
{
    m_iRenderIndex = (m_iRenderIndex + 1) % NUM_RENDER_TEXTURE;
    size_t szInputBufferSize = GetInputBufferSize();
    if (m_InputFileType == InputFileTypeYUV)
    {
        if (!m_Source.read(m_InputCPUBuffer.data(), szInputBufferSize))
        {
            if (m_iFrameCnt == MAX_FRAMECOUNT)
            {
                std::cout << "End of the input reached." << std::endl;
            }
            else
            {
                std::cout << "More frames has been requested than total frames in the file." << std::endl;
            }
            return false;
        }
        if (m_eSurfaceFormat == NV12Surface)
        {
            ConvertYUVPlanarToNV12(
                    m_InputCPUBuffer.data(),
                    (char*)m_pBufferManagerBase->GetHostPointer(RENDER_HOST, m_iRenderIndex),
                    m_FrameInfo.iRenderWidth,   
                    m_FrameInfo.iRenderHeight,
                    m_FrameInfo.iPitch);
        }
        else if (m_eSurfaceFormat == ARGBSurface)
        {
            ConvertYUVPlanarToBGR(
                    m_InputCPUBuffer.data(),
                    (char*)m_pBufferManagerBase->GetHostPointer(RENDER_HOST, m_iRenderIndex),
                    m_FrameInfo.iRenderWidth,
                    m_FrameInfo.iRenderHeight,
                    m_FrameInfo.iPitch * 4);
        }
    }
    else if (m_InputFileType == InputFileTypePNG)
    {
        if (m_iPNGVectorIndex >= m_vecFileNames.size())
        {
            return false;
        }

        FREE_IMAGE_FORMAT ImageFormat = FreeImage_GetFileType(m_vecFileNames[m_iPNGVectorIndex].c_str());
        if (ImageFormat != FIF_PNG)
        {
            return false;
        }
        FIBITMAP* pBitmap = FreeImage_Load(ImageFormat, m_vecFileNames[m_iPNGVectorIndex].c_str());
        if (pBitmap)
        {
            if (m_FrameInfo.iRenderWidth != FreeImage_GetWidth(pBitmap))
            {
                return false;
            }
            if (m_FrameInfo.iRenderHeight != FreeImage_GetHeight(pBitmap))
            {
                return false;
            }
            auto bpp = FreeImage_GetBPP(pBitmap);
            if (bpp == 24)
            {
                FIBITMAP* newBitmap = FreeImage_ConvertTo32Bits(pBitmap);
                FreeImage_Unload(pBitmap);
                pBitmap = newBitmap;
            }
            else if ((bpp != 32) && (bpp != 8))
            {
                return false;
            }
            void* pSysMem = FreeImage_GetBits(pBitmap);
            auto srcPitch = FreeImage_GetPitch(pBitmap);
            auto destPitch = m_FrameInfo.iPitch;
            for (int iWidthIndex = 0; iWidthIndex < m_FrameInfo.iRenderHeight; ++iWidthIndex)
            {
                uint8_t* pSrcLine = FreeImage_GetScanLine(pBitmap, m_FrameInfo.iRenderHeight - iWidthIndex - 1);
                memcpy((uint8_t*)m_InputCPUBuffer.data() + (iWidthIndex * destPitch), pSrcLine, destPitch);
            }
            ConvertARGBToABGR(
                    m_InputCPUBuffer.data(),
                    (char*)m_pBufferManagerBase->GetHostPointer(RENDER_HOST, m_iRenderIndex),
                    m_FrameInfo.iRenderWidth,
                    m_FrameInfo.iRenderHeight,
                    m_FrameInfo.iPitch);

            FreeImage_Unload(pBitmap);
            GetLastRenderedFileName(m_vecFileNames[m_iPNGVectorIndex]);

            m_iPNGVectorIndex++;
        }
    }

    m_pBufferManagerBase->Render(m_iRenderIndex, RENDER_DEVPTR);

    switch (m_cudaResourceType)
    {
        case CudaResourceCuDevicePtr:
        {
            *ppTexture = (void*)&m_pBufferManagerBase->GetCudaMemPtr<CUdeviceptr>(RENDER_DEVPTR, m_iRenderIndex);
        }
        break;
        case CudaResourceCuArray:
        {
            *ppTexture = (void*)m_pBufferManagerBase->GetCudaMemPtr<CUarray>(RENDER_DEVPTR, m_iRenderIndex);
        }
        break;
        default:
            break;
    }
    
    dRenderTime = m_dLastRenderTime + m_constdRenderInterval;
    m_dLastRenderTime = dRenderTime;
    return true;
}

bool FrameGeneratorCUDA::GetResourceForInterpolation(void** ppTexture, double& dInterMilliSec)
{
    float fTimeOffset = 0;
    switch (m_cudaResourceType)
    {
        case CudaResourceCuDevicePtr:
        {
            *ppTexture = &m_pBufferManagerBase->GetCudaMemPtr<CUdeviceptr>(INTERPOLATED_DEVPTR, 0);
        }
        break;
        case CudaResourceCuArray:
        {
            *ppTexture = m_pBufferManagerBase->GetCudaMemPtr<CUarray>(INTERPOLATED_DEVPTR, 0);
        }
        break;
        default:
            break;
    }
    dInterMilliSec = m_dLastRenderTime + (fTimeOffset - float(m_constdRenderInterval)) * DEFAULT_FRUC_SPEED;
    return true;
}

bool FrameGeneratorCUDA::WaitForUploadToComplete()
{
    return true;
}

bool FrameGeneratorCUDA::WaitForInterpolationToComplete()
{
    return true;
}

bool FrameGeneratorCUDA::CreateTextureBuffer(NvOFFRUCSurfaceFormat eSurfaceFormat)
{
    bool bResult = m_pBufferManagerBase->CreateTextureBuffer();
    return bResult;
}

bool FrameGeneratorCUDA::WriteOutput(void* pTextureDevice, TextureType eTextureType)
{

    memset(m_InputCPUBuffer.data(), 0, GetInputBufferSize());
    bool bIsRenderTexture = (eTextureType == TEXTURE_TYPE_RENDER) ? true : false;

    HRESULT hr = S_OK;
    m_pBufferManagerBase->WriteOutput(m_iRenderIndex, (eTextureType == TEXTURE_TYPE_RENDER) ? RENDER_DEVPTR : INTERPOLATED_DEVPTR);

    void* pTexture = m_pBufferManagerBase->GetHostPointer(INTERPOLATED_HOST, 0);

    if (m_eSurfaceFormat == NV12Surface)
    {
        ConvertNV12ToYUVPlanar(
            (char*)pTexture, 
            m_InputCPUBuffer.data(),
            m_FrameInfo.iRenderWidth,
            m_FrameInfo.iRenderHeight,
            m_FrameInfo.iPitch);

        m_Output.write(
            m_InputCPUBuffer.data(),
            std::streamsize(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5));
    }
    else if (m_eSurfaceFormat == ARGBSurface)
    {
        if (m_InputFileType == InputFileTypeYUV)
        {
            ConvertBGRToYUVPlanar(
                (char*)pTexture,
                m_InputCPUBuffer.data(),
                m_FrameInfo.iRenderWidth,
                m_FrameInfo.iRenderHeight,
                m_FrameInfo.iPitch * 4);

            m_Output.write(
                m_InputCPUBuffer.data(),
                std::streamsize(m_FrameInfo.iRenderWidth * m_FrameInfo.iRenderHeight * 1.5));
        }
        else if (!bIsRenderTexture)
        {
            const int BPP = 32;
            FreeImage_Initialise();
            FIBITMAP* bitmap = FreeImage_Allocate(
                                    m_FrameInfo.iRenderWidth, 
                                    m_FrameInfo.iRenderHeight,
                                    BPP);

            int iPitchCorrectionFactor = 4;
            if (m_InputFileType == InputFileTypePNG)
            {
                iPitchCorrectionFactor = 1;
            }
            ConvertABGRToARGB(
                    (char*)pTexture,
                    m_InputCPUBuffer.data(),
                    m_FrameInfo.iRenderWidth,
                    m_FrameInfo.iRenderHeight,
                    m_FrameInfo.iPitch * iPitchCorrectionFactor);

            struct ARGBPixel
            {
                unsigned char ucBlue;
                unsigned char ucGreen;
                unsigned char ucRed;
                unsigned char ucAlpha;
            };

            ARGBPixel* pARGB = (ARGBPixel*)m_InputCPUBuffer.data();
            for (int32_t iRowIndex = 0; iRowIndex < m_FrameInfo.iRenderHeight; ++iRowIndex)
            {
                for (int32_t iColumnIndex = 0; iColumnIndex < m_FrameInfo.iRenderWidth; ++iColumnIndex)
                {
                    RGBQUAD objRGBQuad;
                    objRGBQuad.rgbReserved = 0xFF;

                    int outputIndex = iColumnIndex + iRowIndex * m_FrameInfo.iRenderWidth;
                    objRGBQuad.rgbBlue = pARGB[outputIndex].ucBlue;
                    objRGBQuad.rgbGreen = pARGB[outputIndex].ucGreen;
                    objRGBQuad.rgbRed = pARGB[outputIndex].ucRed;
                    FreeImage_SetPixelColor(bitmap, iColumnIndex, m_FrameInfo.iRenderHeight - 1 - iRowIndex, &objRGBQuad);
                }
            }

            std::string OutputPath = m_strOutputDirectory + "\\" + m_strLastRenderFileName;
#ifndef _MSC_VER
            std::replace(OutputPath.begin(), OutputPath.end(), '\\', '/');
#endif
            FreeImage_Save(
                FIF_PNG,
                bitmap,
                OutputPath.c_str(),
                PNG_DEFAULT);

            FreeImage_Unload(bitmap);
            FreeImage_DeInitialise();
        }
    }
    return true;
}
