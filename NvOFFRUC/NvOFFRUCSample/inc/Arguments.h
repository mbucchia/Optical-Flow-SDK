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
#include "CMDParser.h"
#include <assert.h>
#include <limits.h>
#include "../../Interface/NvOFFRUC.h"

#if defined _MSC_VER
#include <windows.h>
#endif

#define MAX_FRAMECOUNT INT_MAX


class Arguments
{
public:
    Arguments()
    {
        m_Width                 = 1920;
        m_Height                = 1080;
        m_EndFrame              = MAX_FRAMECOUNT;//if the user does not specify framecnt, INT_MAX will ensure that we run FRUC till end of file
        m_StartFrame            = 0;
        m_InputFileType         = InputFileTypeYUV;
        m_ResourceType          = CudaResource;
        m_CudaResourceType      = CudaResourceCuDevicePtr;
        m_InputSurfaceFormat    = NV12Surface;
        CreateParser();
    }
    bool ParseArguments(int argc, const char* argv[]);
    void CreateParser();
    NvOFFRUCInputFileType GetInputFileTypeFromFilename();

    CmdParser           m_CmdParser;
    std::string         m_InputName;
    std::string         m_OutputName;
    int                 m_D3DLevel;
    NvOFFRUCInputFileType m_InputFileType;
    int                 m_Width;
    int                 m_Height;
    int                 m_EndFrame;
    int                 m_StartFrame;
    int                 m_ResourceType;
    int                 m_CudaResourceType;
    int                 m_InputSurfaceFormat;
};