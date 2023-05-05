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
#include "Arguments.h"

#define DIR_SEP "\\"
#define DIR_NAME_PREFIX "C:"

void Arguments::CreateParser()
{
    m_CmdParser.AddOptions("input", m_InputName, "Input name "
        "[ e.g. " DIR_NAME_PREFIX DIR_SEP "input*.png, "
        DIR_NAME_PREFIX DIR_SEP "input_wxh.yuv ],"
        " Path must be Absolute");
    m_CmdParser.AddOptions("width", m_Width, "Input frame width");
    m_CmdParser.AddOptions("height", m_Height, "Input frame height");
    m_CmdParser.AddOptions("output", m_OutputName, "Output directory to store results/ debug helper files, Path could be Absolute or Relative");
    m_CmdParser.AddOptions("surfaceformat", m_InputSurfaceFormat, "Input Surface format : Set 0 for NV12, 1 for ABGR");
    m_CmdParser.AddOptions("startframe", m_StartFrame, "Start processing from this frame index");
    m_CmdParser.AddOptions("endframe", m_EndFrame, "Process till this frame index");
    m_CmdParser.AddOptions("allocationtype", m_ResourceType, "Specify 0 to create CUDA and 1 to create DX allocations");
    m_CmdParser.AddOptions("cudasurfacetype", m_CudaResourceType, "Specify 0 to create cuDevicePtr and 1 for cuArray, please note this option takes effect if client wants to use CUDA code path");

}

NvOFFRUCInputFileType Arguments::GetInputFileTypeFromFilename()
{
    std::string YUV("YUV");
    std::string PNG("PNG");

    size_t nFileNameLength = m_InputName.rfind('.', m_InputName.length());
    if (nFileNameLength != std::string::npos) {
        std::string strFileExtension = m_InputName.substr(nFileNameLength + 1, m_InputName.length() - nFileNameLength);
        transform(strFileExtension.begin(), strFileExtension.end(), strFileExtension.begin(), ::toupper);

        if (strFileExtension.compare(YUV) == 0)
        {
            return InputFileTypeYUV;
        }
        else if (strFileExtension.compare(PNG) == 0)
        {
            return InputFileTypePNG;
        }
    }
    return InputFileTypeUndefined;
}

bool Arguments::ParseArguments(int argc, const char* argv[])
{
    NVOFFRUC_ARGS_PARSE(m_CmdParser, argc, (const char**)argv);

    //Checking if Arguments are valid
    if (m_InputName.empty())
    {
        return false;
    }

    if (m_InputSurfaceFormat < NV12Surface)
    {
        std::cerr << "Unsupported surface format specified." << std::endl;
        return false;
    }

    m_InputFileType = GetInputFileTypeFromFilename();
    switch (m_InputFileType)
    {
        case InputFileTypeUndefined:
        {
            std::cerr << "Unsupported input file type specified." << std::endl;
            return false;
        }
        break;
        case InputFileTypePNG:
        {
            m_InputSurfaceFormat = ARGBSurface; //forcing ARGBSurface as input here
        }
        break;
        
    }

    if (m_ResourceType == DirectX11Resource)
    {
        m_CudaResourceType = CudaResourceTypeUndefined;
    }

    return true;
}