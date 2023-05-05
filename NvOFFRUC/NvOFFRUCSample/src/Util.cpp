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
#define NOMINMAX

#if defined _MSC_VER
    #include <windows.h>
#elif defined __GNUC__
    #include <dirent.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    typedef unsigned char byte;
#endif  

#include <regex>
#include <iostream>
#include <sstream>
#include "FreeImage.h"
#include "Common.h"

template<DXGI_FORMAT FORMAT>
void Save(
        char* strFilename,
        BYTE* byData,
        DWORD iWidth,
        DWORD iHeight, DWORD iRowPitch)
{
    if (FORMAT == DXGI_FORMAT_NV12)
    {
        SaveNV12(strFilename, byData, iWidth, iHeight, iRowPitch);
    }
    else if (FORMAT == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        SaveARGB(strFilename, byData, iWidth, iHeight, iRowPitch / 4);
    }
    else
    {
        std::cout << "unsupported format" << std::endl;
    }
}

std::string GenerateRegexPattern(const std::string& strImageNamePattern)
{
    std::string strRegex_Pat;
    std::string strImage;
    std::string temp;

    for (auto it = strImageNamePattern.cbegin(); it != strImageNamePattern.cend(); ++it)
    {
        if (*it == '*')
        {
            strImage.append(".*");
        }
        else if (*it == '?')
        {
            strImage.append(".");
        }
        else
        {
            strImage.append(1, *it);
        }
    }

    size_t pos = strImage.find_first_of("%");
    if (pos != std::string::npos)
    {
        if (pos > 0)
        {
            strRegex_Pat.append(strImage.substr(0, pos));
        }
        temp = strImage.substr(pos + 1);
        pos = temp.find_first_of("d");
        if (pos != std::string::npos)
        {
            if (pos > 0)
            {
                auto nd = atoi(temp.substr(0, pos).c_str());
                std::ostringstream ss;
                ss << "([0-9]){" << nd << ",}";
                strRegex_Pat.append(ss.str());
            }
            else
            {
                strRegex_Pat.append("([0 - 9]){1,}");
            }
            strRegex_Pat.append(temp.substr(pos + 1));
        }
    }
    else
    {
        strRegex_Pat.append(strImage);
    }
    return strRegex_Pat;
}

std::vector<std::pair<std::string, std::string>> ReadDirectory(const std::string& strPath)
{
    std::vector<std::pair<std::string, std::string>> vecFiles;
#ifdef _WIN32
    WIN32_FIND_DATAA file;
    std::string filter(strPath + "//*");
    HANDLE searchHandle = FindFirstFileExA(filter.c_str(),
        FindExInfoStandard, &file, FindExSearchNameMatch, NULL, 0);
    if (searchHandle != INVALID_HANDLE_VALUE)
    {
        do
        {
            const char* strName = file.cFileName;
            if ((strName[0] == 0) ||
                (strName[0] == '.' && strName[1] == 0) ||
                (strName[0] == '.' && strName[1] == '.' && strName[2] == 0))
                continue;

            if ((file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
                continue;

            vecFiles.push_back(std::make_pair(strPath + "\\" + std::string(strName), std::string(strName)));
        } while (FindNextFileA(searchHandle, &file));
        FindClose(searchHandle);
    }
#else
    DIR* d;
    struct dirent* dir;
    d = opendir(strPath.c_str());
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            const char* strName = dir->d_name;
            if ((strName[0] == 0) ||
                (strName[0] == '.' && strName[1] == 0) ||
                (strName[0] == '.' && strName[1] == '.' && strName[2] == 0))
                continue;

            struct stat buf;
            if ((stat(strName, &buf) == 0) &&
                S_ISDIR(buf.st_mode))
                continue;

            vecFiles.push_back(std::make_pair(strPath + "/" + std::string(strName), std::string(strName)));
        }

        closedir(d);
    }
#endif

    return vecFiles;
}

void glob(
        const std::string& strImage, 
        std::vector<std::string>& vecStringResult)
{
    const char dir_separators[] = "/\\";
    std::string strWildChart;
    std::string strPath;
    size_t pos = strImage.find_last_of(dir_separators);
    if (pos == std::string::npos)
    {
        strWildChart = strImage;
        strPath = ".";
    }
    else
    {
        strPath = strImage.substr(0, pos);
        strWildChart = strImage.substr(pos + 1);
    }
    std::string strRegex = GenerateRegexPattern(strWildChart);
    std::regex strRegex_Pat{ strRegex };
#ifndef NDEBUG
    std::cout << "strInput file directory strPath : " << strPath << std::endl;
    std::cout << "strInput file pattern : " << strWildChart << std::endl;
#endif
    std::vector<std::pair<std::string, std::string>> vec_pair_FileNames = ReadDirectory(strPath);
    for (const auto& p : vec_pair_FileNames)
    {
        if (!p.first.empty() && !p.second.empty())
        {
            auto strFilename = p.second;
            if (!strWildChart.empty())
            {
                if (regex_match(strFilename, strRegex_Pat))
                {
                    vecStringResult.push_back(p.first);
                }
            }
        }
    }

    if (!vecStringResult.empty())
    {
        std::sort(vecStringResult.begin(), vecStringResult.end());
    }
}

int GetPitch(const std::string& strFilename)
{
    int iWidth = 0;
    int iHeight = 0;
    int iPitch = 0;
    int iBPP = 0;

    FREE_IMAGE_FORMAT imageFormat = FreeImage_GetFileType(strFilename.c_str());
    if (imageFormat != FIF_PNG)
    {
        return 0;
    }
    FIBITMAP* pBitmap = FreeImage_Load(imageFormat, strFilename.c_str(), FIF_LOAD_NOPIXELS);
    if (pBitmap)
    {
        iWidth = FreeImage_GetWidth(pBitmap);
        iHeight = FreeImage_GetHeight(pBitmap);
        iBPP = FreeImage_GetBPP(pBitmap);
        FreeImage_Unload(pBitmap);
        if ((iBPP != 32) && (iBPP != 24) && (iBPP != 8))
        {
            return 0;
        }
        
        if ((iBPP == 32) || (iBPP == 24))
        {
            iPitch = iWidth * 4;
        }
        else
        {
            iPitch = iWidth;
        }
    }
    return iPitch;
}

// BT709-Full
void ConvertNV12ToYUVPlanar(
            char* strInput, 
            char* strOutput, 
            int iWidth, 
            int iHeight, 
            int iRowPitch)
{
    struct chroma
    {
        char u;
        char v;
    };
       
    for (int i = 0; i < iHeight; ++i)
    {
        memcpy(strOutput, strInput, iWidth);
        strOutput += iWidth;
        strInput += iRowPitch;
    }
    char* uOutput = (char*)(strOutput);
    char* vOutput = (char*)(uOutput + iHeight * iWidth / 4);
    chroma* uvInput = (chroma* )strInput;
    for (int y = 0; y < iHeight / 2; y++)
    {
        for (int x = 0; x < iWidth / 2; x++)
        {
            *(uOutput + x + y * iWidth / 2) = (uvInput + x + y * iRowPitch / 2)->u;
            *(vOutput + x + y * iWidth / 2) = (uvInput + x + y * iRowPitch / 2)->v;
        }
    }

}

void ConvertYUVPlanarToNV12(
            char* strInput, 
            char* strOutput, 
            int iWidth, 
            int iHeight, 
            int iRowPitch)
{
    for (int i = 0; i < iHeight; i++)
    {
        memcpy(strOutput, strInput, iWidth);
        strInput = strInput + iWidth;
        strOutput = strOutput + iRowPitch;
    }
    char* uvOutput = (strOutput);
    char* uInput = strInput;
    char* vInput = uInput + iHeight * iWidth / 4;
    for (int y = 0; y < iHeight / 2; y++)
    {
        for (int x = 0; x < iWidth / 2; x++)
        {
            int outputIndex = 2 * x + y * iRowPitch;
            int uIndex = outputIndex;
            int vIndex = uIndex + 1;
            *(uvOutput + uIndex) = *(uInput + x + y * iWidth / 2);
            *(uvOutput + vIndex) = *(vInput + x + y * iWidth / 2);
        }
    }

}

void ConvertYUVPlanarToBGR(
            char* strInput, 
            char* strOutput, 
            int iWidth, 
            int iHeight, 
            int outRowPitch)
{
    struct ABGR
    {
        unsigned char R;
        unsigned char G;
        unsigned char B;
        unsigned char A;
    };
    byte* yInput = (byte*)strInput;
    byte* uInput = yInput + iWidth * iHeight;
    byte* vInput = uInput + iHeight * iWidth / 4;
    ABGR* argbOutput = (ABGR*)(strOutput);
    for (int y = 0; y < iHeight; y++)
    {
        for (int x = 0; x < iWidth; x++)
        {
            int outputIndex = x + y * outRowPitch / 4;
            int uvIndex = x / 2 + y / 2 * iWidth / 2;
            int yIndex = x + y * iWidth;
            ABGR argb;
            argb.R = FRUC_CLAMP_255((int)(1.164 * (yInput[yIndex] - 16) + 0.000 * (uInput[uvIndex] - 128) + 1.793 * (vInput[uvIndex] - 128)));
            argb.G = FRUC_CLAMP_255((int)(1.164 * (yInput[yIndex] - 16) - 0.213 * (uInput[uvIndex] - 128) - 0.534 * (vInput[uvIndex] - 128)));
            argb.B = FRUC_CLAMP_255((int)(1.164 * (yInput[yIndex] - 16) + 2.115 * (uInput[uvIndex] - 128) + 0.000 * (vInput[uvIndex] - 128)));
            argb.A = 1;
            argbOutput[outputIndex] = argb;
        }
    }
}

void ConvertARGBToABGR(
            char* strInput, 
            char* strOutput, 
            int iWidth, 
            int iHeight, 
            int outRowPitch)
{
    struct ABGRPixel
    {
        unsigned char red;
        unsigned char green;
        unsigned char blue;
        unsigned char alpha;
    };
    struct ARGBPixel
    {
        unsigned char blue;
        unsigned char green;
        unsigned char red;
        unsigned char alpha;
    };

    ABGRPixel* abgrOutput = (ABGRPixel*)(strOutput);
    ARGBPixel* argbInput = (ARGBPixel*)(strInput);
    for (int y = 0; y < iHeight; y++)
    {
        for (int x = 0; x < iWidth; x++)
        {
            int outputIndex = x + y * outRowPitch / 4;

            abgrOutput[outputIndex].red = argbInput[outputIndex].red;
            abgrOutput[outputIndex].green = argbInput[outputIndex].green;
            abgrOutput[outputIndex].blue = argbInput[outputIndex].blue;
            abgrOutput[outputIndex].alpha = 1;
        }
    }
}
/* BT.709 ltd:  RGB to YUV
[[ 0.18295106  0.61429073  0.06186474]
 [-0.10068796 -0.33807773  0.43876569]
 [ 0.43895425 -0.3987922  -0.04016205]]
 */
void ConvertBGRToYUVPlanar(
            char* strInput,
            char* strOutput,
            int iWidth,
            int iHeight,
            int inputRowPitch)
{
    struct ABGR
    {
        unsigned char R;
        unsigned char G;
        unsigned char B;
        unsigned char A;
    };

    byte* yOutput = (byte*)strOutput;
    byte* uOutput = yOutput + iWidth * iHeight;
    byte* vOutput = uOutput + iHeight * iWidth / 4;
    ABGR* argbInput = (ABGR*)(strInput);

    for (int y = 0; y < iHeight; y++)
    {
        for (int x = 0; x < iWidth; x++)
        {
            int inputIndex = x + y * inputRowPitch / sizeof(ABGR);
            ABGR argb;
            argb = argbInput[inputIndex];

            int uvIndex = x / 2 + y / 2 * iWidth / 2;
            int yIndex = x + y * iWidth;
            yOutput[yIndex] = byte(0.18295f * argb.R + 0.61429f * argb.G + 0.06186f * argb.B + 16);
            uOutput[uvIndex] = byte(-0.10069f * argb.R - 0.33807f * argb.G + 0.43876f * argb.B + 128);
            vOutput[uvIndex] = byte(0.43895f * argb.R - 0.39879f * argb.G - 0.04016f * argb.B + 128);
        }
    }

}

void ConvertABGRToARGB(
            char* strInput,
            char* strOutput,
            int iWidth,
            int iHeight,
            int inputRowPitch)
{
    struct ABGRPixel
    {
        unsigned char red;
        unsigned char green;
        unsigned char blue;
        unsigned char alpha;
    };
    struct ARGBPixel
    {
        unsigned char blue;
        unsigned char green;
        unsigned char red;
        unsigned char alpha;
    };

    ABGRPixel* abgrInput = (ABGRPixel*)(strInput);
    ARGBPixel* argbOutput = (ARGBPixel*)(strOutput);
    for (int y = 0; y < iHeight; y++)
    {
        for (int x = 0; x < iWidth; x++)
        {
            int outputIndex = x + y * inputRowPitch / 4;

            argbOutput[outputIndex].red = abgrInput[outputIndex].red;
            argbOutput[outputIndex].green = abgrInput[outputIndex].green;
            argbOutput[outputIndex].blue = abgrInput[outputIndex].blue;
            argbOutput[outputIndex].alpha = 1;
        }
    }
}