/*
* Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/


#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif
#include <regex>
#include "NvOFDataLoader.h"

static std::string generateRegexPattern(const std::string& imageNamePattern)
{
    std::string regex_pat;
    std::string image;
    std::string temp;

    for (auto it = imageNamePattern.cbegin(); it != imageNamePattern.cend(); ++it)
    {
        if (*it == '*')
        {
            image.append(".*");
        }
        else if (*it == '?')
        {
            image.append(".");
        }
        else
        {
            image.append(1, *it);
        }
    }

    size_t pos = image.find_first_of("%");
    if (pos != std::string::npos)
    {
        if (pos > 0)
        {
            regex_pat.append(image.substr(0, pos));
        }
        temp = image.substr(pos + 1);
        pos = temp.find_first_of("d");
        if (pos != std::string::npos)
        {
            if (pos > 0)
            {
                auto nd = atoi(temp.substr(0, pos).c_str());
                std::ostringstream ss;
                ss << "([0-9]){" << nd << ",}";
                regex_pat.append(ss.str());
            }
            else
            {
                regex_pat.append("([0 - 9]){1,}");
            }
            regex_pat.append(temp.substr(pos + 1));
        }
    }
    else
    {
        regex_pat.append(image);
    }
    return regex_pat;
}

static std::vector<std::pair<std::string, std::string>> ReadDirectory(const std::string& path)
{
    std::vector<std::pair<std::string, std::string>> files;
#ifdef _WIN32
    WIN32_FIND_DATAA file;
    std::string filter(path + "//*");
    HANDLE searchHandle = FindFirstFileExA(filter.c_str(),
        FindExInfoStandard, &file, FindExSearchNameMatch, NULL, 0);
    if (searchHandle != INVALID_HANDLE_VALUE)
    {
        do
        {
            const char* name = file.cFileName;
            if ((name[0] == 0) || 
                (name[0] == '.' && name[1] == 0) ||
                (name[0] == '.' && name[1] == '.' && name[2] == 0))
                continue;

            if ((file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
            continue;

            files.push_back(std::make_pair(path + "\\" + std::string(name), std::string(name)));
        } while (FindNextFileA(searchHandle, &file));
        FindClose(searchHandle);
    }
#else
    DIR* d;
    struct dirent* dir;
    d = opendir(path.c_str());
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            const char* name = dir->d_name;
            if ((name[0] == 0) ||
                (name[0] == '.' && name[1] == 0) ||
                (name[0] == '.' && name[1] == '.' && name[2] == 0))
                continue;

            struct stat buf;
            if ((stat(name, &buf) == 0) &&
                S_ISDIR(buf.st_mode))
                continue;

            files.push_back(std::make_pair(path + "/" + std::string(name), std::string(name)));
        }

        closedir(d);
    }
#endif

    return files;
}

static void glob(const std::string& image, std::vector<std::string>& result)
{
    const char dir_separators[] = "/\\";
    std::string wildchart;
    std::string path;
    size_t pos = image.find_last_of(dir_separators);
    if (pos == std::string::npos)
    {
        wildchart = image;
        path = ".";
    }
    else
    {
        path = image.substr(0, pos);
        wildchart = image.substr(pos + 1);
    }
    std::string regex_str = generateRegexPattern(wildchart);
    std::regex regex_pat{ regex_str };
#ifndef NDEBUG
    std::cout << "Input file directory path : " << path << std::endl;
    std::cout << "Input file pattern : " << wildchart << std::endl;
#endif
    std::vector<std::pair<std::string, std::string>> fileNames = ReadDirectory(path);
    for (const auto & p : fileNames)
    {
        if (!p.first.empty() && !p.second.empty())
        {
            auto fileName = p.second;
            if (!wildchart.empty())
            {
                if (regex_match(fileName, regex_pat))
                {
                    result.push_back(p.first);
                }
            }
        }
    }

    if (!result.empty())
    {
        std::sort(result.begin(), result.end());
    }
}

static void GetFrameSizeFromFileName(const char* szFileName, uint32_t& width, uint32_t& height)
{
    const std::string fileName(szFileName);
    std::string reg = "_([0-9]){1,}x([0-9]){1,}";
    std::regex pat{ reg };
    std::smatch sm;
    std::regex_search(fileName, sm, pat);
    int nWidth = 0;
    int nHeight = 0;
#ifdef _WIN32
    sscanf_s(sm.str().c_str(), "_%dx%d_", &width, &height);
#else
    sscanf(sm.str().c_str(), "_%dx%d_", &width, &height);
#endif
    std::cout << "Width  : " << width << "  Height : " << height << std::endl;
}

NvOFDataLoaderPNG::NvOFDataLoaderPNG(const char* szFileName)
{
    glob(szFileName, m_fileNames);

    if (m_fileNames.empty())
    {
        std::ostringstream err;
        err << "Invalid input File name/pattern: " << szFileName << std::endl;
        throw std::runtime_error(err.str());
    }

    InitState(m_fileNames[m_idx]);
    m_pFrameData.reset(new uint8_t[m_pitch * m_height]);
    ReadPNG(m_fileNames[m_idx]);
}

void NvOFDataLoaderPNG::Next()
{
    m_idx++;
    if (m_idx < m_fileNames.size())
    {
        ReadPNG(m_fileNames[m_idx]);
    }
    else
    {
        m_bStatus = false;
    }
}

void NvOFDataLoaderPNG::InitState(const std::string& fileName)
{
    FREE_IMAGE_FORMAT imageFormat = FreeImage_GetFileType(fileName.c_str());
    if (imageFormat != FIF_PNG)
    {
        std::ostringstream err;
        err << "Invalid input File format: " << fileName << std::endl;
        throw std::invalid_argument(err.str());
    }
    FIBITMAP* pBitmap = FreeImage_Load(imageFormat, fileName.c_str(), FIF_LOAD_NOPIXELS);
    if (pBitmap)
    {
        m_width = FreeImage_GetWidth(pBitmap);
        m_height = FreeImage_GetHeight(pBitmap);
        m_bpp = FreeImage_GetBPP(pBitmap);
        FreeImage_Unload(pBitmap);
        if ((m_bpp != 32) && (m_bpp != 24) && (m_bpp != 8))
        {
            std::ostringstream err;
            err << "Unsupported bpp file: " << fileName << std::endl;
            throw std::invalid_argument(err.str());
        }
        if ((m_bpp == 32) || (m_bpp == 24))
            m_pitch = m_width * 4;
        else
            m_pitch = m_width;
    }
}

void NvOFDataLoaderPNG::ReadPNG(const std::string& fileName)
{
    FREE_IMAGE_FORMAT imageFormat = FreeImage_GetFileType(fileName.c_str());
    if (imageFormat != FIF_PNG)
    {
        return;
    }
    FIBITMAP* pBitmap = FreeImage_Load(imageFormat, fileName.c_str());
    if (pBitmap)
    {
        try
        {
            if (m_width != FreeImage_GetWidth(pBitmap))
            {
                std::ostringstream err;
                err << fileName << " width (" << FreeImage_GetWidth(pBitmap) << ") is not same as last file's width (" << m_width << ")" << std::endl;
                throw std::invalid_argument(err.str());
            }
            if (m_height != FreeImage_GetHeight(pBitmap))
            {
                std::ostringstream err;
                err << fileName << " height (" << FreeImage_GetHeight(pBitmap) << ") is not same as last file's height (" << m_height << ")" << std::endl;
                throw std::invalid_argument(err.str());
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
                std::ostringstream err;
                err << "Unsupported bpp file: " << fileName << std::endl;
                throw std::invalid_argument(err.str());
            }
        }
        catch (...)
        {
            FreeImage_Unload(pBitmap);
            throw;
        }
        void* pSysMem = FreeImage_GetBits(pBitmap);
        auto srcPitch = FreeImage_GetPitch(pBitmap);
        auto destPitch = m_pitch;
        for (uint32_t i = 0; i < m_height; ++i)
        {
            uint8_t* pSrcLine = FreeImage_GetScanLine(pBitmap, m_height - i - 1);
            memcpy((uint8_t*)m_pFrameData.get() + (i * destPitch), pSrcLine, m_pitch);
        }
        FreeImage_Unload(pBitmap);
    }
}


NvOFDataLoaderYUV420::NvOFDataLoaderYUV420(const char* szFileName) :
    m_width(0), m_height(0), m_idx(0)
{
    glob(szFileName, m_fileNames);
    if (m_fileNames.empty())
    {
        std::ostringstream err;
        err << "Invalid input File format: " << szFileName << std::endl;
        throw std::runtime_error(err.str());
    }

    GetFrameSizeFromFileName(m_fileNames[0].c_str(), m_fileWidth, m_fileHeight);
    m_fpInput.open(m_fileNames[0], std::ios::in | std::ios::binary);
    m_fpInput.seekg(0, m_fpInput.end);
    auto fileSize = m_fpInput.tellg();
    m_fpInput.seekg(0, m_fpInput.beg);

    auto chromaWidth = (m_fileWidth + 1) / 2;
    auto chromaHeight = (m_fileHeight + 1) / 2;
    auto frameSize = (m_fileWidth * m_fileHeight) + 2 * (chromaWidth * chromaHeight);

    if ((fileSize % frameSize) != 0)
    {
        throw std::runtime_error("Invalid yuv file format");
    }
    m_numFrames = (uint32_t)(fileSize / frameSize);

    // make sure luma plane dimensions are even
    // some yuv files can have odd dimension for luma plane
    m_width = chromaWidth * 2;
    m_height = chromaHeight * 2;

    uint32_t size = m_width * (m_height + m_height / 2);
    m_pFrameData.reset(new uint8_t[size]);
    memset(m_pFrameData.get(), 0, size);
    m_pNv12Data.reset(new uint8_t[m_width * (m_height + m_height / 2)]);
    ReadYUV();
}

NvOFDataLoaderYUV420::~NvOFDataLoaderYUV420()
{
    m_fpInput.close();
}

void NvOFDataLoaderYUV420::Next()
{
    m_idx++;
    if ((m_fileNames.size() > 1) && (m_idx < m_fileNames.size()))
    {
        uint32_t width = 0;
        uint32_t height = 0;
        GetFrameSizeFromFileName(m_fileNames[m_idx].c_str(), width, height);
        if ((m_fileHeight != height) || (m_fileWidth != width))
        {
            std::ostringstream err;
            err << "New frame dimension (" << width << "x" << height << ") is not same as last frame dimention (" << m_fileWidth << "x" << m_fileHeight << ")" << std::endl;
            throw std::invalid_argument(err.str());
        }
        m_fpInput.close();
        m_fpInput.open(m_fileNames[m_idx], std::ios::in | std::ios::binary);
        ReadYUV();
    }
    else
    {
        if (m_idx < m_numFrames)
        {
            ReadYUV();
        }
    }
}

bool NvOFDataLoaderYUV420::IsDone()
{
    if (m_fileNames.size() > 1)
        return (m_idx == m_fileNames.size());
    else
        return (m_idx == m_numFrames);
}


void NvOFDataLoaderYUV420::convertToNV12(const void* yuv420, void* nv12Output)
{
    uint32_t lumaSize = m_width * m_height;
    uint32_t chromaSize = m_width * m_height / 2;
    memcpy(nv12Output, yuv420, lumaSize);
    uint8_t* cbPlane = (uint8_t*)yuv420 + (m_width * m_height);
    uint8_t* crPlane = (uint8_t*)cbPlane + ((m_width / 2) * (m_height / 2));
    uint8_t* nv12Chroma = (uint8_t*)nv12Output + (m_width * m_height);
    for (uint32_t y = 0; y < m_height / 2; y++)
    {
        for (uint32_t x = 0; x < m_width; x = x + 2)
        {
            const uint32_t srcUVWidth = m_width / 2;
            nv12Chroma[(y * m_width) + x] = cbPlane[(y * srcUVWidth) + (x >> 1)];
            nv12Chroma[(y * m_width) + (x + 1)] = crPlane[(y * srcUVWidth) + (x >> 1)];
        }
    }
}

void NvOFDataLoaderYUV420::ReadYUV()
{
    uint32_t lumaSize = m_width * m_height;
    uint32_t chromaSize = m_width * m_height / 2;
    if ((m_fileHeight == m_height) && (m_fileWidth == m_width))
    {
        std::streamsize nRead = m_fpInput.read(reinterpret_cast<char*>(m_pFrameData.get()), lumaSize + chromaSize).gcount();
    }
    else
    {
        // handle odd dimension yuv.
        auto lumaPlaneSize = m_width * m_height;
        auto chromaPlaneSize = (m_width / 2) * (m_height / 2);

        for (uint32_t y = 0; y < m_fileHeight; ++y)
        {
            std::streamsize nRead = m_fpInput.read(reinterpret_cast<char*>(m_pFrameData.get()) + (y * m_width), m_fileWidth).gcount();
        }
        uint32_t planeSize[] = { lumaPlaneSize , lumaPlaneSize + chromaPlaneSize };
        auto fileChromaHeight = (m_fileHeight + 1) / 2;
        auto fileChromaWidth = (m_fileWidth + 1) / 2;
        for (uint32_t i = 0; i < 2; ++i)
        {
            for (uint32_t y = 0; y < fileChromaHeight; ++y)
            {
                std::streamsize nRead = m_fpInput.read(reinterpret_cast<char*>(m_pFrameData.get()) + planeSize[i] + (y * m_width / 2),
                    fileChromaWidth).gcount();
            }
        }
    }
    convertToNV12(m_pFrameData.get(), m_pNv12Data.get());
}

std::unique_ptr<NvOFDataLoader> CreateDataloader(const std::string& dataPath)
{
    std::unique_ptr<NvOFDataLoader> dataLoader;

    auto dotPos = dataPath.find_last_of(".");
    if (dotPos == std::string::npos)
    {
        std::ostringstream err;
        err << "The specified input \"" << dataPath << "\" has no extension" << std::endl;
        throw std::invalid_argument(err.str());
    }

    std::string ext = dataPath.substr(dotPos + 1);
    std::transform(std::begin(ext), std::end(ext), std::begin(ext), ::tolower);
    if (ext == "png")
    {
        dataLoader.reset(new NvOFDataLoaderPNG(dataPath.c_str()));
    }
    else if (ext == "yuv")
    {
        dataLoader.reset(new NvOFDataLoaderYUV420(dataPath.c_str()));
    }
    else
    {
        std::ostringstream err;
        err << "Invalid extension \"" << ext << "\" for input file(s)." << std::endl;
        throw std::invalid_argument(err.str());
    }

    return dataLoader;
}
