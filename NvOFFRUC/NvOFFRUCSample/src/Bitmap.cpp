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


#pragma warning(disable : 4995 4996)

#include "Bitmap.h"
#include <stdio.h>
#include <string>
#include <string.h>

// Macros to help with bitmap padding
#define BITMAP_SIZE(iWidth, iHeight) ((((iWidth) + 3) & ~3) * (iHeight))
#define BITMAP_INDEX(x, y, iWidth) (((y) * (((iWidth) + 3) & ~3)) + (x))

// Describes the structure of a 24-iBPP Bitmap pixel
struct BitmapPixel
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

// Describes the structure of a RGB pixel
struct RGBPixel
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

// Describes the structure of a ARGB pixel
struct ARGBPixel
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    unsigned char alpha;
};

// Describes the structure of a ARGB pixel
struct ABGRPixel
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
};

bool SaveBitmap(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    BITMAPFILEHEADER bmpFileHeader = { 0 };
    BITMAPINFOHEADER bmpInfoHeader = { 0 };
    FILE* outputFile = NULL;
    bool bRet = false;

    if (byData)
    {
        if (outputFile = fopen(strfileName, "wb"))
        {
            iWidth = (iWidth + 3) & (~3);
            int size = iWidth * iHeight * 3; // 24 bits per pixel

            bmpFileHeader.bfType = 0x4D42;
            bmpFileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + size;
            bmpFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

            bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmpInfoHeader.biWidth = iWidth;
            bmpInfoHeader.biHeight = iHeight;
            bmpInfoHeader.biPlanes = 1;
            bmpInfoHeader.biBitCount = 24;
            bmpInfoHeader.biCompression = BI_RGB;
            bmpInfoHeader.biSizeImage = BITMAP_SIZE(iWidth, iHeight);
            bmpInfoHeader.biXPelsPerMeter = 0;
            bmpInfoHeader.biYPelsPerMeter = 0;
            bmpInfoHeader.biClrUsed = 0;
            bmpInfoHeader.biClrImportant = 0;

            fwrite((unsigned char*)&bmpFileHeader, 1, sizeof(BITMAPFILEHEADER), outputFile);
            fwrite((unsigned char*)&bmpInfoHeader, 1, sizeof(BITMAPINFOHEADER), outputFile);
            fwrite(byData, 1, size, outputFile);

            bRet = true;
            fclose(outputFile);
        }
    }

    return bRet;
}

bool SaveRGB(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;

    RGBPixel* pInput = (RGBPixel*)byData;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];

    // Pad bytes need to be set to zero, it's easier to just set the entire chunk of memory
    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            // In a bitmap (0,0) is at the bottom left, in the frame buffer it is the top left.
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iStride) + col;

            pOutput[outputIdx].red = pInput[inputIdx].red;
            pOutput[outputIdx].green = pInput[inputIdx].green;
            pOutput[outputIdx].blue = pInput[inputIdx].blue;
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;

    return bResult;
}

bool SaveBGR(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;

    if (!byData)
        return false;
    RGBPixel* pInput = (RGBPixel*)byData;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];

    // Pad bytes need to be set to zero, it's easier to just set the entire chunk of memory
    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            // In a bitmap (0,0) is at the bottom left, in the frame buffer it is the top left.
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iStride) + col;

            pOutput[outputIdx].red = pInput[inputIdx].blue;
            pOutput[outputIdx].green = pInput[inputIdx].green;
            pOutput[outputIdx].blue = pInput[inputIdx].red;
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;
    return bResult;
}

bool SaveRGBPlanar(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    if (!byData)
        return false;

    const char* nameExt[] = { "red", "green", "blue" };
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if(pOutput == NULL)
        return false;

    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int color = 0; color < 3; ++color)
    {
        for (int row = 0; row < iHeight; ++row)
        {
            for (int col = 0; col < iWidth; ++col)
            {
                int outputIdx = BITMAP_INDEX(col, row, iWidth);
                int inputIdx = ((iHeight - row - 1) * iWidth) + col;

                pOutput[outputIdx].blue = 0;
                pOutput[outputIdx].green = 0;
                pOutput[outputIdx].red = 0;

                switch (color)
                {
                case 0:
                    pOutput[outputIdx].red = byData[inputIdx];
                    break;

                case 1:
                    pOutput[outputIdx].green = byData[inputIdx + (iWidth * iHeight)];
                    break;

                case 2:
                    pOutput[outputIdx].blue = byData[inputIdx + 2 * (iWidth * iHeight)];
                    break;

                default:
                    break;
                }
            }
        }

        std::string outputFile = strfileName;
        size_t find = outputFile.find_last_of(".");

        outputFile.insert(find, "-");
        outputFile.insert(find + 1, nameExt[color]);

        if (!SaveBitmap(outputFile.c_str(), (BYTE*)pOutput, iWidth, iHeight))
        {
            delete[] pOutput;
            return false;
        }
    }

    delete[] pOutput;
    pOutput = NULL;

    return true;
}

bool SaveARGB(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;
    if (!byData)
        return bResult;

    int iPitch = iStride ? iStride : iWidth;

    ARGBPixel* pInput = (ARGBPixel*)byData;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if (pOutput == NULL)
        return false;

    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iPitch) + col;

            pOutput[outputIdx].red = pInput[inputIdx].red;
            pOutput[outputIdx].green = pInput[inputIdx].green;
            pOutput[outputIdx].blue = pInput[inputIdx].blue;
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;

    return bResult;
}

bool SaveABGR(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;
    if (!byData)
        return bResult;

    int iPitch = iStride ? iStride : iWidth;

    ABGRPixel* pInput = (ABGRPixel*)byData;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if (pOutput == NULL)
        return false;

    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iPitch) + col;

            pOutput[outputIdx].red = pInput[inputIdx].red;
            pOutput[outputIdx].green = pInput[inputIdx].green;
            pOutput[outputIdx].blue = pInput[inputIdx].blue;
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;

    return bResult;
}

#define GET_8BIT_R_FROM_ABGR10(bgr10) (unsigned char)((bgr10 & 0x3FF) >> 2)
#define GET_8BIT_G_FROM_ABGR10(bgr10) (unsigned char)((bgr10 & 0xFFC00) >> 12)
#define GET_8BIT_B_FROM_ABGR10(bgr10) (unsigned char)((bgr10 & 0x3FF00000) >> 22)

bool SaveARGB10(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;
    if (!byData)
        return bResult;

    int iPitch = iStride ? iStride : iWidth;


    DWORD* pInput = (DWORD*)byData;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if (pOutput == NULL)
        return false;

    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iPitch) + col;

            pOutput[outputIdx].red = GET_8BIT_R_FROM_ABGR10(pInput[inputIdx]);
            pOutput[outputIdx].green = GET_8BIT_G_FROM_ABGR10(pInput[inputIdx]);
            pOutput[outputIdx].blue = GET_8BIT_B_FROM_ABGR10(pInput[inputIdx]);
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;

    return bResult;
}
bool SaveYUV(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    if (!byData)
        return false;

    int hWidth = iWidth >> 1;
    int hHeight = iHeight >> 1;
    size_t find = -1;
    std::string outputFile("");

    BitmapPixel* luma = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    BitmapPixel* chrom = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];

    if (luma == NULL || chrom == NULL)
        return false;

    memset(luma, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));
    memset(chrom, 0, BITMAP_SIZE(hWidth, hHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iWidth) + col;

            luma[outputIdx].red = byData[inputIdx];
            luma[outputIdx].green = byData[inputIdx];
            luma[outputIdx].blue = byData[inputIdx];
        }
    }

    byData += iWidth * iHeight;

    outputFile = strfileName;
    find = outputFile.find_last_of(".");

    outputFile.insert(find, "-");
    outputFile.insert(find + 1, "y");

    if (!SaveBitmap(outputFile.c_str(), (BYTE*)luma, iWidth, iHeight))
    {
        delete[] luma; 
        luma = NULL;
        delete[] chrom;
        chrom = NULL;
        return false;
    }

    for (int row = 0; row < hHeight; ++row)
    {
        for (int col = 0; col < hWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, hWidth);
            int inputIdx = ((hHeight - row - 1) * hWidth) + col;
            chrom[outputIdx].red = byData[inputIdx];
            chrom[outputIdx].green = 255 - byData[inputIdx];
            chrom[outputIdx].blue = 0;
        }
    }

    byData += hWidth * hHeight;

    outputFile = strfileName;
    find = outputFile.find_last_of(".");

    outputFile.insert(find, "-");
    outputFile.insert(find + 1, "u");

    if (!SaveBitmap(outputFile.c_str(), (BYTE*)chrom, hWidth, hHeight))
    {
        delete[] luma;
        luma = NULL;
        delete[] chrom;
        chrom = NULL;
        return false;
    }

    for (int row = 0; row < hHeight; ++row)
    {
        for (int col = 0; col < hWidth; ++col)
        {
            int outputIdx = BITMAP_INDEX(col, row, hWidth);
            int inputIdx = ((hHeight - row - 1) * hWidth) + col;

            chrom[outputIdx].red = 0;
            chrom[outputIdx].green = 255 - byData[inputIdx];
            chrom[outputIdx].blue = byData[inputIdx];
        }
    }

    byData += hWidth * hHeight;

    outputFile = strfileName;
    find = outputFile.find_last_of(".");

    outputFile.insert(find, "-");
    outputFile.insert(find + 1, "v");

    if (!SaveBitmap(outputFile.c_str(), (BYTE*)chrom, hWidth, hHeight))
    {
        delete[] luma;
        luma = NULL;
        delete[] chrom;
        chrom = NULL;
        return false;
    }

    delete[] luma;
    luma = NULL;
    delete[] chrom;
    chrom = NULL;
    return true;
}

bool SaveY(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    bool bResult = false;
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if (pOutput == NULL)
        return false;

    BYTE* y = byData;
    BYTE* u = y + iWidth * iHeight;
    BYTE* v = u + iWidth * iHeight;

    // Pad bytes need to be set to zero, it's easier to just set the entire chunk of memory
    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            // In a bitmap (0,0) is at the bottom left, in the frame buffer it is the top left.
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iWidth) + col;
            {
                pOutput[outputIdx].red = y[inputIdx];
                pOutput[outputIdx].green = y[inputIdx];// (BYTE)CLAMP_255(1.164*(y[inputIdx] - 16) - 0.213*(u[inputIdx] - 128) - 0.534*(v[inputIdx] - 128));
                pOutput[outputIdx].blue = y[inputIdx];// (BYTE)CLAMP_255(1.164*(y[inputIdx] - 16) + 2.115*(u[inputIdx] - 128) + 0.000*(v[inputIdx] - 128));
            }
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;
    return bResult;
}
#define CLAMP_255(x)  ((x) > 255 ? 255 : ((x) < 0 ? 0 : (x)))
bool SaveYUV444(
        const char* strfileName, 
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    bool bResult = false;
    bool bIsHD = (iWidth * iHeight < 1280 * 720);
    BitmapPixel* pOutput = new BitmapPixel[BITMAP_SIZE(iWidth, iHeight)];
    if (pOutput == NULL)
        return false;

    BYTE* y = byData;
    BYTE* u = y + iWidth * iHeight;
    BYTE* v = u + iWidth * iHeight;

    // Pad bytes need to be set to zero, it's easier to just set the entire chunk of memory
    memset(pOutput, 0, BITMAP_SIZE(iWidth, iHeight) * sizeof(BitmapPixel));

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            // In a bitmap (0,0) is at the bottom left, in the frame buffer it is the top left.
            int outputIdx = BITMAP_INDEX(col, row, iWidth);
            int inputIdx = ((iHeight - row - 1) * iWidth) + col;
            if (!bIsHD)
            {
                pOutput[outputIdx].red = (BYTE)CLAMP_255(1.164 * (y[inputIdx] - 16) + 0.000 * (u[inputIdx] - 128) + 1.793 * (v[inputIdx] - 128));
                pOutput[outputIdx].green = (BYTE)CLAMP_255(1.164 * (y[inputIdx] - 16) - 0.213 * (u[inputIdx] - 128) - 0.534 * (v[inputIdx] - 128));
                pOutput[outputIdx].blue = (BYTE)CLAMP_255(1.164 * (y[inputIdx] - 16) + 2.115 * (u[inputIdx] - 128) + 0.000 * (v[inputIdx] - 128));
            }
            else
            {
                pOutput[outputIdx].red = (BYTE)CLAMP_255(1.0 * (y[inputIdx]) + 0.0000 * (u[inputIdx] - 128) + 1.5400 * (v[inputIdx] - 128));
                pOutput[outputIdx].green = (BYTE)CLAMP_255(1.0 * (y[inputIdx]) - 0.1830 * (u[inputIdx] - 128) - 0.4590 * (v[inputIdx] - 128));
                pOutput[outputIdx].blue = (BYTE)CLAMP_255(1.0 * (y[inputIdx]) + 1.8160 * (u[inputIdx] - 128) + 0.0000 * (v[inputIdx] - 128));
            }
        }
    }

    bResult = SaveBitmap(strfileName, (BYTE*)pOutput, iWidth, iHeight);

    delete[] pOutput;
    pOutput = NULL;
    return bResult;
}

bool YUV420ToYUV444(
        BYTE* in,
        BYTE* out,
        int iWidth,
        int iHeight)
{
    int hWidth = iWidth >> 1;
    int hHeight = iHeight >> 1;
    BYTE* oY = out; BYTE* oU = oY + iWidth * iHeight; BYTE* oV = oU + iWidth * iHeight;
    BYTE* iY = in;  BYTE* iU = iY + iWidth * iHeight; BYTE* iV = iU + hWidth * hHeight;

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int inputIdx = ((iHeight - row - 1) * iWidth) + col;
            int outputIdx = ((iHeight - row - 1) * iWidth) + col;
            int inputChIdx = ((hHeight - row / 2 - 1) * hWidth) + col / 2;

            oY[outputIdx] = iY[inputIdx];
            oU[outputIdx] = iU[inputChIdx];
            oV[outputIdx] = iV[inputChIdx];
        }
    }
    return true;
}

bool NV12ToYUV444(
        BYTE* in,
        BYTE* out,
        int iWidth,
        int iHeight,
        int iPitch)
{
    int hWidth = iWidth >> 1;
    int hHeight = iHeight >> 1;
    int hPitch = iPitch >> 1;
    BYTE* oY = out; BYTE* oU = oY + iWidth * iHeight; BYTE* oV = oU + iWidth * iHeight;
    BYTE* iY = in;  BYTE* iU = iY + iHeight * iPitch;

    for (int row = 0; row < iHeight; ++row)
    {
        for (int col = 0; col < iWidth; ++col)
        {
            int inputIdx = ((iHeight - row - 1) * iPitch) + col;
            int outputIdx = ((iHeight - row - 1) * iWidth) + col;
            int inputChIdx = (((hHeight - row / 2 - 1) * hPitch) + col / 2) * 2;

            oY[outputIdx] = iY[inputIdx];
            oU[outputIdx] = iU[inputChIdx];
            oV[outputIdx] = iU[inputChIdx + 1];
        }

    }
    return true;
}

bool SaveYUV420(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight)
{
    bool bResult = false;

    int hWidth = iWidth >> 1;
    int hHeight = iHeight >> 1;

    BYTE* yuv444 = new BYTE[iWidth * iHeight * 3];
    BYTE* y = yuv444;
    BYTE* u = y + iWidth * iHeight;
    BYTE* v = u + hWidth * hHeight;

    YUV420ToYUV444(byData, yuv444, iWidth, iHeight);
    bResult = SaveYUV444(strfileName, yuv444, iWidth, iHeight);
    delete[] yuv444;
    yuv444 = NULL;
    return bResult;
}

bool SaveNV12(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride)
{
    bool bResult = false;

    int hWidth = iWidth >> 1;
    int hHeight = iHeight >> 1;

    BYTE* yuv444 = new BYTE[iWidth * iHeight * 3];
    BYTE* y = yuv444;
    BYTE* u = y + iWidth * iHeight;
    BYTE* v = u + hWidth * hHeight;

    NV12ToYUV444(byData, yuv444, iWidth, iHeight, iStride);
    bResult = SaveYUV444(strfileName, yuv444, iWidth, iHeight);
    delete[] yuv444;
    yuv444 = NULL;
    return bResult;
}