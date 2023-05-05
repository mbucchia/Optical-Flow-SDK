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
#if defined _MSC_VER
#include <windows.h>
#elif defined __GNUC__
#include "FreeImage.h"
typedef unsigned short      WORD;
typedef unsigned long ULONG;
#define DWORD ULONG
#define LONG long

/* constants for the biCompression field */
#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L
#define BI_JPEG       4L
#define BI_PNG        5L

typedef struct tagBITMAPCOREHEADER {
        DWORD   bcSize;                 /* used to get to color table */
        WORD    bcWidth;
        WORD    bcHeight;
        WORD    bcPlanes;
        WORD    bcBitCount;
} BITMAPCOREHEADER, *LPBITMAPCOREHEADER, *PBITMAPCOREHEADER;

typedef struct tagBITMAPFILEHEADER {
        WORD    bfType;
        DWORD   bfSize;
        WORD    bfReserved1;
        WORD    bfReserved2;
        DWORD   bfOffBits;
} BITMAPFILEHEADER,  *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;

typedef int                 BOOL;
typedef unsigned char       BYTE;
typedef float               FLOAT;
typedef int                 INT;
typedef unsigned int        UINT;
typedef unsigned int        *PUINT;
#endif

// Saves the RGB buffer as a bitmap
bool SaveRGB(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride);

// Saves the BGR buffer as a bitmap
bool SaveBGR(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride);

// Saves the ARGB buffer as a bitmap
bool SaveARGB(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride);

// Saves the ARGB buffer as a bitmap
bool SaveARGB10(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight,
        int iStride);

// Saves the RGBPlanar buffer as three bitmaps, one bitmap for each channel
bool SaveRGBPlanar(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight);

// Saves the YUV420p buffer as three bitmaps, one bitmap for Y', one for U and one for V
bool SaveYUV(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight);

// Saves the YUV444 buffer as single bitmap after converting to RGB using BT709 equations
bool SaveYUV444(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight);

// Saves the YUV420 buffer as single bitmap after converting to YUV444, then RGB using BT709 equations
bool SaveYUV420(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight);

// Saves the YUV420 buffer as single bitmap after converting to YUV444, then RGB using BT709 equations
bool SaveNV12(
        const char* strfileName,
        BYTE* byData, 
        int iWidth,
        int iHeight, 
        int iStride);

// Saves the provided buffer as a bitmap, this method assumes the Data is formated as a bitmap.
bool SaveBitmap(
        const char* strfileName,
        BYTE* byData,
        int iWidth,
        int iHeight);

bool SaveY(
        const char* strfileName, 
        BYTE* byData, 
        int iWidth,
        int iHeight);

// Saves the ABGR buffer as bitmap
bool SaveABGR(
        const char* strfileName,
        BYTE* byData,     
        int iWidth,
        int iHeight,
        int iStride);
