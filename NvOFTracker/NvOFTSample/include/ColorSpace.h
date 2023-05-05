/*
* Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once
#include <stdint.h>

union RGB_F32 {
    float3 v;
    struct {
        float r, g, b;
    } c;
};

struct RGB_F32_2
{
    RGB_F32 x;
    RGB_F32 y;
};

union R8G8B8
{
    uchar3 v;
    struct {
        unsigned char r, g, b;
    } c;
};

typedef enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_Reserved = 3,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470 = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_YCgCo = 8,
    ColorSpaceStandard_BT2020 = 9,
    ColorSpaceStandard_BT2020C = 10
} ColorSpaceStandard;

template<class RGB>
void Nv12ToRGBPlanar(uint8_t *dpNv12, uint32_t nNv12Pitch, uint8_t *dpRgb, uint32_t nRgbpPitch, uint32_t nWidth, uint32_t nHeight, float fNormalize = 1.0f, int iMatrix = 0);
