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

#include "ColorSpace.h"

#include <cuda_runtime.h>

__constant__ float matYuv2Rgb[3][3];

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

void inline GetConstants(int iMatrix, float &wr, float &wb, int &black, int &white, int &max) {
    black = 16; white = 235;
    max = 255;

    switch (iMatrix)
    {
    case ColorSpaceStandard_BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470:
    case ColorSpaceStandard_BT601:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

void SetMatYuv2Rgb(int iMatrix) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matYuv2Rgb, mat, sizeof(mat));
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
        float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;

    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    Rgb rgb;
    rgb.c.r = Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf);
    rgb.c.g = Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf);
    rgb.c.b = Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);

    return rgb;
}

template<class NV12Unit2, class RGB, class RGBUnit2>
__global__
void NV12ToRGBPlanarKernel(uint8_t *pNv12, uint32_t nNv12Pitch, uint8_t *pRgb, uint32_t nRgbpPitch, uint32_t nWidth, uint32_t nHeight, float fNormalize)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pNv12 + x * sizeof(NV12Unit2) / 2 + y * nNv12Pitch;

    NV12Unit2 l0 = *(NV12Unit2 *)pSrc;
    NV12Unit2 l1 = *(NV12Unit2 *)(pSrc + nNv12Pitch);
    NV12Unit2 ch = *(NV12Unit2 *)(pSrc + (nHeight - y / 2) * nNv12Pitch);

    RGB rgb0 = YuvToRgbForPixel<RGB>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<RGB>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<RGB>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<RGB>(l1.y, ch.x, ch.y);

    uint8_t *pDst = pRgb + x * sizeof(RGBUnit2) / 2 + y * nRgbpPitch;
    *(RGBUnit2 *)pDst = RGBUnit2{ __fdividef(rgb0.v.x, fNormalize), __fdividef(rgb1.v.x, fNormalize) };
    *(RGBUnit2 *)(pDst + nRgbpPitch) = RGBUnit2{ __fdividef(rgb2.v.x, fNormalize), __fdividef(rgb3.v.x, fNormalize) };
    pDst += nRgbpPitch * nHeight;
    *(RGBUnit2 *)pDst = RGBUnit2{ __fdividef(rgb0.v.y, fNormalize), __fdividef(rgb1.v.y, fNormalize) };
    *(RGBUnit2 *)(pDst + nRgbpPitch) = RGBUnit2{ __fdividef(rgb2.v.y, fNormalize), __fdividef(rgb3.v.y, fNormalize) };
    pDst += nRgbpPitch * nHeight;
    *(RGBUnit2 *)pDst = RGBUnit2{ __fdividef(rgb0.v.z, fNormalize), __fdividef(rgb1.v.z, fNormalize) };
    *(RGBUnit2 *)(pDst + nRgbpPitch) = RGBUnit2{ __fdividef(rgb2.v.z, fNormalize), __fdividef(rgb3.v.z, fNormalize) };
}

template<>
void Nv12ToRGBPlanar<RGB_F32>(uint8_t *dpNv12, uint32_t nNv12Pitch, uint8_t *dpRgb, uint32_t nRgbpPitch, uint32_t nWidth, uint32_t nHeight, float fNormalize, int iMatrix)
{
    SetMatYuv2Rgb(iMatrix);
    NV12ToRGBPlanarKernel<uchar2, RGB_F32, float2>
        << <dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2) >> >
        (dpNv12, nNv12Pitch, dpRgb, nRgbpPitch, nWidth, nHeight, fNormalize);
}
