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

Texture2D<int2> src : register(t0);
RWTexture2D<int2> dst : register(u0);

Texture2D<int> src2 : register(t1);
RWTexture2D<int> dst2 : register(u1);

cbuffer globals : register(b0)
{
    uint src_w;
    uint src_h;
    uint dst_w;
    uint dst_h;
    uint nScaleFactor;
    uint reserved1;
    uint reserved2;
    uint reserved3;
};

// data required to do 2x upsampling (BLOCKDIM_Y/2)*(BLOCKDIM_X/2)
groupshared int2 shared_data[8][16]; // Same can be used for 4x upsampling also

[numthreads(32, 16, 1)]

void CSOpticalFlowMain(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex)
{
    uint x = dispatchThreadId.x;
    uint y = dispatchThreadId.y;

    uint x0 = dispatchThreadId.x/nScaleFactor;
    uint y0 = dispatchThreadId.y/nScaleFactor;

    uint i = groupThreadId.x/nScaleFactor;
    uint j = groupThreadId.y/nScaleFactor;

    if ((x%nScaleFactor == 0) && (y%nScaleFactor == 0))
    {
        shared_data[j][i] = src[int2(x0, y0)];
    }

    GroupMemoryBarrierWithGroupSync();

    if ((x < dst_w) && (y < dst_h) && (x0 < src_w) && (y0 < src_h))
    {
        dst[int2(x,y)] = shared_data[j][i];
    }
}

groupshared int shared_data2[8][16];

[numthreads(32, 16, 1)]

void CSStereoMain(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex)
{
    uint x = dispatchThreadId.x;
    uint y = dispatchThreadId.y;

    uint x0 = dispatchThreadId.x/nScaleFactor;
    uint y0 = dispatchThreadId.y/nScaleFactor;

    uint i = groupThreadId.x/nScaleFactor;
    uint j = groupThreadId.y/nScaleFactor;

    if ((x%nScaleFactor == 0) && (y%nScaleFactor == 0))
    {

        shared_data2[j][i] = src2[int2(x0, y0)];
    }

    GroupMemoryBarrierWithGroupSync();

    if ((x < dst_w) && (y < dst_h) && (x0 < src_w) && (y0 < src_h))
    {
        dst2[int2(x,y)] = shared_data2[j][i];
    }
}