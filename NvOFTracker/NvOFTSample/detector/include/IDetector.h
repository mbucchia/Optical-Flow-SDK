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
#include <vector>

enum class FrameFormat
{
    FORMAT_RGB,  // Channels first RGB
};

struct FrameProps
{
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    FrameFormat format;
};

struct BoundingBox
{
    float x;      // top left x coordinate
    float y;      // top left y coordinate
    float w;      // box width
    float h;      // box height
    uint32_t class_; // class name
    float prob;   // class probability
};

class IDetector
{
public:
    virtual ~IDetector() {}
public:
    virtual std::vector<BoundingBox> Run(void* inputFrame, FrameProps props) = 0;
};
