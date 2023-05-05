/*
* Copyright 2019-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <stdint.h>

/*This interface is intended for implementations based on 
1) Using Nvidia Optical Flow through OpenCV
2) Direct use of Nvidia OpticalFlow SDK.
We are "likely" to keep using Opencv Mats irrespective of using either of the above approach.
We could have used a more conservative approach of raw pointers and could have 
done a manual copy (based on width, height, stride information). But that is just more code and possibility of bugs.
*/
class IOpticalFlow
{
public:
    virtual ~IOpticalFlow() {}
public:
    virtual void GetFlow(const void* inputFrame, const size_t inputFrameSize, const size_t inputFramePitch, cv::InputOutputArray flowVectors) = 0;
    virtual void GetFlowCost(cv::OutputArray flowCost) = 0;
};