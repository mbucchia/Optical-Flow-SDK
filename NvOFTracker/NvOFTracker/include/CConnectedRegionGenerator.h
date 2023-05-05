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
#include "FlowData.h"

#include "opencv2/core.hpp"

#include <stdint.h>

const float SIMILARITY_GLOBAL_TH = 1.8f;
const float SIMILARITY_THRESHOLD_X = 1.2f;
const float SIMILARITY_THRESHOLD_Y = 1.2f;

const uint32_t HIST_SIZE = (uint32_t)((SIMILARITY_GLOBAL_TH + 1) * 2) + 2;

enum class FlowRank
{
    PRIMARY,
    SECONDARY,
    TERTIARY,
};

class CConnectedRegionGenerator
{
public:
    CConnectedRegionGenerator(const cv::Mat& flowX, const cv::Mat& flowY, int32_t widthFlow, int32_t heightFlow);
    void Process(const cv::Rect& rectIn);
    void GenerateRepresentativeFlow(FlowData& repFlow);

private:
    void GrowRegion(int32_t x, int32_t y);
    bool IsSameRegion(int32_t curPixelIndex, int32_t nextPixelIndex, float thresholdX, 
        float thresholdY, float startFlowX, float startFlowY);
    bool IsInsideROI(int32_t x, int32_t y);
    bool IsInsideFlowBoundary(int32_t x, int32_t y);
    void GetMedianFlow(FlowRank flowRank, int32_t& flowX, int32_t& flowY);
    /* 
       We have two coordinate systems
       1. Flow dimension -> Reference rectangle in this system is the flow rectangle
       2. ROI dimension -> Reference rectangle in this system is the ROI rectangle
       The following functions Convert a linear pixel index between the two systems
    */
    static int32_t GetFlowLinearIndex(int32_t roiIndex, const cv::Rect& roiRect, int32_t flowWidth) 
    { 
        auto x = roiIndex % roiRect.width;
        auto y = roiIndex / roiRect.width;
        auto val = x + roiRect.x + (y + roiRect.y) * flowWidth;

        return val;
    }
    static int32_t GetROILinearIndex(int32_t linearIndex, const cv::Rect& roiRect, int32_t flowWidth)
    {
        auto x = linearIndex % flowWidth;
        auto y = linearIndex / flowWidth;
        auto val = x - roiRect.x + (y - roiRect.y) * roiRect.width;

        return val;
    }

private:
    cv::Mat m_flowX;
    cv::Mat m_flowY;
    int32_t m_widthFlow;
    int32_t m_heightFlow;
    cv::Rect m_rectIn;
    int32_t m_ROILeft;
    int32_t m_ROITop;
    int32_t m_ROIRight;
    int32_t m_ROIBottom;
    int32_t m_ROISize;
    int32_t m_midX;
    int32_t m_midY;
    int32_t m_histogramOffset;
    int32_t m_midXROI;
    int32_t m_midYROI;

    std::vector<int32_t> m_regions;
    std::vector<std::vector<std::vector<int32_t>>> m_histogram;
    std::vector<int32_t> m_regionStartFlowX;
    std::vector<int32_t> m_regionStartFlowY;

    cv::Mat m_ROIMask;

    uint32_t m_curRegion;
    int32_t m_maxRegionLabel;
    int32_t m_secondMaxRegionLabel;
    int32_t m_thirdMaxRegionLabel;
    int32_t m_maxRegionFlowCount;
    int32_t m_secondMaxRegionFlowCount;
    int32_t m_thirdMaxRegionFlowCount;
};
