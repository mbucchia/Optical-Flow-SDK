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

#include "CConnectedRegionGenerator.h"
#include "CNvOFTrackerException.h"

#include "opencv2/imgproc/imgproc.hpp"

#include <queue>

CConnectedRegionGenerator::CConnectedRegionGenerator(const cv::Mat& flowX, const cv::Mat& flowY, int32_t widthFlow, int32_t heightFlow)
{
    m_flowX = flowX;
    m_flowY = flowY;
    m_widthFlow = widthFlow;
    m_heightFlow = heightFlow;
    m_midX = widthFlow / 2;
    m_midY = heightFlow / 2;
    m_histogramOffset = HIST_SIZE / 2;
}

void CConnectedRegionGenerator::Process(const cv::Rect& rectIn)
{
    m_rectIn = rectIn;
    m_ROILeft = 0;
    m_ROITop = 0;
    m_ROIRight = m_rectIn.width;
    m_ROIBottom = m_rectIn.height;
    m_midXROI = m_rectIn.width / 2;
    m_midYROI = m_rectIn.height / 2;
    m_ROISize = m_rectIn.width * m_rectIn.height;

    // reset states
    m_regions = std::vector<int32_t>(m_rectIn.width * m_rectIn.height, 0);
    m_ROIMask = cv::Mat::zeros(cv::Size(m_rectIn.width, m_rectIn.height), CV_8UC1);
    m_histogram.clear();
    m_regionStartFlowX.clear();
    m_regionStartFlowY.clear();
    m_curRegion = 0;
    m_maxRegionLabel = 0;
    m_secondMaxRegionLabel = 0;
    m_thirdMaxRegionLabel = 0;
    m_maxRegionFlowCount = 0;
    m_secondMaxRegionFlowCount = 0;
    m_thirdMaxRegionFlowCount = 0;

    int32_t maxRadius = (m_rectIn.width > m_rectIn.height) ? m_rectIn.width / 2 : m_rectIn.height / 2;
    int32_t xPos = m_midXROI;
    int32_t yPos = m_midYROI;
    int32_t xFlowPos;
    int32_t yFlowPos;

    for (int32_t radius = 0; radius <= maxRadius; ++radius)
    {
        for (int32_t y = 0; y < radius; ++y)
        {
            xPos = m_midXROI + radius;
            yPos = m_midYROI + y;
            xFlowPos = m_rectIn.x + xPos;
            yFlowPos = m_rectIn.y + yPos;
            if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
            {
                GrowRegion(xPos, yPos);
            }

            xPos = m_midXROI - radius;
            yPos = m_midYROI + y;
            xFlowPos = m_rectIn.x + xPos;
            yFlowPos = m_rectIn.y + yPos;
            if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
            {
                GrowRegion(xPos, yPos);
            }

            if (y)
            {
                xPos = m_midXROI + radius;
                yPos = m_midYROI - y;
                xFlowPos = m_rectIn.x + xPos;
                yFlowPos = m_rectIn.y + yPos;
                if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
                {
                    GrowRegion(xPos, yPos);
                }

                xPos = m_midXROI - radius;
                yPos = m_midYROI - y;
                xFlowPos = m_rectIn.x + xPos;
                yFlowPos = m_rectIn.y + yPos;
                if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
                {
                    GrowRegion(xPos, yPos);
                }
            }
        }

        for (int32_t x = 0; x <= radius; ++x)
        {
            xPos = m_midXROI + x;
            yPos = m_midYROI + radius;
            xFlowPos = m_rectIn.x + xPos;
            yFlowPos = m_rectIn.y + yPos;
            if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
            {
                GrowRegion(xPos, yPos);
            }

            if (radius)
            {
                xPos = m_midXROI + x;
                yPos = m_midYROI - radius;
                xFlowPos = m_rectIn.x + xPos;
                yFlowPos = m_rectIn.y + yPos;
                if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
                {
                    GrowRegion(xPos, yPos);
                }
            }

            if (x)
            {
                xPos = m_midXROI - x;
                yPos = m_midYROI + radius;
                xFlowPos = m_rectIn.x + xPos;
                yFlowPos = m_rectIn.y + yPos;
                if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
                {
                    GrowRegion(xPos, yPos);
                }

                if (radius)
                {
                    xPos = m_midXROI - x;
                    yPos = m_midYROI - radius;
                    xFlowPos = m_rectIn.x + xPos;
                    yFlowPos = m_rectIn.y + yPos;
                    if (IsInsideROI(xPos, yPos) && IsInsideFlowBoundary(xFlowPos, yFlowPos))
                    {
                        GrowRegion(xPos, yPos);
                    }
                }
            }

        }
    }

}

bool CConnectedRegionGenerator::IsSameRegion(int32_t curPixelIndex, int32_t nextPixelIndex, float thresholdX, float thresholdY, float startFlowX, float startFlowY)
{
    int32_t nextX = nextPixelIndex % m_widthFlow;
    int32_t nextY = nextPixelIndex / m_widthFlow;
    if (nextX >= m_widthFlow || nextY >= m_heightFlow)
    {
        return false;
    }

    float curMvX = m_flowX.at<float>(curPixelIndex);
    float curMvY = m_flowY.at<float>(curPixelIndex);
    float nextMvX = m_flowX.at<float>(nextPixelIndex);
    float nextMvY = m_flowY.at<float>(nextPixelIndex);

    float diffMvX = curMvX - nextMvX;
    float diffMvY = curMvY - nextMvY;

    float maxdiffMvX = startFlowX - nextMvX;
    float maxdiffMvY = startFlowY - nextMvY;

    if ((fabs(diffMvX) < thresholdX && fabs(diffMvY) < thresholdY) && 
        (fabs(maxdiffMvX) < SIMILARITY_GLOBAL_TH && fabs(maxdiffMvY) < SIMILARITY_GLOBAL_TH))
    {
        return true;
    }
    return false;
}

bool CConnectedRegionGenerator::IsInsideROI(int32_t x, int32_t y)
{
    if ((x >= 0 && x < m_rectIn.width) && 
        (y >= 0 && y < m_rectIn.height))
    {
        return true;
    }
    return false;
}

bool CConnectedRegionGenerator::IsInsideFlowBoundary(int32_t x, int32_t y)
{
    if ((x >= 0 && x < m_widthFlow) && 
        (y >= 0 && y < m_heightFlow))
    {
        return true;
    }
    return false;
}

void CConnectedRegionGenerator::GrowRegion(int32_t x, int32_t y)
{
    int32_t roiLinearIndex  = x + y * m_rectIn.width;
    int32_t flowLinearIndex = GetFlowLinearIndex(roiLinearIndex, m_rectIn, m_widthFlow);
    float startMvx = m_flowX.at<float>(flowLinearIndex);
    float startMvy = m_flowY.at<float>(flowLinearIndex);

    if (m_regions[roiLinearIndex])
    {
        // Visited
    }
    else
    {
        std::queue<int32_t> Q;
        Q.push(roiLinearIndex);
        m_curRegion++;

        m_regions[roiLinearIndex] = m_curRegion;

        // The flow vectors are in quarter pixel resolution
        // For histogram we discard the fractional part and only use integer part.
        m_regionStartFlowX.push_back((int32_t)round(startMvx));
        m_regionStartFlowY.push_back((int32_t)round(startMvy));

        int32_t regionPixelNum = 0;
        int32_t regionPixelNumInsideROI = 0;

        auto regionHistogram = std::vector<std::vector<int32_t>>(HIST_SIZE, std::vector<int32_t>(HIST_SIZE, 0));

        while (!Q.empty())
        {
            int32_t currentROIPixelIndex = Q.front();
            Q.pop();
            regionPixelNum++;
            // Maintain histogram for each of the region
            auto currPixelLinearIndex = GetFlowLinearIndex(currentROIPixelIndex, m_rectIn, m_widthFlow);
            float curFlowX = m_flowX.at<float>(currPixelLinearIndex);
            float curFlowY = m_flowY.at<float>(currPixelLinearIndex);
            int32_t curROIx = currentROIPixelIndex % m_rectIn.width;
            int32_t curROIy = currentROIPixelIndex / m_rectIn.width;
            m_ROIMask.at<uchar>(cv::Point(curROIx, curROIy)) = m_curRegion;

            int32_t xIndex, yIndex;
            if (IsInsideROI(curROIx, curROIy))
            {
                xIndex = (int32_t)round(curFlowX) - (int32_t)round(startMvx) + m_histogramOffset;
                yIndex = (int32_t)round(curFlowY) - (int32_t)round(startMvy) + m_histogramOffset;
                regionHistogram[xIndex][yIndex]++;
                regionPixelNumInsideROI++;
            }

            // Check the value at the next right postion
            int32_t tempPixelIndex = currentROIPixelIndex + 1;
            if (curROIx < (m_ROIRight - 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex + 1, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next left postion
            tempPixelIndex = currentROIPixelIndex - 1;
            if (curROIx > (m_ROILeft + 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex - 1, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next bottom postion
            tempPixelIndex = currentROIPixelIndex + m_rectIn.width;
            if (curROIy < (m_ROIBottom - 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex + m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next top postion
            tempPixelIndex = currentROIPixelIndex - m_rectIn.width;
            if (curROIy > (m_ROITop + 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex - m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next top right corner
            tempPixelIndex = currentROIPixelIndex + 1 - m_rectIn.width;
            if (curROIx < (m_ROIRight - 1) && curROIy >(m_ROITop + 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex + 1 - m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next bottom right corner
            tempPixelIndex = currentROIPixelIndex + 1 + m_rectIn.width;
            if (curROIx < (m_ROIRight - 1) && curROIy < (m_ROIBottom - 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex + 1 + m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next top left corner
            tempPixelIndex = currentROIPixelIndex - 1 - m_rectIn.width;
            if (curROIx > (m_ROILeft + 1) && curROIy > (m_ROITop + 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex - 1 - m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
            // Check the value at the next bottom left corner
            tempPixelIndex = currentROIPixelIndex - 1 + m_rectIn.width;
            if (curROIx > (m_ROILeft + 1) && curROIy < (m_ROIBottom - 1) && m_regions[tempPixelIndex] == 0)
            {
                if (IsSameRegion(currPixelLinearIndex, currPixelLinearIndex - 1 + m_widthFlow, 
                    SIMILARITY_THRESHOLD_X, SIMILARITY_THRESHOLD_Y, startMvx, startMvy))
                {
                    m_regions[tempPixelIndex] = m_curRegion;
                    Q.push(tempPixelIndex);
                }
            }
        }

        m_histogram.push_back(regionHistogram);
        if (regionPixelNumInsideROI < m_maxRegionFlowCount && 
            regionPixelNumInsideROI > m_secondMaxRegionFlowCount)
        {
            m_secondMaxRegionLabel = m_curRegion;
            m_secondMaxRegionFlowCount = regionPixelNumInsideROI;
        }
        else if (regionPixelNumInsideROI > m_maxRegionFlowCount)
        {
            m_secondMaxRegionLabel = m_maxRegionLabel;
            m_secondMaxRegionFlowCount = m_maxRegionFlowCount;
            m_maxRegionLabel = m_curRegion;
            m_maxRegionFlowCount = regionPixelNumInsideROI;
        }
    }
}

void CConnectedRegionGenerator::GenerateRepresentativeFlow(FlowData& repFlow)
{
    // Get the flow vectors of the dominant region
    // We are using median value of all the values in that region
    int32_t repFlowX, repFlowY;
    GetMedianFlow(FlowRank::PRIMARY, repFlowX, repFlowY);

    // if repFlowX == 0 and repFlowY == 0, we need to check the whole ROI is having the same flow vector.
    // Else pick the next biggest region having some non-zero motion.
    if (repFlowX == 0 && repFlowY == 0 && (m_maxRegionFlowCount * 1.0 / m_ROISize < 0.95))
    {
        GetMedianFlow(FlowRank::SECONDARY, repFlowX, repFlowY);
    }

    // Update the output flow
    repFlow.flowX = repFlowX;
    repFlow.flowY = repFlowY;
}

void CConnectedRegionGenerator::GetMedianFlow(FlowRank flowRank, int32_t& flowX, int32_t& flowY)
{
    int32_t regionLabel, histSize;
    switch (flowRank)
    {
    case FlowRank::PRIMARY:
        regionLabel = m_maxRegionLabel;
        histSize = m_maxRegionFlowCount;
        break;
    case FlowRank::SECONDARY:
        regionLabel = m_secondMaxRegionLabel;
        histSize = m_secondMaxRegionFlowCount;
        break;
    // Currently not implemented
    case FlowRank::TERTIARY:
        regionLabel = m_thirdMaxRegionLabel;
        histSize = m_thirdMaxRegionFlowCount;
        break;
    default:
        assert(0);
    }
    int32_t medianPosX, medianPosY;
    int32_t sum = 0;
    int32_t regionFlowX = m_regionStartFlowX[regionLabel - 1];
    int32_t regionFlowY = m_regionStartFlowY[regionLabel - 1];
    bool medianFlowFound = false;

    for (int32_t i = 0; i < HIST_SIZE; i++)
    {
        int32_t j;
        for (j = 0; j < HIST_SIZE; j++)
        {
            sum += m_histogram[regionLabel - 1][i][j];
            if (sum >= histSize / 2)
            {
                medianPosX = i;
                medianPosY = j;
                medianFlowFound = true;
                break;
            }
        }
        if (j < HIST_SIZE)
        {
            break;
        }
    }

    if (!medianFlowFound)
    {
        NVOFTRACKER_THROW_ERROR("Median Flow not found.", NvOFT_ERR_GENERIC);
    }

    flowX = medianPosX - m_histogramOffset + regionFlowX;
    flowY = medianPosY - m_histogramOffset + regionFlowY;
}
