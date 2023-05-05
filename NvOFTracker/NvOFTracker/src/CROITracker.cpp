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

#include "CROITracker.h"
#include "CConnectedRegionGenerator.h"
#include "CNvOFTrackerException.h"

#include <algorithm>

const int32_t CROITracker::LIFE_CONST = 3;
const int32_t CROITracker::INITIAL_DELAY = 3;
const int32_t CROITracker::LIFE_CONST_PERIODIC_DETECTION = 4;
const int32_t CROITracker::INITIAL_DELAY_PERIODIC_DETECTION = 0;

const int32_t CROITracker::ROI_MIN_WIDTH = 5;
const int32_t CROITracker::ROI_MIN_HEIGHT = 5;

const int32_t CROITracker::MAX_COST = INT_MAX;
const double CROITracker::IOU_WEIGHT = 150.0;

void CROITracker::GenerateIDforNewObjects(std::vector<LabeledInObject>& inObjectList, std::vector<LabeledOutObject>& outObjects)
{
    cv::Rect inRect;
    uint index = 0;
    bool makePersistent = false;

    // ROI list being empty indicates we have no tracked objects and new set of objects are incoming. Make
    // them immediately persistent.
    // Also make the new objects immediately persistent if initial delay is zero
    if (m_ROI.empty() || (GetInitialDelay() == 0))
    {
        makePersistent = true;
    }

    for (auto itr = inObjectList.begin(); itr != inObjectList.end(); itr++)
    {
        if (!itr->m_bMatchFound)
        {
            inRect = itr->m_rectangle;
            LabeledTrackedObject newTrackedObj(inRect, ++m_TrackingId, itr->m_classId, itr->m_confidence);
            newTrackedObj.SetMatchedRect(inRect);
            newTrackedObj.SetInputIndex(index);

            itr->m_id = newTrackedObj.GetUID();
            itr->m_age = newTrackedObj.GetAge();
            if (makePersistent)
            {
                newTrackedObj.SetMatchFound(true);
                newTrackedObj.SetPersistingROI(true);
                newTrackedObj.SetLife(GetLifeConstant());
                UpdateOutObjectList(*itr, outObjects);
            }
            m_ROI.push_back(newTrackedObj);
        }
        index++;
    }
}

void CROITracker::Track(const cv::Mat& flowVectors, std::vector<LabeledInObject>& inObjectList, std::vector<LabeledOutObject>& outObjects)
{
    if (flowVectors.empty())
    {
        GenerateIDforNewObjects(inObjectList, outObjects);
        return;
    }

    m_flowWidth = flowVectors.cols;
    m_flowHeight = flowVectors.rows;

    // Get Flowx and FlowY
    std::vector<cv::Mat> flowPerChannel(2);
    cv::split(flowVectors, flowPerChannel);

    std::unique_ptr<CConnectedRegionGenerator> regionGenerator(new CConnectedRegionGenerator(flowPerChannel[0], flowPerChannel[1], flowPerChannel[0].cols, flowPerChannel[0].rows));
    if (regionGenerator == nullptr)
    {
        NVOFTRACKER_THROW_ERROR("Error allocating memory for region generator", NvOFT_ERR_OUT_OF_SYSTEM_MEMORY);
    }
    for (auto& v : m_ROI)
    {
        // Pass the grid size and scale according to the gridsize
        cv::Rect scaledRect;
        FlowData flow;
        cv::Rect OFObjRect;
        scaledRect.x = v.GetCurrRect().x / 4;
        scaledRect.y = v.GetCurrRect().y / 4;
        scaledRect.width = (v.GetCurrRect().width + 3) / 4;
        scaledRect.height = (v.GetCurrRect().height + 3) / 4;
        regionGenerator->Process(scaledRect);
        regionGenerator->GenerateRepresentativeFlow(flow);

        v.UpdateCentroid(v.GetCurrRect());
        v.UpdateFlow(flow);
    }
    MatchRectAndGenerateID(inObjectList, outObjects);
    // Cleanup the tracked object which didn't find a match
    UpdateTrackedObjList();
    GenerateIDforNewObjects(inObjectList, outObjects);

    m_previousDetectionDone = m_currentDetectionDone;
}

void CROITracker::Reset()
{
    m_ROI.clear();
}

void CROITracker::MatchRectAndGenerateID(std::vector<LabeledInObject> &inObjects, std::vector<LabeledOutObject> &outObjects)
{
    const size_t inListSize = inObjects.size();
    const size_t prevListSize = m_ROI.size();

    // No proessing is required if there is no previous ROIs
    if (prevListSize == 0)
    {
        return;
    }

    auto costMatrix = std::vector<std::vector<double>>(prevListSize, std::vector<double>(inListSize, MAX_COST));
    std::vector<int> matches(prevListSize, -1);

    cv::Point ptTracked;
    if (inListSize > 0)
    {
        m_currentDetectionDone = true;
        for (uint32_t prevIndex = 0; prevIndex < prevListSize; ++prevIndex)
        {
            auto& v = m_ROI[prevIndex];
            ptTracked.x = v.GetCentroid().x + v.GetRepresentativeFlowX();
            ptTracked.y = v.GetCentroid().y + v.GetRepresentativeFlowY();

            // Find the list of input rects containing the point ptTracked
            for (uint32_t currIndex = 0; currIndex < inListSize; ++currIndex)
            {
                if (inObjects[currIndex].m_classId == v.GetClassId() && inObjects[currIndex].ContainsPt(ptTracked))
                {
                    costMatrix[prevIndex][currIndex] = GenerateCost(v, inObjects[currIndex]);
                }
            }
        }
        double cost = m_HungarianAlgo.Solve(costMatrix, matches);

        if (m_previousDetectionDone)
        {
            m_periodicDetection = false;
        }
    }
    else
    {
        m_currentDetectionDone = false;
        if (!m_periodicDetection)
        {
            for (auto& v : m_ROI)
            {
                v.SetPersistingROI(true);
                auto life = (v.GetLife() == LIFE_CONST) ? LIFE_CONST_PERIODIC_DETECTION : v.GetLife();
                v.SetLife(life);
            }
            m_periodicDetection = true;
        }
    }

    int matchingIdx;
    bool bMatchFound;
    for (size_t i = 0; i < prevListSize; ++i)
    {
        auto &trackedObj = m_ROI[i];
        matchingIdx = matches[i];

        bMatchFound = false;
        if (matchingIdx >= 0 && matchingIdx < inListSize)
        {
            if (costMatrix[i][matchingIdx] != MAX_COST)
            {
                trackedObj.UpdateMatchedRect(inObjects[matchingIdx].m_rectangle);
                trackedObj.SetInputIndex(matchingIdx);

                // Set matchfound in the inputOjbect to prevent matching next time
                inObjects[matchingIdx].m_bMatchFound = true;
                inObjects[matchingIdx].m_id = trackedObj.GetUID();
                inObjects[matchingIdx].m_age = trackedObj.GetAge();

                bMatchFound = true;
            }
        }
        trackedObj.SetMatchFound(bMatchFound);
        trackedObj.UpdateStates(bMatchFound, GetInitialDelay(), GetLifeConstant());
        // If match not found, treat the current ROI rect as input.
        // Add the flow to the current rect and create output rect
        if (!bMatchFound)
        {
            cv::Rect tempCurrRect = trackedObj.GetCurrRect();
            tempCurrRect.x += trackedObj.GetRepresentativeFlowX();
            tempCurrRect.y += trackedObj.GetRepresentativeFlowY();

            // Validate that tempCurrRect is not going out of the frame
            if (tempCurrRect.x < 1)
            {
                tempCurrRect.width = tempCurrRect.x + tempCurrRect.width;
                tempCurrRect.x = 1;
            }
            if (tempCurrRect.y < 1)
            {
                tempCurrRect.height = tempCurrRect.y + tempCurrRect.height;
                tempCurrRect.y = 1;
            }
            if (tempCurrRect.x + (uint32_t)tempCurrRect.width >= (m_frameWidth - 1))
            {
                tempCurrRect.width = m_frameWidth - tempCurrRect.x - 1;
            }
            if (tempCurrRect.y + (uint32_t)tempCurrRect.height >= (m_frameHeight - 1))
            {
                tempCurrRect.height = m_frameHeight - tempCurrRect.y - 1;
            }
            if (tempCurrRect.width < ROI_MIN_WIDTH || tempCurrRect.height < ROI_MIN_HEIGHT)
            {
                trackedObj.SetLife(0);
                trackedObj.SetPersistingROI(false);
            }
            trackedObj.UpdateCurrRect(tempCurrRect);
            trackedObj.SetMatchedRect(tempCurrRect);
        }
    }
    UpdateOutObjectList(inObjects, outObjects);
}

double CROITracker::GenerateCost(LabeledTrackedObject& prevObj, LabeledInObject& currInObj)
{
    cv::Point ptTracked;
    double cost;
    ptTracked.x = prevObj.GetCentroid().x + prevObj.GetRepresentativeFlowX();
    ptTracked.y = prevObj.GetCentroid().y + prevObj.GetRepresentativeFlowY();

    currInObj.CalcDistanceFromCentroid(ptTracked);

    cv::Rect tempRect = prevObj.GetCurrRect();
    tempRect.x = tempRect.x + prevObj.GetRepresentativeFlowX();
    tempRect.y = tempRect.y + prevObj.GetRepresentativeFlowY();
    currInObj.CalcIntersectionRatio(tempRect);

    cost = currInObj.m_distance + IOU_WEIGHT * (1 - currInObj.m_ratio);
    return cost;
}

void CROITracker::UpdateTrackedObjList()
{
    for (int i = 0; i < m_ROI.size();)
    {
        if (m_ROI[i].IsMatchFound())
        {
            m_ROI[i].SetMatchFound(false);
            m_ROI[i].UpdateCurrRect(m_ROI[i].GetMatchedRect());
            i++;
        }
        else
        {
            if (m_ROI[i].GetLife() == 0)
            {
                m_ROI.erase(m_ROI.begin() + i);
            }
            else
            {
                i++;
            }
        }
    }
}

void CROITracker::UpdateOutObjectList(const LabeledInObject& inObject, std::vector<LabeledOutObject>& outObjects)
{
    LabeledOutObject newOutObject;

    newOutObject.m_rectangle = inObject.m_rectangle;
    newOutObject.m_classId = inObject.m_classId;
    newOutObject.m_trackingId = inObject.m_id;
    newOutObject.m_age = inObject.m_age;
    newOutObject.m_confidence = inObject.m_confidence;
    newOutObject.m_ObjIn = inObject.m_obj_in;

    outObjects.push_back(newOutObject);
}

void CROITracker::UpdateOutObjectList(std::vector<LabeledInObject>& inObjects, std::vector<LabeledOutObject>& outObjects)
{
    for (auto &v : m_ROI)
    {
        if (v.IsPersistingROI())
        {
            LabeledOutObject newOutObject;
            if (v.IsMatchFound())
            {
                auto inObj = inObjects[v.GetInputIndex()];

                newOutObject.m_rectangle = inObj.m_rectangle;
                newOutObject.m_classId = inObj.m_classId;
                newOutObject.m_trackingId = inObj.m_id;
                newOutObject.m_age = inObj.m_age;
                newOutObject.m_confidence = inObj.m_confidence;
                newOutObject.m_ObjIn = inObj.m_obj_in;
            }
            else
            {
                newOutObject.m_rectangle = v.GetMatchedRect();

                newOutObject.m_classId = v.GetClassId();
                newOutObject.m_trackingId = v.GetUID();
                newOutObject.m_age = v.GetAge();
                newOutObject.m_confidence = v.GetConfidence();
                newOutObject.m_ObjIn = NULL;
            }
            outObjects.push_back(newOutObject);
        }
    }
}

void LabeledInObject::UpdateCentroid()
{
    m_centroid.x = m_rectangle.x + m_rectangle.width / 2;
    m_centroid.y = m_rectangle.y + m_rectangle.height / 2;
}

bool LabeledInObject::ContainsPt(cv::Point pt) const
{
    if (((pt.x >= m_rectangle.x) && (pt.x < m_rectangle.x + m_rectangle.width)) && ((pt.y >= m_rectangle.y) && (pt.y < m_rectangle.y + m_rectangle.height)))
    {
        return true;
    }
    return false;
}

void LabeledInObject::CalcDistanceFromCentroid(cv::Point pt)
{
    cv::Point diff = m_centroid - pt;
    m_distance = (float)sqrt(diff.x * diff.x + diff.y * diff.y);
}

void LabeledInObject::CalcIntersectionRatio(cv::Rect rect)
{
    cv::Rect iRect = rect & m_rectangle;
    cv::Rect oRect = rect | m_rectangle;
    m_ratio = (float)((iRect.width * iRect.height * 1.0) / (oRect.width * oRect.height));
}

void LabeledInObject::CalcCombinedCost()
{
    // TODO Currently the number 8 is arbitrary. need to be tweaked
    m_combined = m_distance + (float)8 / m_ratio;
}

void LabeledTrackedObject::UpdateMatchedRect(const cv::Rect& rect)
{
    m_matchedRect = rect;
    m_bMatchFound = true;
}

void LabeledTrackedObject::UpdateCentroid(const cv::Rect& rect)
{
    m_centroid.x = rect.x + rect.width / 2;
    m_centroid.y = rect.y + rect.height / 2;
}

void LabeledTrackedObject::UpdateFlow(const FlowData& flow)
{
    m_flowX = flow.flowX;
    m_flowY = flow.flowY;
}

void LabeledTrackedObject::UpdateStates(bool bTracked, int32_t initialDelay, int32_t lifeConstant)
{
    ++m_age;

    if (!m_bPersistingROI)
    {
        if (bTracked)
        {
            if (m_age > (uint32_t)initialDelay)
            {
                m_bPersistingROI = true;
                m_life = lifeConstant;
            }
        }
        else
        {
            m_life = 0;
        }
    }
    else
    {
        if (bTracked)
        {
            m_life = lifeConstant;
        }
        else
        {
            if (m_life > 0)
            {
                m_life--;
            }
        }
    }
}
