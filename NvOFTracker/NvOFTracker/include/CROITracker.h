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
#include "CHungarianAlgorithm.h"
#include "FlowData.h"

#include "opencv2/core.hpp"

#include <vector>
#include <stdint.h>

struct LabeledInObject
{
    void UpdateCentroid();
    bool ContainsPt(cv::Point pt) const;
    void CalcDistanceFromCentroid(cv::Point pt);
    void CalcIntersectionRatio(cv::Rect rect);
    void CalcCombinedCost();

    float m_confidence;
    int32_t m_classId;
    void* m_obj_in;
    cv::Point m_centroid;
    float m_ratio;
    float m_distance;
    float m_combined;
    cv::Rect m_rectangle;
    bool m_bMatchFound;
    uint32_t m_age;
    uint64_t m_id;
};

struct LabeledOutObject
{
    cv::Rect m_rectangle;
    int32_t m_classId;
    uint64_t m_trackingId;
    uint32_t m_age;
    float m_confidence;
    void* m_ObjIn;
};

class LabeledTrackedObject
{
public:
    LabeledTrackedObject(cv::Rect currentRect, uint64_t uid, int32_t classId, float confidence):
        m_currRect(currentRect),
        m_uid(uid),
        m_classId(classId),
        m_confidence(confidence)
    {}
    uint64_t GetUID() const { return m_uid; }
    uint32_t GetAge() const { return m_age; }
    float GetConfidence() const { return m_confidence; }
    int32_t GetClassId() const { return m_classId; }
    cv::Point GetCentroid() const { return m_centroid; }
    int32_t GetRepresentativeFlowX() const { return m_flowX; }
    int32_t GetRepresentativeFlowY() const { return m_flowY; }
    void UpdateCentroid(const cv::Rect& rect);
    void UpdateFlow(const FlowData& flow);
    void UpdateMatchedRect(const cv::Rect& rect);
    void UpdateStates(bool bTracked, int32_t initialDelay, int32_t lifeConst);

    cv::Rect GetCurrRect() const { return m_currRect; }
    void UpdateCurrRect(const cv::Rect& rect) { m_currRect = rect; }
    cv::Rect GetMatchedRect() const { return m_matchedRect; }
    void SetMatchedRect(const cv::Rect& rect) { m_matchedRect = rect; }
    bool IsMatchFound() const { return m_bMatchFound; }
    void SetMatchFound(bool bMatch) { m_bMatchFound = bMatch; }
    bool IsPersistingROI() const { return m_bPersistingROI; }
    void SetPersistingROI(bool bPersisting) { m_bPersistingROI = bPersisting; }
    uint32_t GetInputIndex() const { return m_inputIndex; }
    void SetInputIndex(uint32_t index) { m_inputIndex = index; }
    uint32_t GetLife() const { return m_life; }
    void SetLife(uint32_t life) { m_life = life; }

private:
    cv::Rect m_currRect;                // inputRect. It gets updated either from the app incoming rect or the m_matched rect
    uint64_t m_uid;
    int32_t m_classId;
    float m_confidence;
    cv::Rect m_matchedRect;             // It contains the matched rect at end of tracking.
    cv::Point m_centroid;               // This represents the cetroid of the OF generated rect.
    bool m_bMatchFound = false;         // Match for the current object is found in the current frame. Need to reset at the end of frame.
    int32_t m_flowX = 0;                // Representative flow. Currently median flow of the OF generated object
    int32_t m_flowY = 0;                // Representative flow. Currently median flow of the OF generated object
    uint32_t m_age = 0;
    bool m_bPersistingROI = 0;
    uint32_t m_inputIndex = 0;
    uint32_t m_life = 0;
};

class CROITracker
{
public:
    CROITracker(uint32_t width, uint32_t height) : m_frameWidth(width), m_frameHeight(height) {}
    void Track(const cv::Mat& flowVectors, std::vector<LabeledInObject>& inObjects, std::vector<LabeledOutObject>& outObjects);
    void Reset();
    int32_t GetLifeConstant()
    {
        if (m_periodicDetection)
        {
            return LIFE_CONST_PERIODIC_DETECTION;
        }
        return LIFE_CONST;
    }
    int32_t GetInitialDelay()
    {
        if (m_periodicDetection)
        {
            return INITIAL_DELAY_PERIODIC_DETECTION;
        }
        return INITIAL_DELAY;
    }

private:
    void MatchRectAndGenerateID(std::vector<LabeledInObject>& inObjects, std::vector<LabeledOutObject>& outObjects);
    double GenerateCost(LabeledTrackedObject& prevObj, LabeledInObject& currInObj);
    void UpdateTrackedObjList();
    void UpdateOutObjectList(const LabeledInObject& inObject, std::vector<LabeledOutObject>& outObjects);
    void UpdateOutObjectList(std::vector<LabeledInObject>& inObjects, std::vector<LabeledOutObject>& outObjects);
    void GenerateIDforNewObjects(std::vector<LabeledInObject>& inObjects, std::vector<LabeledOutObject>& outObjects);

private:
    uint32_t m_flowWidth = 0;
    uint32_t m_flowHeight = 0;
    uint32_t m_frameWidth = 0;
    uint32_t m_frameHeight = 0;
    uint64_t m_TrackingId = 0;
    CHungarianAlgorithm m_HungarianAlgo;
    std::vector<LabeledTrackedObject> m_ROI;
    bool m_periodicDetection = false;
    bool m_currentDetectionDone = true; // Is detection done for this frame.
    bool m_previousDetectionDone = false; // Was detection done for previous frame

    static const int32_t MAX_COST; // Used for cost matrix as default value
    static const double IOU_WEIGHT;
    static const int32_t LIFE_CONST;
    static const int32_t INITIAL_DELAY;
    static const int32_t LIFE_CONST_PERIODIC_DETECTION;
    static const int32_t INITIAL_DELAY_PERIODIC_DETECTION;
    static const int32_t ROI_MIN_WIDTH;
    static const int32_t ROI_MIN_HEIGHT;
};
