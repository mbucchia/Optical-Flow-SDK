
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
#include "IDetector.h"
#include "CTRTManager.h"

#include "opencv2/opencv.hpp"

#include <vector>
#include <string>
#include <math.h>
#include <memory>

/*
Class to manage YOLOv3 detector operations
*/
class CDetector : public IDetector
{
public:
    CDetector() {}
    CDetector(const std::string& engineFile, uint32_t gpuId = 0);
    std::vector<BoundingBox> Run(void* inputFrame, FrameProps props) override;
    static std::string GetObjectClass(uint16_t classId)
    {
        if (classId < NUM_CLASSES)
        {
            return CLASS_NAMES.at(classId);
        }
        return "";
    }
    static uint32_t GetOperatingHeight() { return DETECTOR_HEIGHT; }
    static uint32_t GetOperatingWidth() { return DETECTOR_WIDTH; }

private:
    static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }
    static uint32_t GetIndex(uint32_t a, uint32_t f, uint32_t y, uint32_t x, uint32_t gridWidth, uint32_t gridHeight);
    static std::vector<BoundingBox> GetCandidateBoxes(const std::vector<void*>& outputPointers);
    static std::vector<BoundingBox> DoNMS(std::vector<BoundingBox>& inBoxes, float nmsThreshold);
    std::vector<BoundingBox> PostprocessResult(const std::vector<void*>& outputPointers);

private:
    std::unique_ptr<CTRTManager> m_TRTManager;
    uint32_t m_InputWidth = 0;
    uint32_t m_InputHeight = 0;

    // COCO dataset class names 
    static const std::vector<std::string> CLASS_NAMES;
    // {number of anchor boxes, predictions, gridHeight, gridWidth}
    // At 3 scales
    static const std::vector<std::vector<uint32_t>> OUTPUT_SHAPE;
    // Anchor box dimension at each scale relative to which the box dimensions are predicted
    static const std::vector<std::vector<std::vector<uint32_t>>> ANCHOR_DIMS;
    static const uint32_t DETECTOR_WIDTH;
    static const uint32_t DETECTOR_HEIGHT;
    static const uint32_t NUM_CLASSES;
    static const uint32_t NUM_ANCHORS;
    static const float KEEP_THRESHOLD;
    static const float NMS_THRESHOLD;
};
