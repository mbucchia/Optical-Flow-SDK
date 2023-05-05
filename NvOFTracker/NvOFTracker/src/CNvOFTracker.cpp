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

#include "CNvOFTracker.h"
#include "CNvOFTrackerException.h"

CNvOFTracker::CNvOFTracker(const NvOFT_CREATE_PARAMS* createParams)
{
    m_Width = createParams->width;
    m_Height = createParams->height;
    m_GPUId = createParams->gpuID;
    m_SurfType = createParams->surfType;
    m_SurfFormat = createParams->surfFormat;

    // Initialize ROTTracker
    m_ROITracker.reset(new CROITracker(m_Width, m_Height));

    // Initialize Optical Flow
    NvOpticalFlowParams opticalFlowParams;
    opticalFlowParams.width = m_Width;
    opticalFlowParams.height = m_Height;
    opticalFlowParams.surfType = m_SurfType;
    opticalFlowParams.surfFormat = m_SurfFormat;
    opticalFlowParams.gpuID = m_GPUId;
    m_OpticalFlow.reset(new CNvOpticalFlowOCV(opticalFlowParams));
}

void CNvOFTracker::TrackObjects(const NvOFT_PROCESS_IN_PARAMS* inParams, NvOFT_PROCESS_OUT_PARAMS* outParams)
{
    std::vector<LabeledInObject> inObjectList;
    std::vector<LabeledOutObject> outObjectList;

    if (inParams == NULL)
    {
        NVOFTRACKER_THROW_ERROR("Invalid NvOFT_PROCESS_IN_PARAMS pointer", NvOFT_ERR_INVALID_PTR);
    }
    auto currFrame = inParams->surfParams.frameDataPtr;
    auto currFrameSize = inParams->surfParams.frameDataSize;
    auto currFramePitch = inParams->surfParams.pitch;
    if (currFrameSize < currFramePitch * m_Height)
    {
        NVOFTRACKER_THROW_ERROR("Invalid input frame size", NvOFT_ERR_INVALID_PARAM);
    }
    if (m_Width != inParams->surfParams.width ||
        m_Height != inParams->surfParams.height)
    {
        NVOFTRACKER_THROW_ERROR("Invalid input dimensions", NvOFT_ERR_INVALID_PARAM);
    }
    /*
    *************************************************Truth Table for various bools***************************************************************
    |  reset  |  detectionDone  | inParams->list  |                               Comments                                                      |
    |   0     |        1        |         1       | Normal Tracking                                                                             |
    |   0     |        1        |         0       | No ROI found by Detector. Reset the roitracker                                              |
    |   0     |        0        |         0       | Detection was not done and list is empty. Track the objects based on previous information.  |
                                                  | This is a periodic detection scenario.                                                      |
    |   0     |        0        |         1       | Ignore inParams->list and proceed as above.                                                 |
    |   1     |        1        |         1       | Reset OpticalFlow and ROITracker. Eg. Scene change scenario. We will generate new Ids       |
    |   1     |        1        |         0       | Reset OpticalFlow and ROITracker. No ROI found by detector. Return early.                   |
    |   1     |        0        |         0       | Reset OpticalFlow and ROITracker. Nothing to be done (we dont have previous rects to track) |
    |   1     |        0        |         1       | Reset OpticalFlow and ROITracker. Ignore inParams->list                                     |
    *********************************************************************************************************************************************
    */

    // This indicates reset both OpticalFlow and ROITracker (Eg. Scene change)
    if (inParams->reset)
    {
        m_ROITracker->Reset();

        NvOpticalFlowParams opticalFlowParams;
        opticalFlowParams.width = m_Width;
        opticalFlowParams.height = m_Height;
        opticalFlowParams.surfType = m_SurfType;
        opticalFlowParams.surfFormat = m_SurfFormat;
        opticalFlowParams.gpuID = m_GPUId;
        m_OpticalFlow.reset(new CNvOpticalFlowOCV(opticalFlowParams));
    }

    // This indicates there was no object found by the detector. Reset the ROITracker.
    // Call optical flow so the frame and temporal data is updated
    if(inParams->detectionDone && (inParams->list == NULL))
    {
        m_ROITracker->Reset();

        cv::cuda::GpuMat flowVectorsGPU;
        m_OpticalFlow->GetFlow(currFrame, currFrameSize, currFramePitch, flowVectorsGPU);

        outParams->frameNum = inParams->frameNum;
        outParams->list = NULL;
        outParams->listSizeFilled = 0;

        return;
    }

    auto numRects = inParams->listSize;
    auto inList = inParams->list;
    if (numRects > 0 && inList == NULL)
    {
        std::ostringstream oss;
        oss << "NvOFT_OBJ_TO_TRACK list is NULL but numRects = " << numRects;
        NVOFTRACKER_THROW_ERROR(oss.str(), NvOFT_ERR_INVALID_PARAM);
    }
    // If detection was not done and we receive rects then ignore the rects.
    // Continue as if this is periodic detection scenario
    if (!inParams->detectionDone && (inParams->list != NULL))
    {
        numRects = 0;
        inList = NULL;
    }
    for (unsigned int i = 0; i < numRects; ++i)
    {
        cv::Rect newRect;
        LabeledInObject newInObject;

        newRect.x = inList[i].bbox.x;
        newRect.y = inList[i].bbox.y;
        newRect.width = inList[i].bbox.width;
        newRect.height = inList[i].bbox.height;
        SanitizeRect(newRect);

        newInObject.m_rectangle = newRect;
        newInObject.m_classId = inList[i].classId;
        newInObject.m_obj_in = (void* )&inList[i];

        // Updating some internal states
        newInObject.m_id = 0;
        newInObject.m_confidence = inList[i].confidence;
        newInObject.m_bMatchFound = false;
        newInObject.UpdateCentroid();

        inObjectList.push_back(newInObject);
    }
    cv::Mat flowVectors;
    m_OpticalFlow->GetFlow(currFrame, currFrameSize, currFramePitch, flowVectors);
    
    m_ROITracker->Track(flowVectors, inObjectList, outObjectList);

    auto numRectsAllocated = outParams->listSizeAllocated;
    auto outList = outParams->list;
    outParams->listSizeFilled = 0;
    if (numRectsAllocated <= 0)
    {
        // Not possible to send the output list
        outParams->frameNum = inParams->frameNum;
        outParams->list = NULL;
        outParams->listSizeFilled = 0;
        NVOFTRACKER_THROW_ERROR("Invalid value for the listSizeAllocated.", NvOFT_ERR_INVALID_PARAM);
    }
    if (outList == NULL)
    {
        // Not possible to send the output list
        outParams->frameNum = inParams->frameNum;
        outParams->list = NULL;
        outParams->listSizeFilled = 0;
        NVOFTRACKER_THROW_ERROR("Received NULL list in NvOFT_TRACKED_OBJ", NvOFT_ERR_INVALID_PTR);
    }
    if (numRectsAllocated < outObjectList.size())
    {
        std::ostringstream oss;
        oss << "Number of tracked objects exceeds the size of allocated list."
            << "Need list of size at least: " << outObjectList.size()
            << ". But allocated only: " << numRectsAllocated
            << ". Skipping the rest of the objects" << std::endl;
        NVOFTRACKER_THROW_ERROR(oss.str(), NvOFT_SUCCESS);
    }
    for (unsigned int i = 0; i < outObjectList.size() && i < numRectsAllocated; ++i)
    {
        outParams->frameNum = inParams->frameNum;
        outList[i].classId = outObjectList[i].m_classId;
        outList[i].trackingId = outObjectList[i].m_trackingId;
        outList[i].bbox.x = outObjectList[i].m_rectangle.x;
        outList[i].bbox.y = outObjectList[i].m_rectangle.y;
        outList[i].bbox.width = outObjectList[i].m_rectangle.width;
        outList[i].bbox.height = outObjectList[i].m_rectangle.height;
        outList[i].confidence = outObjectList[i].m_confidence;
        outList[i].age = outObjectList[i].m_age;
        outList[i].associatedObjectIn = (NvOFT_OBJ_TO_TRACK* )outObjectList[i].m_ObjIn;
        ++outParams->listSizeFilled;
    }
}
