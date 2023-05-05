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

#include "NvOFTSampleUtils.h"
#include "CNvOFTSampleException.h"
#include "CDetector.h"
#include "COFTracker.h"
#include "CFramesProducer.h"

#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

const uint32_t FRAME_RATE = 24;

void DrawTrackedObjects(const std::vector<NvOFT_TRACKED_OBJ>& objects, uint32_t filledSize, cv::Mat& inputMat)
{
    for (uint32_t i = 0; i < filledSize && i < objects.size(); ++i)
    {
        const auto& v = objects[i];
        cv::Rect rect;
        rect.x = v.bbox.x;
        rect.y = v.bbox.y;
        rect.width = v.bbox.width;
        rect.height = v.bbox.height;

        cv::rectangle(inputMat, rect, cv::Scalar(255, 0, 0));
        std::stringstream sstream;
        sstream << CDetector::GetObjectClass(v.classId) << ":" << v.trackingId;
        cv::putText(inputMat, sstream.str(), cv::Point(v.bbox.x, v.bbox.y-1), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);
    }
}

void DumpTrackedObjects(const std::vector<NvOFT_TRACKED_OBJ>& objects, uint32_t frameNum, uint32_t filledSize, std::ostream& streamHandle)
{
    for (uint32_t i = 0; i < filledSize && i < objects.size(); ++i)
    {
        const auto& v = objects[i];
        streamHandle << frameNum << ", "
                     << v.trackingId << ", "
                     << v.classId << ", "
                     << v.bbox.x << ", "
                     << v.bbox.y << ", "
                     << v.bbox.width << ", "
                     << v.bbox.height << ", "
                     << v.confidence << ", "
                     << v.age << ", "
                     << std::endl
                     ;
    }
}

int main(int argc, char** argv)
{
    CommandLineFields clFields;
    try
    {
        ParseCommandLine(argc, argv, clFields);

        std::unique_ptr<IDetector> objectDetector(new CDetector(clFields.detectorEngineFile, clFields.gpuId));
        auto detectorWidth = CDetector::GetOperatingWidth();
        auto detectorHeight = CDetector::GetOperatingHeight();

        CFramesProducer framesProducer(clFields.inputFile, detectorWidth, detectorHeight, clFields.gpuId);
        auto decodeWidth = framesProducer.GetDecodeWidth();
        auto decodeHeight = framesProducer.GetDecodeHeight();
        cv::VideoWriter videoOut;
        if (!clFields.outputFile.empty())
        {
            videoOut.open(clFields.outputFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'j', 'p', 'g'), FRAME_RATE, cv::Size((int)decodeWidth, (int)decodeHeight));
        }
        COFTracker objectTracker(detectorWidth, detectorHeight, NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR, NvOFT_SURFACE_FORMAT_Y, clFields.gpuId);
        objectTracker.InitTracker();
        uint32_t nFrame = 0;
        std::ofstream dumpFile;
        if (!clFields.trackerDumpFile.empty())
        {
            dumpFile.open(clFields.trackerDumpFile, std::ofstream::out);
            if (!dumpFile.is_open())
            {
                std::ostringstream oss;
                oss << "Dumpfile(" << clFields.trackerDumpFile << ") " << "open failed";
                NVOFTSAMPLE_THROW_ERROR(oss.str());
            }
            std::cout << "Dumping Tracked objects to file: " << clFields.trackerDumpFile << std::endl;
        }

        int videoBytes = 0, numFrames = 0;
        do
        {
            numFrames = framesProducer.Decode(videoBytes);
            for (uint32_t i = 0; i < numFrames; ++i)
            {
                std::vector<void*> frames;
                std::cout << "Working on Frame: " << nFrame++ << std::endl;
                framesProducer.GetFrames(frames);
                FrameProps props;
                props.width = detectorWidth;
                props.height = detectorHeight;
                props.channels = 3;
                props.format = FrameFormat::FORMAT_RGB;

                bool doDetection = !((nFrame - 1) % (clFields.skipInterval + 1));
                std::vector<BoundingBox> boxes;
                if (doDetection)
                {
                    boxes = objectDetector->Run(frames[1], props);
                }

                std::vector<NvOFT_OBJ_TO_TRACK> inObjVec;
                for (const auto &v : boxes)
                {
                    NvOFT_OBJ_TO_TRACK obj;
                    obj.classId = (int)v.class_;
                    obj.bbox.x = (int)v.x;
                    obj.bbox.y = (int)v.y;
                    obj.bbox.width = (int)v.w;
                    obj.bbox.height = (int)v.h;
                    obj.confidence = v.prob;
                    obj.pCustomData = nullptr;

                    inObjVec.push_back(obj);
                }
            
                auto trackedObjects = objectTracker.TrackObjects(frames[2], detectorWidth * detectorHeight, detectorWidth, inObjVec, doDetection);
                // Bounding boxes are relative to detector dimensions. Scale to original dimension
                for (uint32_t i = 0; i < trackedObjects.filledSize && i < trackedObjects.objects.size(); ++i)
                {
                    auto& v = trackedObjects.objects[i];
                    v.bbox.x = (int)((v.bbox.x * decodeWidth) / detectorWidth);
                    v.bbox.y = (int)((v.bbox.y * decodeHeight) / detectorHeight);
                    v.bbox.width = (int)((v.bbox.width * decodeWidth) / detectorWidth);
                    v.bbox.height = (int)((v.bbox.height * decodeHeight) / detectorHeight);
                }

                if (videoOut.isOpened())
                {
                    // Copy the original device frame to host
                    auto height = decodeHeight;
                    auto width = decodeWidth;
                    cv::Mat hostFrame(height + height / 2, width, CV_8UC1);
                    cv::Mat hostFrameBGR;
                    auto size = width * height + width * (height / 2);
                    cudaMemcpy(hostFrame.data, frames[0], size, cudaMemcpyDeviceToHost);
                    cv::cvtColor(hostFrame, hostFrameBGR, cv::COLOR_YUV2BGR_NV12);

                    DrawTrackedObjects(trackedObjects.objects, trackedObjects.filledSize, hostFrameBGR);
                    videoOut.write(hostFrameBGR);
                }

                if (dumpFile.is_open())
                {
                    DumpTrackedObjects(trackedObjects.objects, trackedObjects.frameNum, trackedObjects.filledSize, dumpFile);
                }
                if(clFields.dumpToConsole)
                {
                    DumpTrackedObjects(trackedObjects.objects, trackedObjects.frameNum, trackedObjects.filledSize, std::cout);
                }
            }
        } while (videoBytes);
    }
    catch(const std::exception& e)
    {
        std::cout << e.what();
        exit(1);
    }

    return 0;
}
