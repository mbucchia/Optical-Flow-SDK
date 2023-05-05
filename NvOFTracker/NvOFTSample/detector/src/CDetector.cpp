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

#include "CDetector.h"

#include <set>
#include <sstream>

const uint32_t CDetector::DETECTOR_WIDTH = 608;
const uint32_t CDetector::DETECTOR_HEIGHT = 608;
const uint32_t CDetector::NUM_CLASSES = 80;
const uint32_t CDetector::NUM_ANCHORS = 3;
const float CDetector::KEEP_THRESHOLD = 0.6f;
const float CDetector::NMS_THRESHOLD = 0.5f;

const std::vector<std::vector<uint32_t>> CDetector::OUTPUT_SHAPE = {
    {CDetector::NUM_ANCHORS, 4 + 1 + CDetector::NUM_CLASSES, 19, 19},
    {CDetector::NUM_ANCHORS, 4 + 1 + CDetector::NUM_CLASSES, 38, 38},
    {CDetector::NUM_ANCHORS, 4 + 1 + CDetector::NUM_CLASSES, 76, 76}
};
const std::vector<std::vector<std::vector<uint32_t>>> CDetector::ANCHOR_DIMS = {
    { {116, 90}, {156, 198}, {373, 326} },
    { {30, 61},  {62, 45},   {59, 119}  },
    { {10, 13},  {16, 30},   {33, 23}   }
};
const std::vector<std::string> CDetector::CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

CDetector::CDetector(const std::string& engineFile, uint32_t gpuId)
{
    m_TRTManager.reset(new CTRTManager(engineFile, gpuId, false, true));
}

/*
Data is arrranged in shape "box x features x height x width" format
Return the correct feature index based on individual shape parameters
Params:
a: anchor number
f: feature number
y: location in grid in y direction
x: location in grid in x direction
gridWidth: Width of the grid
gridHeight: Height of the grid
Returns:
uint32_t: returns a index in the flat array corresponding to the value
*/
uint32_t CDetector::GetIndex(uint32_t a, uint32_t f, uint32_t y, uint32_t x, uint32_t gridWidth, uint32_t gridHeight)
{
    return (a * (4 + 1 + NUM_CLASSES) * gridHeight * gridWidth + // offset for next box
            f * gridHeight * gridWidth + // offset for next feature
            y * gridHeight + // ofset for next row
            x); // offset for next element within row
}

std::vector<BoundingBox> CDetector::GetCandidateBoxes(const std::vector<void*>& outputPointers)
{
    std::vector<BoundingBox> bBoxes;
    // For yolov3 there are three output bindings. The outputs returned (flat array) have data in the following shape and order
    // mentioned in OUTPUT_SHAPE
    for (uint32_t i = 0; i < outputPointers.size(); ++i)
    {
        auto gridWidth = OUTPUT_SHAPE[i][3];
        auto gridHeight = OUTPUT_SHAPE[i][2];
        auto numAnchors = NUM_ANCHORS;
        float* pOutput = (float* )outputPointers[i];

        for (uint32_t y = 0; y < gridHeight; ++y)
        {
            for (uint32_t x = 0; x < gridWidth; ++x)
            {
                for (uint32_t a = 0; a < numAnchors; ++a)
                {
                    auto currAnchorDim = ANCHOR_DIMS[i][a];
                    float tx, ty, tw, th, to;
                    tx = pOutput[GetIndex(a, 0, y, x, gridWidth, gridHeight)];
                    ty = pOutput[GetIndex(a, 1, y, x, gridWidth, gridHeight)];
                    tw = pOutput[GetIndex(a, 2, y, x, gridWidth, gridHeight)];
                    th = pOutput[GetIndex(a, 3, y, x, gridWidth, gridHeight)];
                    to = pOutput[GetIndex(a, 4, y, x, gridWidth, gridHeight)]; // objectness

                    float bx, by, bw, bh, bo;
                    // Box dimensions are relative to detector dimension. Later scale up by input resolution
                    bw = (currAnchorDim[0] * exp(tw));  // box width
                    bh = (currAnchorDim[1] * exp(th)); // box height
                    // Coordinates are relative to the grid. Scale to detector dimension. 
                    // Later scale up by input resolution.
                    bx = (((Sigmoid(tx) + x) * DETECTOR_WIDTH) / gridWidth) - (bw / 2);     // top left x coordinate
                    by = (((Sigmoid(ty) + y) * DETECTOR_HEIGHT) / gridHeight) - (bh / 2);    // top left y coordinate
                    bo = Sigmoid(to);

                    // Find the appropriate class
                    float maxProb = 0.0f;
                    uint32_t maxIndex = 0;
                    for (uint32_t c = 0; c < NUM_CLASSES; ++c)
                    {
                        float prob = Sigmoid(pOutput[GetIndex(a, 5 + c, y, x, gridWidth, gridHeight)]);
                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxIndex = c;
                        }
                    }
                    maxProb *= bo;
                    if (maxProb >= KEEP_THRESHOLD)
                    {
                        BoundingBox box;
                        box.x = bx;
                        box.y = by;
                        box.w = bw;
                        box.h = bh;
                        box.class_ = maxIndex;
                        box.prob = maxProb;
                        bBoxes.push_back(box);
                    }
                }
            }
        }
    }
    return bBoxes;
}

/*
Do Non maximal suppression based on certain threshold. Note that this function has no notion of class.
The callee is responsible to send a collection of bouding boxes belonging to the same class.
Params:
inBoxes: A vector of BoudingBox on which NMS needs to be performed
nmsThreshold: An IOU(interesection over union) threshold above which the box with lower class probability is discarded.
Return:
vector<BoundingBox>: A vector of filtered bounding boxes.
*/
std::vector<BoundingBox> CDetector::DoNMS(std::vector<BoundingBox>& inBoxes, float nmsThreshold)
{
    std::vector<BoundingBox> outBoxes;
    // Sort based on class probabilities
    std::sort(inBoxes.begin(), inBoxes.end(), [](const BoundingBox& b1, const BoundingBox& b2) { return b1.prob > b2.prob; });
    auto computeIOU = [](const BoundingBox& b1, const BoundingBox& b2) -> float
    {
        // cordinates of interesection rectangle (top left and bottom right)
        float xLeft = std::max(b1.x, b2.x);
        float yLeft = std::max(b1.y, b2.y);
        float xRight = std::min(b1.x + b1.w, b2.x + b2.w);
        float yRight = std::min(b1.y + b1.h, b2.y + b2.h);

        float width = std::max(0.0f, xRight - xLeft + 1);
        float height = std::max(0.0f, yRight - yLeft + 1);

        float intersectionArea = width * height;
        float totalArea = b1.w * b1.h + b2.w * b2.h;
        
        return intersectionArea/(totalArea - intersectionArea);
    };
    for (const auto& in : inBoxes)
    {
        bool keep = true;
        for (auto& out : outBoxes)
        {
            if (keep)
            {
                auto iou = computeIOU(in, out);
                keep = iou <= nmsThreshold;
            }
            else
            {
                break;
            }
            
        }
        if (keep)
        {
            outBoxes.push_back(in);
        }
    }

    return outBoxes;
}

/*
TRT results are flat arrays. Interpret and Post process the flat array to get useful data.
Params:
outputPointers: A vector of host void pointers to output surfaces.
Returns:
vector<BoundingBox>: Flat output array interpreted as bounding boxes. A vector of all the bounding boxes
*/
std::vector<BoundingBox> CDetector::PostprocessResult(const std::vector<void*>& outputPointers)
{
    std::vector<BoundingBox> bBoxes, bBoxesFinal;
    bBoxes = GetCandidateBoxes(outputPointers);

    // Create a set of Bounding Boxes based on class id as the key. This will create a unique list.
    std::set<BoundingBox, std::function<bool(const BoundingBox&, const BoundingBox&)>>
    boxSet([](const BoundingBox& b1, const BoundingBox& b2) -> bool { return b1.class_ > b2.class_;});
    std::copy(bBoxes.begin(), bBoxes.end(), std::inserter(boxSet, boxSet.begin()));
    for (auto& v : boxSet)
    {
        std::vector<BoundingBox> boxes; 
        std::copy_if(bBoxes.begin(), bBoxes.end(), std::back_inserter(boxes), [&v](const BoundingBox b){ return b.class_ == v.class_; });
        auto outBoxes = DoNMS(boxes, NMS_THRESHOLD);
        bBoxesFinal.insert(bBoxesFinal.end(), outBoxes.begin(), outBoxes.end());
    }

    return bBoxesFinal;
}

/*
Function which calls TRT manager to do the actual inference. Sets up the input and output buffers in format
needed by TRT manager.
Params:
inputFrame: A host pointer to inputFrame.
props: Frame properties
Returns:
vector<BoundingBox> : A vector contianing all the detected bouding boxes
*/
// Assuming input is RGB with channel first (HWC)
std::vector<BoundingBox> CDetector::Run(void* inputFrame, FrameProps props)
{
    std::vector<std::vector<float>> output(3);
    std::vector<void*> inputPointers;
    std::vector<void*> outputPointers;

    switch (props.channels)
    {
        case 3:
            break;
        default:
            std::ostringstream oss;
            oss << "Invalid value for channels field. Supported value is 3. Got: " << props.channels;
            NVOFTSAMPLE_THROW_ERROR(oss.str());
            break;
    }

    inputPointers = { inputFrame };
    auto inSize = m_TRTManager->GetInputSize();
    auto outSize = m_TRTManager->GetOutputSize();
    output[0].reserve(outSize[0]/sizeof(float));
    output[1].reserve(outSize[1]/sizeof(float));
    output[2].reserve(outSize[2]/sizeof(float));
    outputPointers = { output[0].data(), output[1].data(), output[2].data() };

    m_TRTManager->RunInference(inputPointers, outputPointers); 

    std::vector<BoundingBox> bBoxes;
    bBoxes = PostprocessResult(outputPointers);

    return bBoxes;
}
