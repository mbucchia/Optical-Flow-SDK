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

#include "CFramesProducer.h"
#include "ColorSpace.h"
#include "CNvOFTSampleException.h"

#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#include <npps.h>
#include <nppcore.h>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/highgui.hpp"

#include <assert.h>
#include <stdint.h>
#include <ostream>
#include <fstream>

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

CFramesProducer::CFramesProducer(std::string& file, uint32_t detectorWidth, uint32_t detectorHeight, int gpuId) :
    m_scaledWidth(detectorWidth),
    m_scaledHeight(detectorHeight)
{
    CK_CUDA(cudaSetDevice(gpuId));
    CK_CUDA(cudaFree(0));
    CK_CU(cuCtxGetCurrent(&m_context));

    m_demuxer.reset(new FFmpegDemuxer(file.c_str()));
    m_decoder.reset(new NvDecoder(m_context, true, FFmpeg2NvCodecId(m_demuxer->GetVideoCodec())));
    assert(m_demuxer->GetBitDepth() == 8);
    m_decodeWidth = m_demuxer->GetWidth();
    m_decodeHeight = m_demuxer->GetHeight();
    // Assuming 8bpp
    m_decodePitch = m_decodeWidth;

    m_RGBFrame = { AllocCudaMemory(m_decodeWidth * 3 * m_decodeHeight * sizeof(float)), m_cudaRelease };
    m_RGBScaledFrame = { AllocCudaMemory(m_scaledWidth * 3 * m_scaledHeight * sizeof(float)), m_cudaRelease };
    m_YScaledFrame = { AllocCudaMemory(m_scaledWidth * m_scaledHeight * sizeof(uint8_t)), m_cudaRelease };
}

int CFramesProducer::Decode(int& videoBytes)
{
    int numFrames = 0;
    uint8_t *pVideo = NULL;

    m_demuxer->Demux(&pVideo, &videoBytes);
    numFrames = m_decoder->Decode(pVideo, videoBytes);

    return numFrames;
}

void CFramesProducer::GetFrames(std::vector<void*>& frames)
{
    uint8_t* pDecFrame;

    pDecFrame = m_decoder->GetFrame();
    assert(m_decoder->GetBPP() == 1);

    // Produce input frame for detector (the current detector needs the input to be normalized by 255.of)
    Nv12ToRGBPlanar<RGB_F32>(pDecFrame, (uint32_t)m_decodePitch, (uint8_t*)m_RGBFrame.get(), (uint32_t)m_decodeWidth*sizeof(float), (uint32_t)m_decodeWidth, (uint32_t)m_decodeHeight, 255.0f, 0);

    if (nppGetStream() != NULL)
    {
        nppSetStream(NULL);
    }
    // Scale the detctor input frame to detector width and height
    const float* pSrc[] = { (float*)m_RGBFrame.get(), (float*)m_RGBFrame.get() + m_decodeWidth * m_decodeHeight, (float*)m_RGBFrame.get() + m_decodeWidth * m_decodeHeight * 2 };
    float* pDst[] = { (float*)m_RGBScaledFrame.get(), (float*)m_RGBScaledFrame.get() + m_scaledWidth * m_scaledHeight, (float*)m_RGBScaledFrame.get() + m_scaledWidth * m_scaledHeight * 2 };
    nppiResize_32f_P3R(pSrc, (int)m_decodeWidth * sizeof(float), { (int)m_decodeWidth, (int)m_decodeHeight }, { 0, 0, (int)m_decodeWidth, (int)m_decodeHeight }, pDst, (int)m_scaledWidth * sizeof(float), { (int)m_scaledWidth, (int)m_scaledHeight }, { 0, 0, (int)m_scaledWidth, (int)m_scaledHeight }, NPPI_INTER_CUBIC);
    // Produce input frame for tracker (scaled to detector width and height)
    nppiResize_8u_C1R(pDecFrame, (int)m_decodePitch, { (int)m_decodeWidth, (int)m_decodeHeight }, { 0, 0, (int)m_decodeWidth, (int)m_decodeHeight }, (uint8_t*)m_YScaledFrame.get(), (int)m_scaledWidth, { (int)m_scaledWidth, (int)m_scaledHeight }, { 0, 0, (int)m_scaledWidth, (int)m_scaledHeight }, NPPI_INTER_CUBIC);

    frames.resize(3);
    frames = { pDecFrame, m_RGBScaledFrame.get(), m_YScaledFrame.get() };
}
