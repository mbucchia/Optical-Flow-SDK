/*
* Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/


#pragma once

#include "NvOFUtils.h"

class NvOFUtilsD3D11 : public NvOFUtils
{
public:
    NvOFUtilsD3D11(ID3D11Device *pDevice, ID3D11DeviceContext* pContext, NV_OF_MODE eMode);
    void CreateComputeShader();
    void CreateConstantBuffer();
    void UpdateConstantBuffer(uint32_t nSrcWidth, uint32_t nSrcHeight, uint32_t nDstWidth, uint32_t nDstHeight, uint32_t nScaleFactor);
    virtual void Upsample(NvOFBuffer* srcBuffer, NvOFBuffer* dstBuffer, uint32_t nScaleFactor) override;

private:
    ID3D11Device* m_pDevice;
    ID3D11DeviceContext* m_pContext;
    ID3D11Buffer* m_pConstBuffer;
    ID3D11ComputeShader* m_pComputeShader;
    ID3D11ShaderResourceView* m_pInputShaderResourceView[2];
    ID3D11UnorderedAccessView* m_pOutputUnorderedAccessView[2];

    uint32_t m_nSrcWidth;
    uint32_t m_nSrcHeight;
    uint32_t m_nDstWidth;
    uint32_t m_nDstHeight;
};
