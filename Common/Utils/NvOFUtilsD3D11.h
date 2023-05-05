/*
* Copyright (c) 2018-2023 NVIDIA Corporation
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the software, and to permit persons to whom the
* software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
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
