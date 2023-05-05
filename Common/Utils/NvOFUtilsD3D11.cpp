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


#include <D3DCompiler.h>
#include "NvOFD3D11.h"
#include "NvOFUtilsD3D11.h"
#include "UpsampleCS.h"
#include "NvOFD3DCommon.h"

NvOFUtilsD3D11::NvOFUtilsD3D11(ID3D11Device *pDevice, ID3D11DeviceContext* pContext, NV_OF_MODE eMode)
    : NvOFUtils(eMode)
    , m_pDevice(pDevice)
    , m_pContext(pContext)
    , m_pComputeShader(nullptr)
    , m_pConstBuffer(nullptr)
{
    ZeroMemory(m_pInputShaderResourceView, sizeof(m_pInputShaderResourceView));
    ZeroMemory(m_pOutputUnorderedAccessView, sizeof(m_pOutputUnorderedAccessView));
    CreateComputeShader();
    CreateConstantBuffer();
}

void NvOFUtilsD3D11::CreateComputeShader()
{
    D3D_API_CALL(m_pDevice->CreateComputeShader(g_UpsampleCS, sizeof(g_UpsampleCS), nullptr, &m_pComputeShader));
}

void NvOFUtilsD3D11::CreateConstantBuffer()
{
    uint32_t constantBufferData[] = { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    D3D11_BUFFER_DESC desc;
    desc.ByteWidth = sizeof(constantBufferData);
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;

    D3D11_SUBRESOURCE_DATA data;
    data.pSysMem = constantBufferData;
    data.SysMemPitch = 0;
    data.SysMemSlicePitch = 0;

    D3D_API_CALL(m_pDevice->CreateBuffer(&desc, &data, &m_pConstBuffer));
}

void NvOFUtilsD3D11::UpdateConstantBuffer(uint32_t nSrcWidth, uint32_t nSrcHeight,
                                          uint32_t nDstWidth, uint32_t nDstHeight,
                                          uint32_t nScaleFactor)
{
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));
        uint32_t data[] = { nSrcWidth, nSrcHeight, nDstWidth, nDstHeight, nScaleFactor, 0x0, 0x0, 0x0};
        m_pContext->Map(m_pConstBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        memcpy(mappedResource.pData, data, sizeof(data));
        m_pContext->Unmap(m_pConstBuffer, 0);
}

void NvOFUtilsD3D11::Upsample(NvOFBuffer* srcBuf, NvOFBuffer* dstBuf, uint32_t nScaleFactor)
{
    NvOFBufferD3D11* srcBufD3D11 = dynamic_cast<NvOFBufferD3D11*>(srcBuf);
    NvOFBufferD3D11* dstBufD3D11 = dynamic_cast<NvOFBufferD3D11*>(dstBuf);

    uint32_t nBufIdx = (m_eMode == NV_OF_MODE_OPTICALFLOW) ? 0 : 1;
    D3D_API_CALL(m_pDevice->CreateShaderResourceView(srcBufD3D11->getD3D11TextureHandle(), nullptr, &m_pInputShaderResourceView[nBufIdx]));
    D3D_API_CALL(m_pDevice->CreateUnorderedAccessView(dstBufD3D11->getD3D11TextureHandle(), nullptr, &m_pOutputUnorderedAccessView[nBufIdx]));
    UpdateConstantBuffer(srcBuf->getWidth(), srcBuf->getHeight(), dstBuf->getWidth(), dstBuf->getHeight(), nScaleFactor);

    m_pContext->CSSetShader(m_pComputeShader, nullptr, 0);
    m_pContext->CSSetShaderResources(nBufIdx, 1, &m_pInputShaderResourceView[nBufIdx]);
    m_pContext->CSSetConstantBuffers(0, 1, &m_pConstBuffer);
    m_pContext->CSSetUnorderedAccessViews(nBufIdx, 1, &m_pOutputUnorderedAccessView[nBufIdx], nullptr);

    m_pContext->Dispatch(dstBuf->getWidth(), dstBuf->getHeight(), 1);

    ID3D11ShaderResourceView* nullSRV[] = { nullptr };
    m_pContext->CSSetShaderResources(0, 1, nullSRV);
    ID3D11UnorderedAccessView* nullUAV[] = { nullptr };
    m_pContext->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
    ID3D11Buffer* nullCB[] = { nullptr };
    m_pContext->CSSetConstantBuffers(0, 1, nullCB);
    m_pContext->CSSetShader(nullptr, nullptr, 0);
}
