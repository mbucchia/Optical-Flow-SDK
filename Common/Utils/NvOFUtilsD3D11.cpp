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
