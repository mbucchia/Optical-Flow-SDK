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

class NvOFUtilsD3D12 : public NvOFUtils
{
public:

    NvOFUtilsD3D12(ID3D12Device* pDevice, Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> pList, NV_OF_MODE eMode);
    virtual void Upsample(NvOFBuffer* srcBuffer, NvOFBuffer* dstBuffer, uint32_t nScaleFactor) override;

private:
    ID3D12Device* m_pDevice;
    ID3D12CommandQueue* m_pQ;

    void UpdateConstantBuffer(NV_OF_CONSTANTBUFFER cbData);
    void CreateConstantBuffer();

    uint32_t m_nSrcWidth;
    uint32_t m_nSrcHeight;
    uint32_t m_nDstWidth;
    uint32_t m_nDstHeight;
    UINT     su_handle_size;

    NV_OF_CONSTANTBUFFER m_cbData;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_constantBuffer;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_computeState;
};
