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

#include "NvOFD3D12.h"
#include "NvOFUtilsD3D12.h"
#include "NvOFD3DCommon.h"
#include "UpsampleCS.h"

using Microsoft::WRL::ComPtr;

bool operator==(const NV_OF_CONSTANTBUFFER& lhs, const NV_OF_CONSTANTBUFFER& rhs)
{
    if ((lhs.srcWidth == rhs.srcWidth) &&
        (lhs.srcHeight == rhs.srcHeight) &&
        (lhs.dstWidth == rhs.dstWidth) &&
        (lhs.dstHeight == rhs.dstHeight) &&
        (lhs.nScaleFactor == rhs.nScaleFactor))
        return true;
    else
        return false;
}

NvOFUtilsD3D12::NvOFUtilsD3D12(ID3D12Device* pDevice,  ComPtr<ID3D12GraphicsCommandList> pList, NV_OF_MODE eMode)
    : m_computeCommandList(pList),
    m_pDevice(pDevice),
    NvOFUtils(eMode)
{
    m_cbData = {};
    D3D12_DESCRIPTOR_HEAP_DESC suHeapDesc = {};
    suHeapDesc.NumDescriptors = 2;
    suHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    suHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    D3D_API_CALL(m_pDevice->CreateDescriptorHeap(&suHeapDesc, IID_PPV_ARGS(&m_descHeap)));
    su_handle_size = m_pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_DESCRIPTOR_RANGE srv_range = {};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 1;
    srv_range.BaseShaderRegister = 0;
    D3D12_DESCRIPTOR_RANGE uav_range = {};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 1;
    uav_range.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0; 
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable.NumDescriptorRanges = 1;
    params[2].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 3;
    desc.pParameters = params;

    ComPtr<ID3DBlob> blob;
    D3D_API_CALL(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr));
    D3D_API_CALL(m_pDevice->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
    
    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    
    computePsoDesc.pRootSignature = m_rootSignature.Get();
    computePsoDesc.CS = { g_UpsampleCS , sizeof(g_UpsampleCS)};

    D3D_API_CALL(m_pDevice->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computeState)));
    CreateConstantBuffer();
}

void NvOFUtilsD3D12::CreateConstantBuffer()
{
    uint32_t data[] = { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_BUFFER };
    desc.Width = sizeof(data);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc = { 1, 0 };
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES props = { D3D12_HEAP_TYPE_UPLOAD };

    D3D_API_CALL(m_pDevice->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_constantBuffer)));

}

void NvOFUtilsD3D12::UpdateConstantBuffer(NV_OF_CONSTANTBUFFER cbData)
{
    if (m_cbData == cbData)
    {
        return;
    }

    void* mapped = nullptr;
    m_constantBuffer->Map(0, nullptr, &mapped);
    std::memcpy(mapped, &cbData, sizeof(cbData));
    m_constantBuffer->Unmap(0, nullptr);
    m_cbData = cbData;
}

void NvOFUtilsD3D12::Upsample(NvOFBuffer* srcBuffer, NvOFBuffer* dstBuffer, uint32_t nScalefactor)
{
    NvOFBufferD3D12<RWPolicyDeviceAndHost>* pSrcBufferD3D12 = dynamic_cast<NvOFBufferD3D12<RWPolicyDeviceAndHost>*>(srcBuffer);
    NvOFBufferD3D12<RWPolicyDeviceAndHost>* pDstBufferD3D12 = dynamic_cast<NvOFBufferD3D12<RWPolicyDeviceAndHost>*>(dstBuffer);
    UpdateConstantBuffer({ srcBuffer->getWidth(), srcBuffer->getHeight(), dstBuffer->getWidth(), dstBuffer->getHeight(), nScalefactor, 0, 0, 0 });

    D3D12_CPU_DESCRIPTOR_HANDLE su_cpu_handle = m_descHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uav_cpu_handle = su_cpu_handle;
    uav_cpu_handle.ptr += su_handle_size;
    
    D3D12_GPU_DESCRIPTOR_HANDLE su_gpu_handle = m_descHeap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu_handle = su_gpu_handle;
    uav_gpu_handle.ptr += su_handle_size;
    m_pDevice->CreateShaderResourceView(pSrcBufferD3D12->getD3D12ResourceHandle(), nullptr, su_cpu_handle);
    
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
    desc.Format = NvOFBufferFormatToDxgiFormat(pDstBufferD3D12->getBufferFormat());
    desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipSlice = 0;
    m_pDevice->CreateUnorderedAccessView(pDstBufferD3D12->getD3D12ResourceHandle(), nullptr, &desc, uav_cpu_handle);

    ID3D12DescriptorHeap *const descriptor_heaps[] = { m_descHeap.Get()};
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pDstBufferD3D12->getD3D12ResourceHandle();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_computeCommandList->ResourceBarrier(1, &barrier);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pSrcBufferD3D12->getD3D12ResourceHandle();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_computeCommandList->ResourceBarrier(1, &barrier);

    m_computeCommandList->SetDescriptorHeaps(1, descriptor_heaps);
    m_computeCommandList->SetComputeRootSignature(m_rootSignature.Get());
    m_computeCommandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    m_computeCommandList->SetComputeRootDescriptorTable(1, su_gpu_handle);
    m_computeCommandList->SetComputeRootDescriptorTable(2, uav_gpu_handle);
    m_computeCommandList->SetPipelineState(m_computeState.Get());
    m_computeCommandList->Dispatch(dstBuffer->getWidth(), dstBuffer->getHeight(), 1);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pDstBufferD3D12->getD3D12ResourceHandle();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_computeCommandList->ResourceBarrier(1, &barrier);
    
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pSrcBufferD3D12->getD3D12ResourceHandle();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_computeCommandList->ResourceBarrier(1, &barrier);
} 