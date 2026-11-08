
// SPDX-License-Identifier: Apache-2.0
// Experiment derived from comfy-kitchen0.2.40; see Kitchen-LICENSE/NOTICE.
#include <cassert>
#include <cuda_runtime.h>
#include "sparse_test.cuh"
__global__ void selected_attention(int8_t* q,int8_t* k,int8_t* v,half* o,
 float* qs,float* ks,float* vs,int n,int h,int padded,
 const int32_t* index,const int32_t* count,const int32_t* valid,int width) {
 qk_int_sv_i8_attn_body<true,128,128,16,128,128,DataType::kInt8,
 QuantGranularity::kPerThread,QuantGranularity::kPerThread,float,false,half,
 ComputeUnit::kCudaCore,MaskMode::kPreparedKey,false,true,false,false,true,true,uint32_t,false,true>(
 q,k,v,o,nullptr,qs,ks,vs,nullptr,nullptr,0,0,0,0,0,n,n,1,
 h*n*128,128,n*128,h*n*128,128,n*128,h*128*padded,128*padded,padded,
 h*n*128,128,n*128,0.08838834764831845f,nullptr,index,count,valid,width);
}
extern "C" int run_sparse(void* q,void* k,void* v,void* o,void* qs,void* ks,void* vs,
 int n,int h,int padded,void* index,void* count,void* valid,int width,void* stream) {
 cudaError_t e=cudaFuncSetAttribute(selected_attention,cudaFuncAttributeMaxDynamicSharedMemorySize,49152);
 if(e!=cudaSuccess)return e;
 // Kitchen indexes lane=threadIdx.x and warp=threadIdx.y, not linear tid.
 selected_attention<<<dim3((n+127)/128,h),dim3(32,8),49152,(cudaStream_t)stream>>>(
 (int8_t*)q,(int8_t*)k,(int8_t*)v,(half*)o,(float*)qs,(float*)ks,(float*)vs,n,h,padded,
 (int32_t*)index,(int32_t*)count,(int32_t*)valid,width);
 return cudaGetLastError();
}
extern "C" int resources(int* out) {
 cudaFuncAttributes a;int blocks=0;
 cudaError_t e=cudaFuncGetAttributes(&a,selected_attention);if(e!=cudaSuccess)return e;
 e=cudaFuncSetAttribute(selected_attention,cudaFuncAttributeMaxDynamicSharedMemorySize,49152);if(e!=cudaSuccess)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,selected_attention,256,49152);
 out[0]=a.numRegs;out[1]=a.localSizeBytes;out[2]=blocks;return e;
}
