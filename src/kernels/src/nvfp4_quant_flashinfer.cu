/*
 * FlashInfer activation NVFP4 quant (SM120 / SM121).
 *
 * Matches flashinfer.fp4_quantize(..., backend="cuda") →
 * tensorrt_llm::kernels::invokeFP4Quantization with SWIZZLED_128x4 and
 * e4m3Max=448 (no custom uint8-126 clamp). SGLang uses this on SM12x
 * together with flashinfer.mm_fp4 (cutlass).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#if defined(USE_FLASHINFER) && defined(ENABLE_FP4_SM120) && \
    defined(ATTENTION_RS_USE_FLASHINFER_FP4_QUANT)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"

namespace {

template <typename T>
void launch_flashinfer_fp4_quant(
    const void* input,
    void* output_packed,
    void* output_sf_swizzled,
    const float* global_scale,
    int m,
    int k,
    cudaStream_t stream)
{
  int sm_count = tensorrt_llm::common::getMultiProcessorCount();
  tensorrt_llm::kernels::invokeFP4Quantization<T, 16>(
      1,
      m,
      k,
      static_cast<T const*>(input),
      global_scale,
      static_cast<int64_t*>(output_packed),
      static_cast<int32_t*>(output_sf_swizzled),
      false,  // useUE8M0 (NVFP4 uses E4M3 scales)
      tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4,
      sm_count,
      false,  // enable_pdl
      false,  // use_row_wise_scale
      false,  // inverse_scale: global_scale is already 1/input_scale
      stream);
}

}  // namespace

extern "C" {

void flashinfer_nvfp4_quantize_activation_f16(
    const void* input,
    void* output_packed,
    void* output_sf_swizzled,
    const float* global_scale,
    int M,
    int K,
    int64_t stream)
{
  launch_flashinfer_fp4_quant<half>(
      input,
      output_packed,
      output_sf_swizzled,
      global_scale,
      M,
      K,
      reinterpret_cast<cudaStream_t>(stream));
}

void flashinfer_nvfp4_quantize_activation_bf16(
    const void* input,
    void* output_packed,
    void* output_sf_swizzled,
    const float* global_scale,
    int M,
    int K,
    int64_t stream)
{
  launch_flashinfer_fp4_quant<__nv_bfloat16>(
      input,
      output_packed,
      output_sf_swizzled,
      global_scale,
      M,
      K,
      reinterpret_cast<cudaStream_t>(stream));
}

}  // extern "C"

#else  // stubs when FlashInfer SM120 quant is not compiled in

extern "C" {

void flashinfer_nvfp4_quantize_activation_f16(
    const void*, void*, void*, const float*, int, int, int64_t)
{
}

void flashinfer_nvfp4_quantize_activation_bf16(
    const void*, void*, void*, const float*, int, int, int64_t)
{
}

}  // extern "C"

#endif
