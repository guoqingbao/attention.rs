/*
 * FlashInfer's SM103 Store256 NVFP4 GEMM instantiations.
 *
 * The upstream implementation is generated from a Jinja template by
 * FlashInfer's Python build.  attention.rs uses FlashInfer headers directly,
 * so keep the small set of required instantiations here instead.  This file
 * is enabled only for SM103; SM100 uses the regular TMA epilogue path.
 */

#if defined(ENABLE_FP4_SM100) && defined(ATTENTION_RS_FLASHINFER_SM103)

#include "flashinfer/gemm/cutlass_gemm_configs.h"

#define FLASHINFER_SM103_GENERIC_STORE256_NAMESPACE 1
#define FLASHINFER_SM103_GENERIC_NOSMEM_EPILOGUE 1
#include "flashinfer/gemm/fp4_gemm_template_sm100.h"

namespace flashinfer {
namespace gemm {
namespace sm103_generic_store256 {

#define INSTANTIATE_SM103_STORE256(TYPE, CTA_M, CTA_N, CTA_K)                              \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 1, 1, _1SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 2, 1, _1SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 4, 1, _1SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 1, 1, _2SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 2, 1, _2SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 4, 1, _2SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 1, 1, _2SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 2, 1, _2SM)           \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 4, 1, _2SM)

#define INSTANTIATE_SM103_STORE256_ALL(TYPE) \
  INSTANTIATE_SM103_STORE256(TYPE, 128, 64, 128)  \
  INSTANTIATE_SM103_STORE256(TYPE, 128, 256, 128) \
  INSTANTIATE_SM103_STORE256(TYPE, 128, 128, 256) \
  INSTANTIATE_SM103_STORE256(TYPE, 128, 256, 256)

INSTANTIATE_SM103_STORE256_ALL(half)
INSTANTIATE_SM103_STORE256_ALL(__nv_bfloat16)

#undef INSTANTIATE_SM103_STORE256_ALL
#undef INSTANTIATE_SM103_STORE256

} // namespace sm103_generic_store256
} // namespace gemm
} // namespace flashinfer

#endif // ENABLE_FP4_SM100 && ATTENTION_RS_FLASHINFER_SM103
