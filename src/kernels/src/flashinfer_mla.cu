#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <stdio.h>

#ifdef USE_FLASHINFER

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;

namespace mla_detail {

constexpr uint32_t HEAD_DIM_CKV = 512;
constexpr uint32_t HEAD_DIM_KPE = 64;

// Re-use the same DefaultDecodeAttention variant as flashinfer_adapter.cu
#if !defined(SM_90_PASS)
using DefaultDecodeAttentionMLA = DefaultAttention<false, false, false, false>;
#else
struct DefaultDecodeAttentionMLA {
    static constexpr bool use_softmax = true;
    uint32_t kv_len;
    uint32_t window_left;
    float sm_scale_log2;
    float soft_cap_pre_tanh_scale;
    bool use_logits_soft_cap;

    template <typename Params>
    __device__ __host__ DefaultDecodeAttentionMLA(const Params& params, uint32_t batch_idx,
                                                  uint8_t* smem_ptr) {
        (void)smem_ptr;
        kv_len = params.get_kv_len(batch_idx);
        window_left = (params.window_left >= 0) ? params.window_left : kv_len;
        use_logits_soft_cap = params.logits_soft_cap > 0.f;
        if (use_logits_soft_cap) {
            soft_cap_pre_tanh_scale = params.sm_scale / params.logits_soft_cap;
            sm_scale_log2 = math::log2e * params.logits_soft_cap;
        } else {
            soft_cap_pre_tanh_scale = 0.f;
            sm_scale_log2 = params.sm_scale * math::log2e;
        }
    }

    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                                 uint32_t qo_idx, uint32_t kv_idx,
                                                 uint32_t qo_head_idx, uint32_t kv_head_idx) {
        if (use_logits_soft_cap) {
            logits = math::tanh(logits * soft_cap_pre_tanh_scale);
        }
        return logits;
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
        return (kv_idx + 1 + window_left >= kv_len + qo_idx);
    }

    template <typename Params, typename T, typename T_M>
    __device__ __forceinline__ T OutputTransform(const Params& params, T output, uint32_t batch_idx,
                                                 uint32_t qo_idx, uint32_t qo_head_idx, T_M& m,
                                                 float& d, float scale) {
        float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
        return output * d_rcp;
    }
};
#endif

// ============================================================================
// MLA Decode: plan + run (two-phase, mirrors standard FlashInfer decode)
// ============================================================================

template <typename DType>
void mla_decode_plan_typed(
    const int32_t* kv_indptr_host,
    int32_t batch_size, int32_t num_qo_heads, int32_t page_size,
    void* float_workspace, size_t float_workspace_size,
    void* int_workspace, size_t int_workspace_size,
    void* page_locked_buffer, size_t page_locked_size,
    bool enable_cuda_graph,
    int64_t* plan_info_out, cudaStream_t stream)
{
    using DTypeQ = DType;
    using DTypeKV = DType;
    using DTypeOut = DType;
    using IdType = int32_t;
    using AttentionType = DefaultDecodeAttentionMLA;
    using ParamsType = BatchDecodeParamsMLA<DTypeQ, DTypeKV, DTypeOut, IdType>;

    DecodePlanInfo plan_info;
    DecodePlan<HEAD_DIM_CKV, PosEncodingMode::kNone, AttentionType, ParamsType>(
        float_workspace, float_workspace_size,
        int_workspace, page_locked_buffer, int_workspace_size,
        plan_info,
        const_cast<int32_t*>(kv_indptr_host), batch_size, num_qo_heads, page_size,
        enable_cuda_graph, stream,
        BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<
            HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionType, ParamsType>
    );

    if (plan_info_out != nullptr) {
        plan_info_out[0] = plan_info.padded_batch_size;
        plan_info_out[1] = plan_info.v_offset;
        plan_info_out[2] = plan_info.s_offset;
        plan_info_out[3] = plan_info.request_indices_offset;
        plan_info_out[4] = plan_info.kv_tile_indices_offset;
        plan_info_out[5] = plan_info.o_indptr_offset;
        plan_info_out[6] = plan_info.block_valid_mask_offset;
        plan_info_out[7] = plan_info.kv_chunk_size_ptr_offset;
        plan_info_out[8] = plan_info.enable_cuda_graph;
        plan_info_out[9] = plan_info.split_kv;
    }
}

template <typename DType>
void mla_decode_run_typed(
    void* o, void* q_nope, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    int32_t batch_size, int32_t num_qo_heads, int32_t page_size,
    float sm_scale, float rope_scale, float rope_theta,
    void* float_workspace, size_t float_workspace_size,
    void* int_workspace, size_t int_workspace_size,
    const int64_t* plan_info_vec, cudaStream_t stream)
{
    using DTypeQ = DType;
    using DTypeKV = DType;
    using DTypeOut = DType;
    using IdType = int32_t;
    using AttentionType = DefaultDecodeAttentionMLA;
    using ParamsType = BatchDecodeParamsMLA<DTypeQ, DTypeKV, DTypeOut, IdType>;

    paged_kv_mla_t<DTypeKV, IdType> paged_kv(
        page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
        static_cast<DTypeKV*>(ckv_cache), static_cast<DTypeKV*>(kpe_cache),
        const_cast<IdType*>(kv_indices), const_cast<IdType*>(kv_indptr),
        const_cast<IdType*>(kv_last_page_len));

    DecodePlanInfo plan_info;
    std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 10);
    plan_info.FromVector(vec);

    ParamsType params(
        static_cast<DTypeQ*>(q_nope), static_cast<DTypeQ*>(q_pe),
        nullptr /* q_rope_offset */, paged_kv, static_cast<DTypeOut*>(o),
        nullptr /* lse */, num_qo_heads,
        -1 /* window_left */, 0.f /* logits_soft_cap */,
        sm_scale, rope_scale, rope_theta);

    params.request_indices = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.request_indices_offset);
    params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);
    params.partition_kv = plan_info.split_kv;
    params.padded_batch_size = plan_info.padded_batch_size;
    params.block_valid_mask = nullptr;
    if (plan_info.split_kv && plan_info.enable_cuda_graph) {
        params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
    }

    DTypeOut* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        tmp_v = GetPtrFromBaseOffset<DTypeOut>(float_workspace, plan_info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_workspace, plan_info.s_offset);
    }

    cudaError_t status = BatchDecodeWithPagedKVCacheDispatchedMLA<
        HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionType, ParamsType>(
        params, tmp_v, tmp_s, false /* pdl */, stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "[flashinfer][mla_decode_run] failed: %s\n",
                cudaGetErrorString(status));
    }
}

// ============================================================================
// MLA Prefill: plan + run (two-phase, uses MLAPlan + BatchMLAPagedAttention)
// ============================================================================

template <typename DType>
void mla_prefill_run_typed(
    void* o, void* q_nope, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    const int32_t* kv_indices,
    int32_t num_heads, int32_t page_size, float sm_scale,
    void* float_workspace, size_t float_workspace_size,
    void* int_workspace, size_t int_workspace_size,
    const int64_t* plan_info_vec, bool causal, cudaStream_t stream)
{
    using DTypeQ = DType;
    using DTypeKV = DType;
    using DTypeOut = DType;
    using IdType = int32_t;
    using Params = MLAParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

    MLAPlanInfo plan_info;
    std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 18);
    plan_info.FromVector(vec);

    Params params;
    params.q_nope = static_cast<DTypeQ*>(q_nope);
    params.q_pe = static_cast<DTypeQ*>(q_pe);
    params.ckv = static_cast<DTypeKV*>(ckv_cache);
    params.kpe = static_cast<DTypeKV*>(kpe_cache);

    params.q_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_indptr_offset);
    params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_indptr_offset);
    params.partial_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.partial_indptr_offset);
    params.kv_indices = const_cast<IdType*>(kv_indices);
    params.q_len = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_len_offset);
    params.kv_len = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_len_offset);
    params.q_start = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_start_offset);
    params.kv_start = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_start_offset);
    params.kv_end = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_end_offset);
    params.work_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.work_indptr_offset);
    params.merge_packed_offset_start = GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.merge_packed_offset_start_offset);
    params.merge_packed_offset_end = GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.merge_packed_offset_end_offset);
    params.merge_partial_packed_offset_start = GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.merge_partial_packed_offset_start_offset);
    params.merge_partial_packed_offset_end = GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.merge_partial_packed_offset_end_offset);
    params.merge_partial_stride = GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.merge_partial_stride_offset);

    params.final_o = static_cast<DTypeOut*>(o);
    params.final_lse = nullptr;
    params.partial_o = GetPtrFromBaseOffset<DTypeOut>(float_workspace, plan_info.partial_o_offset);
    params.partial_lse = GetPtrFromBaseOffset<float>(float_workspace, plan_info.partial_lse_offset);

    params.num_heads = uint_fastdiv(num_heads);
    params.block_size = uint_fastdiv(page_size);

    // q_nope: [n, num_heads, HEAD_DIM_CKV], q_pe: [n, num_heads, HEAD_DIM_KPE]
    params.q_nope_stride_n = num_heads * HEAD_DIM_CKV;
    params.q_nope_stride_h = HEAD_DIM_CKV;
    params.q_pe_stride_n = num_heads * HEAD_DIM_KPE;
    params.q_pe_stride_h = HEAD_DIM_KPE;
    // ckv_cache: [num_pages, page_size, HEAD_DIM_CKV]
    params.ckv_stride_page = page_size * HEAD_DIM_CKV;
    params.ckv_stride_n = HEAD_DIM_CKV;
    // kpe_cache: [num_pages, page_size, HEAD_DIM_KPE]
    params.kpe_stride_page = page_size * HEAD_DIM_KPE;
    params.kpe_stride_n = HEAD_DIM_KPE;
    // o: [n, num_heads, HEAD_DIM_CKV]
    params.o_stride_n = num_heads * HEAD_DIM_CKV;
    params.o_stride_h = HEAD_DIM_CKV;

    params.sm_scale = sm_scale;
    params.return_lse_base_on_e = false;

    cudaError_t status;
    if (causal) {
        status = mla::BatchMLAPagedAttention<MaskMode::kCausal, HEAD_DIM_CKV, HEAD_DIM_KPE, Params>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);
    } else {
        status = mla::BatchMLAPagedAttention<MaskMode::kNone, HEAD_DIM_CKV, HEAD_DIM_KPE, Params>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);
    }
    if (status != cudaSuccess) {
        fprintf(stderr, "[flashinfer][mla_prefill_run] failed: %s\n",
                cudaGetErrorString(status));
    }
}

} // namespace mla_detail

using namespace mla_detail;

// ============================================================================
// C API
// ============================================================================

extern "C" {

void flashinfer_mla_decode_plan_wrapper(
    const int32_t* kv_indptr_host,
    int32_t batch_size, int32_t num_qo_heads, int32_t page_size,
    void* float_workspace, int64_t float_workspace_size,
    void* int_workspace, int64_t int_workspace_size,
    void* page_locked_buffer, int64_t page_locked_size,
    bool enable_cuda_graph, uint32_t dtype,
    int64_t* plan_info_out, cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (page_locked_buffer == nullptr || page_locked_size < int_workspace_size) {
        fprintf(stderr, "[flashinfer][mla_decode_plan] page_locked_buffer too small\n");
        return;
    }
    if (dtype == 1) {
        mla_decode_plan_typed<nv_bfloat16>(
            kv_indptr_host, batch_size, num_qo_heads, page_size,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            page_locked_buffer, page_locked_size,
            enable_cuda_graph, plan_info_out, stream);
    } else {
        mla_decode_plan_typed<half>(
            kv_indptr_host, batch_size, num_qo_heads, page_size,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            page_locked_buffer, page_locked_size,
            enable_cuda_graph, plan_info_out, stream);
    }
#endif
}

void flashinfer_mla_decode_run_wrapper(
    void* o, void* q_nope, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    int32_t batch_size, int32_t num_qo_heads, int32_t page_size,
    float sm_scale, float rope_scale, float rope_theta,
    void* float_workspace, int64_t float_workspace_size,
    void* int_workspace, int64_t int_workspace_size,
    const int64_t* plan_info,
    uint32_t dtype, cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (plan_info == nullptr) {
        fprintf(stderr, "[flashinfer][mla_decode_run] plan_info is null\n");
        return;
    }
    if (dtype == 1) {
        mla_decode_run_typed<nv_bfloat16>(
            o, q_nope, q_pe, ckv_cache, kpe_cache,
            kv_indptr, kv_indices, kv_last_page_len,
            batch_size, num_qo_heads, page_size,
            sm_scale, rope_scale, rope_theta,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            plan_info, stream);
    } else {
        mla_decode_run_typed<half>(
            o, q_nope, q_pe, ckv_cache, kpe_cache,
            kv_indptr, kv_indices, kv_last_page_len,
            batch_size, num_qo_heads, page_size,
            sm_scale, rope_scale, rope_theta,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            plan_info, stream);
    }
#endif
}

void flashinfer_mla_prefill_plan_wrapper(
    const int32_t* qo_indptr_host,
    const int32_t* kv_indptr_host,
    const int32_t* kv_len_arr_host,
    int32_t batch_size, int32_t num_heads, int32_t head_dim_ckv,
    bool causal,
    void* float_workspace, int64_t float_workspace_size,
    void* int_workspace, int64_t int_workspace_size,
    void* page_locked_buffer, int64_t page_locked_size,
    int64_t* plan_info_out, cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (page_locked_buffer == nullptr || page_locked_size < int_workspace_size) {
        fprintf(stderr, "[flashinfer][mla_prefill_plan] page_locked_buffer too small\n");
        return;
    }

    MLAPlanInfo plan_info;
    cudaError_t status = MLAPlan<int32_t>(
        float_workspace, static_cast<size_t>(float_workspace_size),
        int_workspace, page_locked_buffer, static_cast<size_t>(int_workspace_size),
        plan_info,
        const_cast<int32_t*>(qo_indptr_host),
        const_cast<int32_t*>(kv_indptr_host),
        const_cast<int32_t*>(kv_len_arr_host),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim_ckv),
        causal, stream);

    if (status != cudaSuccess) {
        fprintf(stderr, "[flashinfer][mla_prefill_plan] failed: %s\n",
                cudaGetErrorString(status));
        return;
    }

    if (plan_info_out != nullptr) {
        auto vec = plan_info.ToVector();
        for (int i = 0; i < 18; ++i) {
            plan_info_out[i] = vec[i];
        }
    }
#endif
}

void flashinfer_mla_prefill_run_wrapper(
    void* o, void* q_nope, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    const int32_t* kv_indices,
    int32_t num_heads, int32_t page_size, float sm_scale,
    void* float_workspace, int64_t float_workspace_size,
    void* int_workspace, int64_t int_workspace_size,
    const int64_t* plan_info,
    bool causal,
    uint32_t dtype, cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (plan_info == nullptr) {
        fprintf(stderr, "[flashinfer][mla_prefill_run] plan_info is null\n");
        return;
    }
    if (dtype == 1) {
        mla_prefill_run_typed<nv_bfloat16>(
            o, q_nope, q_pe, ckv_cache, kpe_cache, kv_indices,
            num_heads, page_size, sm_scale,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            plan_info, causal, stream);
    } else {
        mla_prefill_run_typed<half>(
            o, q_nope, q_pe, ckv_cache, kpe_cache, kv_indices,
            num_heads, page_size, sm_scale,
            float_workspace, float_workspace_size,
            int_workspace, int_workspace_size,
            plan_info, causal, stream);
    }
#endif
}

} // extern "C"

#endif // USE_FLASHINFER
