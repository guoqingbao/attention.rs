#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

#if defined(USE_FLASHINFER) && defined(FLASHINFER_ENABLE_FP8_E4M3)
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/quantization/prefill_sm90.cuh>
#include <flashinfer/page.cuh>
#include <cutlass/numeric_types.h>

using namespace flashinfer;

extern "C" void flashinfer_fp8_quantize_q_scalar(const void* input, void* output_q, int64_t numel,
                                                 const float* q_scale, bool is_input_f16,
                                                 int64_t stream_);
extern "C" void flashinfer_fp8_quantize_kv_scalar(const void* k_in, const void* v_in,
                                                  void* k_out, void* v_out, int64_t numel,
                                                  const float* k_scale, const float* v_scale,
                                                  bool is_input_f16, int64_t stream_);

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct FP8BatchPrefillPagedParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    DTypeQ* q_ptr;
    DTypeKV* k_ptr;
    DTypeKV* v_ptr;
    DTypeO* o_ptr;
    float* lse_ptr;

    IdType* qo_tile_indices;
    IdType* qo_indptr;
    IdType* kv_indptr;
    IdType* kv_indices;
    IdType* qo_lens;
    IdType* kv_lens;
    IdType* head_indices;
    IdType* work_indptr;
    IdType* batch_indices;

    struct AdditionalParams {
        float* maybe_scale_q;
        float* maybe_scale_k;
        float* maybe_scale_v;
        double sm_scale;
        double scale_q_scalar;
        double scale_k_scalar;
        double scale_v_scalar;
    } additional_params;

    int64_t q_stride_n;
    int64_t k_stride_n;
    int64_t v_stride_n;
    int64_t o_stride_n;
    int64_t q_stride_h;
    int64_t k_stride_h;
    int64_t v_stride_h;
    int64_t o_stride_h;
    int64_t nnz_qo;
    int64_t k_page_stride;
    int64_t v_page_stride;

    int head_dim;
    int num_qo_heads;
    int num_kv_heads;
    int group_size;
    int page_size;
    int window_left;

    bool causal;
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct FP8BatchPrefillRaggedParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    DTypeQ* q_ptr;
    DTypeKV* k_ptr;
    DTypeKV* v_ptr;
    DTypeO* o_ptr;
    float* lse_ptr;

    IdType* qo_tile_indices;
    IdType* qo_indptr;
    IdType* kv_indptr;
    IdType* qo_lens;
    IdType* kv_lens;
    IdType* head_indices;
    IdType* work_indptr;
    IdType* batch_indices;

    struct AdditionalParams {
        float* maybe_scale_q;
        float* maybe_scale_k;
        float* maybe_scale_v;
        double sm_scale;
        double scale_q_scalar;
        double scale_k_scalar;
        double scale_v_scalar;
    } additional_params;

    int64_t q_stride_n;
    int64_t k_stride_n;
    int64_t v_stride_n;
    int64_t o_stride_n;
    int64_t q_stride_h;
    int64_t k_stride_h;
    int64_t v_stride_h;
    int64_t o_stride_h;
    int64_t nnz_qo;
    int64_t nnz_kv;

    int head_dim;
    int num_qo_heads;
    int num_kv_heads;
    int group_size;
    int window_left;

    bool causal;
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
static inline void FillFP8PagedParams(
    FP8BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params,
    void* q_ptr,
    void* k_data,
    void* v_data,
    void* out_ptr,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    int64_t nnz_qo,
    float sm_scale,
    double q_scale_scalar,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    double k_scale_scalar,
    double v_scale_scalar,
    bool use_per_head_kv_scales,
    IdType* indices,
    void* workspace_int,
    const PrefillPlanSM90Info& plan_info
) {
    params.q_ptr = static_cast<DTypeQ*>(q_ptr);
    params.k_ptr = static_cast<DTypeKV*>(k_data);
    params.v_ptr = static_cast<DTypeKV*>(v_data);
    params.o_ptr = static_cast<DTypeO*>(out_ptr);
    params.lse_ptr = nullptr;
    params.q_stride_n = static_cast<int64_t>(num_qo_heads) * head_dim;
    params.q_stride_h = head_dim;
    params.o_stride_n = params.q_stride_n;
    params.o_stride_h = params.q_stride_h;
    params.k_stride_n = static_cast<int64_t>(num_kv_heads) * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = params.k_stride_n;
    params.v_stride_h = params.k_stride_h;
    params.k_page_stride = static_cast<int64_t>(page_size) * num_kv_heads * head_dim;
    params.v_page_stride = params.k_page_stride;
    params.nnz_qo = nnz_qo;
    params.head_dim = head_dim;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.page_size = page_size;
    params.window_left = -1;
    params.causal = true;
    params.additional_params.sm_scale = static_cast<double>(sm_scale);
    params.additional_params.maybe_scale_q = nullptr;
    params.additional_params.maybe_scale_k =
        use_per_head_kv_scales ? const_cast<float*>(k_scale_ptr) : nullptr;
    params.additional_params.maybe_scale_v =
        use_per_head_kv_scales ? const_cast<float*>(v_scale_ptr) : nullptr;
    params.additional_params.scale_q_scalar = q_scale_scalar;
    params.additional_params.scale_k_scalar = k_scale_scalar;
    params.additional_params.scale_v_scalar = v_scale_scalar;

    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
    params.qo_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_indptr_offset);
    params.kv_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_indptr_offset);
    params.qo_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_len_offset);
    params.kv_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_len_offset);
    params.head_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.head_indices_offset);
    params.work_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.work_indptr_offset);
    params.batch_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.batch_indices_offset);
    params.kv_indices = indices;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
static inline void FillFP8RaggedParams(
    FP8BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params,
    void* q_ptr,
    void* k_data,
    void* v_data,
    void* out_ptr,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int64_t nnz_qo,
    int64_t nnz_kv,
    float sm_scale,
    double q_scale_scalar,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    double k_scale_scalar,
    double v_scale_scalar,
    bool use_per_head_kv_scales,
    void* workspace_int,
    const PrefillPlanSM90Info& plan_info
) {
    params.q_ptr = static_cast<DTypeQ*>(q_ptr);
    params.k_ptr = static_cast<DTypeKV*>(k_data);
    params.v_ptr = static_cast<DTypeKV*>(v_data);
    params.o_ptr = static_cast<DTypeO*>(out_ptr);
    params.lse_ptr = nullptr;
    params.q_stride_n = static_cast<int64_t>(num_qo_heads) * head_dim;
    params.q_stride_h = head_dim;
    params.o_stride_n = params.q_stride_n;
    params.o_stride_h = params.q_stride_h;
    params.k_stride_n = static_cast<int64_t>(num_kv_heads) * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = params.k_stride_n;
    params.v_stride_h = params.k_stride_h;
    params.nnz_qo = nnz_qo;
    params.nnz_kv = nnz_kv;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.window_left = -1;
    params.causal = true;
    params.additional_params.sm_scale = static_cast<double>(sm_scale);
    params.additional_params.maybe_scale_q = nullptr;
    params.additional_params.maybe_scale_k = nullptr;
    params.additional_params.maybe_scale_v = nullptr;
    params.additional_params.scale_q_scalar = q_scale_scalar;
    params.additional_params.scale_k_scalar = k_scale_scalar;
    params.additional_params.scale_v_scalar = v_scale_scalar;
    params.head_dim = head_dim;

    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
    params.qo_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_indptr_offset);
    params.kv_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_indptr_offset);
    params.qo_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_len_offset);
    params.kv_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_len_offset);
    params.head_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.head_indices_offset);
    params.work_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.work_indptr_offset);
    params.batch_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.batch_indices_offset);
}

static inline bool ReadScaleScalar(const float* dev_ptr, double& out, cudaStream_t stream) {
    if (dev_ptr == nullptr) {
        return false;
    }
    float tmp = 1.0f;
    if (cudaMemcpyAsync(&tmp, dev_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }
    out = static_cast<double>(tmp);
    return true;
}

static inline bool CheckCudaLaunch(const char* tag) {
    cudaError_t st = cudaPeekAtLastError();
    if (st != cudaSuccess) {
        fprintf(stderr, "[flashinfer][fp8-ragged] %s launch failed: %s\n", tag, cudaGetErrorString(st));
        return false;
    }
    return true;
}

static inline bool CheckCudaSync(const char* tag, cudaStream_t stream) {
    cudaError_t st = cudaStreamSynchronize(stream);
    if (st != cudaSuccess) {
        fprintf(stderr, "[flashinfer][fp8-ragged] %s stream sync failed: %s\n", tag,
                cudaGetErrorString(st));
        return false;
    }
    return true;
}

static inline bool CheckCudaStatus(const char* tag, cudaError_t st) {
    if (st != cudaSuccess) {
        fprintf(stderr, "[flashinfer][fp8-ragged] %s failed: %s\n", tag, cudaGetErrorString(st));
        return false;
    }
    return true;
}

#define DISPATCH_HEAD_DIM_SM90(HEAD_DIM_VALUE, HEAD_DIM, ...) \
    if ((HEAD_DIM_VALUE) == 64) {                              \
        constexpr uint32_t HEAD_DIM = 64;                      \
        __VA_ARGS__;                                           \
    } else if ((HEAD_DIM_VALUE) == 128) {                      \
        constexpr uint32_t HEAD_DIM = 128;                     \
        __VA_ARGS__;                                           \
    } else if ((HEAD_DIM_VALUE) == 256) {                      \
        constexpr uint32_t HEAD_DIM = 256;                     \
        __VA_ARGS__;                                           \
    } else {                                                   \
        return;                                                \
    }

extern "C" {

void flashinfer_append_kv_cache_fp8(
    void* k_data_ptr,
    void* v_data_ptr,
    void* new_k_ptr,
    void* new_v_ptr,
    int32_t* paged_kv_indices,
    int32_t* paged_kv_indptr,
    int32_t* paged_kv_last_len,
    int32_t* batch_indices,
    int32_t* positions,
    int32_t nnz,
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t page_size,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    bool is_input_f16,
    int32_t data_type,
    cudaStream_t stream
) {
    if (data_type != 2 || !k_scale_ptr || !v_scale_ptr) {
        return;
    }
    void* k_fp8_ptr = nullptr;
    void* v_fp8_ptr = nullptr;
    int64_t numel = static_cast<int64_t>(nnz) * num_heads * head_dim;
    cudaMallocAsync(&k_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream);
    cudaMallocAsync(&v_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream);
    flashinfer_fp8_quantize_kv_scalar(
        new_k_ptr,
        new_v_ptr,
        k_fp8_ptr,
        v_fp8_ptr,
        numel,
        k_scale_ptr,
        v_scale_ptr,
        is_input_f16,
        (int64_t)stream);

    paged_kv_t<uint8_t, int32_t> paged_kv(
        num_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
        (uint8_t*)k_data_ptr, (uint8_t*)v_data_ptr,
        paged_kv_indices, paged_kv_indptr, paged_kv_last_len
    );
    if (batch_size > 0 && batch_indices && positions) {
        size_t stride_n = num_heads * head_dim;
        size_t stride_h = head_dim;
        AppendPagedKVCache(paged_kv, (uint8_t*)k_fp8_ptr, (uint8_t*)v_fp8_ptr,
                           batch_indices, positions, nnz,
                           stride_n, stride_h, stride_n, stride_h, stream);
    } else {
        AppendPagedKVCacheDecode(paged_kv, (uint8_t*)k_fp8_ptr, (uint8_t*)v_fp8_ptr, stream);
    }

    if (k_fp8_ptr) {
        cudaFreeAsync(k_fp8_ptr, stream);
    }
    if (v_fp8_ptr) {
        cudaFreeAsync(v_fp8_ptr, stream);
    }
}

void flashinfer_decode_plan_wrapper_fp8(
    int32_t* indptr_host,
    int32_t* qo_indptr_host,
    int32_t* kv_len_arr_host,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    int64_t* plan_info_out,
    cudaStream_t stream
) {
    if (data_type != 2) {
        return;
    }
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (qo_indptr_host == nullptr || kv_len_arr_host == nullptr) {
        return;
    }
    PrefillPlanSM90Info plan_info;
    PrefillSM90Plan<int32_t>(
        workspace_float, workspace_float_size,
        workspace_int, page_locked_int_buffer, workspace_int_size,
        plan_info,
        qo_indptr_host, indptr_host, kv_len_arr_host,
        batch_size, batch_size,
        num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
        false, enable_cuda_graph,
        2,
        stream
    );
    if (plan_info_out != nullptr) {
        plan_info_out[0] = plan_info.qo_tile_indices_offset;
        plan_info_out[1] = plan_info.qo_indptr_offset;
        plan_info_out[2] = plan_info.kv_indptr_offset;
        plan_info_out[3] = plan_info.qo_len_offset;
        plan_info_out[4] = plan_info.kv_len_offset;
        plan_info_out[5] = plan_info.head_indices_offset;
        plan_info_out[6] = plan_info.work_indptr_offset;
        plan_info_out[7] = plan_info.batch_indices_offset;
        plan_info_out[8] = plan_info.same_schedule_for_all_heads;
    }
}

void flashinfer_decode_run_wrapper_fp8(
    void* out_ptr,
    void* q_ptr,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    const int64_t* plan_info_vec,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
    if (data_type != 2 || !plan_info_vec) {
        return;
    }
    if (!k_scale_ptr || !v_scale_ptr) {
        return;
    }
    using IdType = int32_t;
    std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 9);
    PrefillPlanSM90Info plan_info;
    plan_info.FromVector(vec);
    double k_scale_scalar = 1.0;
    double v_scale_scalar = 1.0;
    (void)ReadScaleScalar(k_scale_ptr, k_scale_scalar, stream);
    (void)ReadScaleScalar(v_scale_ptr, v_scale_scalar, stream);

    void* q_scale_ptr = nullptr;
    void* q_fp8_ptr = nullptr;
    double q_scale_scalar = 1.0;
    int64_t numel = static_cast<int64_t>(batch_size) * num_qo_heads * head_dim;
    cudaMallocAsync(&q_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream);
    cudaMallocAsync(&q_scale_ptr, sizeof(float), stream);

    bool is_input_f16 = (out_data_type == 0);
    flashinfer_fp8_quantize_q_scalar(
        q_ptr, q_fp8_ptr, numel, static_cast<float*>(q_scale_ptr), is_input_f16, (int64_t)stream);
    (void)ReadScaleScalar(static_cast<float*>(q_scale_ptr), q_scale_scalar, stream);

    using DTypeQ = cutlass::float_e4m3_t;
    using DTypeKV = cutlass::float_e4m3_t;
    using AttentionType = DefaultFP8Attention;
    if (out_data_type == 1) {
        using DTypeOut = cutlass::bfloat16_t;
        FP8BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_data, v_data, out_ptr,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            batch_size, sm_scale, q_scale_scalar, k_scale_ptr, v_scale_ptr,
            k_scale_scalar, v_scale_scalar, false,
            indices, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                    params, false, stream);
            } else {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                    params, false, stream);
            }
        });
    } else {
        using DTypeOut = cutlass::half_t;
        FP8BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_data, v_data, out_ptr,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            batch_size, sm_scale, q_scale_scalar, k_scale_ptr, v_scale_ptr,
            k_scale_scalar, v_scale_scalar, false,
            indices, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                    params, false, stream);
            } else {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                    params, false, stream);
            }
        });
    }

    if (q_fp8_ptr) {
        cudaFreeAsync(q_fp8_ptr, stream);
    }
    if (q_scale_ptr) {
        cudaFreeAsync(q_scale_ptr, stream);
    }
}

void flashinfer_prefill_wrapper_fp8(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t* q_cu_seqlens_host,
    int32_t* kv_len_arr_host,
    int32_t total_num_rows,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,
    int32_t* indptr_host,
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
    if (data_type != 2) {
        return;
    }
    if (!k_scale_ptr || !v_scale_ptr) {
        return;
    }
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (q_cu_seqlens_host == nullptr || indptr_host == nullptr || kv_len_arr_host == nullptr) {
        return;
    }

    using IdType = int32_t;
    double k_scale_scalar = 1.0;
    double v_scale_scalar = 1.0;
    (void)ReadScaleScalar(k_scale_ptr, k_scale_scalar, stream);
    (void)ReadScaleScalar(v_scale_ptr, v_scale_scalar, stream);
    PrefillPlanSM90Info plan_info;
    PrefillSM90Plan<int32_t>(
        workspace_float, workspace_float_size,
        workspace_int, page_locked_int_buffer, workspace_int_size,
        plan_info,
        q_cu_seqlens_host, indptr_host, kv_len_arr_host,
        total_num_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
        true, enable_cuda_graph,
        (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
        stream
    );

    void* q_scale_ptr = nullptr;
    void* q_fp8_ptr = nullptr;
    double q_scale_scalar = 1.0;
    int64_t numel = static_cast<int64_t>(total_num_rows) * num_qo_heads * head_dim;
    cudaMallocAsync(&q_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream);
    cudaMallocAsync(&q_scale_ptr, sizeof(float), stream);
    bool is_input_f16 = (out_data_type == 0);
    flashinfer_fp8_quantize_q_scalar(
        q_ptr, q_fp8_ptr, numel, static_cast<float*>(q_scale_ptr), is_input_f16, (int64_t)stream);
    (void)ReadScaleScalar(static_cast<float*>(q_scale_ptr), q_scale_scalar, stream);

    using DTypeQ = cutlass::float_e4m3_t;
    using DTypeKV = cutlass::float_e4m3_t;
    using AttentionType = DefaultFP8Attention;
    if (out_data_type == 1) {
        using DTypeOut = cutlass::bfloat16_t;
        FP8BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_data, v_data, out_ptr,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            total_num_rows, sm_scale, q_scale_scalar, k_scale_ptr, v_scale_ptr,
            k_scale_scalar, v_scale_scalar, false,
            indices, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                    params, false, stream);
            } else {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                    params, false, stream);
            }
        });
    } else {
        using DTypeOut = cutlass::half_t;
        FP8BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_data, v_data, out_ptr,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            total_num_rows, sm_scale, q_scale_scalar, k_scale_ptr, v_scale_ptr,
            k_scale_scalar, v_scale_scalar, false,
            indices, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                    params, false, stream);
            } else {
                BatchFP8PrefillWithPagedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                    params, false, stream);
            }
        });
    }

    if (q_fp8_ptr) {
        cudaFreeAsync(q_fp8_ptr, stream);
    }
    if (q_scale_ptr) {
        cudaFreeAsync(q_scale_ptr, stream);
    }
}

void flashinfer_prefill_ragged_wrapper_fp8(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t* kv_cu_seqlens,
    int32_t* q_cu_seqlens_host,
    int32_t* kv_cu_seqlens_host,
    int32_t total_num_rows,
    int32_t total_kv_rows,
    void* k_ptr,
    void* v_ptr,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
    if (data_type != 2 || !k_scale_ptr || !v_scale_ptr) {
        return;
    }
#if !defined(SM_90_PASS)
    return;
#else
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (q_cu_seqlens_host == nullptr || kv_cu_seqlens_host == nullptr ||
        q_cu_seqlens == nullptr || kv_cu_seqlens == nullptr) {
        return;
    }

    const int32_t q_last = q_cu_seqlens_host[batch_size];
    const int32_t kv_last = kv_cu_seqlens_host[batch_size];

    std::vector<int32_t> kv_len_host(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        kv_len_host[i] = kv_cu_seqlens_host[i + 1] - kv_cu_seqlens_host[i];
    }
    PrefillPlanSM90Info plan_info;
    cudaError_t plan_st = PrefillSM90Plan<int32_t>(
        workspace_float, workspace_float_size,
        workspace_int, page_locked_int_buffer, workspace_int_size,
        plan_info,
        q_cu_seqlens_host, kv_cu_seqlens_host, kv_len_host.data(),
        total_num_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, head_dim, 1,
        true, enable_cuda_graph,
        (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
        stream
    );

    double q_scale_scalar = 1.0;
    double k_scale_scalar = 1.0;
    double v_scale_scalar = 1.0;
    void* q_fp8_ptr = nullptr;
    void* q_scale_ptr = nullptr;
    const int64_t q_numel = static_cast<int64_t>(total_num_rows) * num_qo_heads * head_dim;
    cudaMallocAsync(&q_fp8_ptr, static_cast<size_t>(q_numel) * sizeof(uint8_t), stream);
    cudaMallocAsync(&q_scale_ptr, sizeof(float), stream);

    bool is_input_f16 = (out_data_type == 0);
    flashinfer_fp8_quantize_q_scalar(
        q_ptr, q_fp8_ptr, q_numel, static_cast<float*>(q_scale_ptr), is_input_f16, (int64_t)stream);
    (void)ReadScaleScalar(static_cast<float*>(q_scale_ptr), q_scale_scalar, stream);

    void* k_fp8_ptr = nullptr;
    void* v_fp8_ptr = nullptr;
    const int64_t kv_numel = static_cast<int64_t>(total_kv_rows) * num_kv_heads * head_dim;
    cudaMallocAsync(&k_fp8_ptr, static_cast<size_t>(kv_numel) * sizeof(uint8_t), stream);
    cudaMallocAsync(&v_fp8_ptr, static_cast<size_t>(kv_numel) * sizeof(uint8_t), stream);
    flashinfer_fp8_quantize_kv_scalar(
        k_ptr, v_ptr, k_fp8_ptr, v_fp8_ptr, kv_numel,
        k_scale_ptr, v_scale_ptr, is_input_f16, (int64_t)stream);
    (void)ReadScaleScalar(k_scale_ptr, k_scale_scalar, stream);
    (void)ReadScaleScalar(v_scale_ptr, v_scale_scalar, stream);

    using IdType = int32_t;
    using DTypeQ = cutlass::float_e4m3_t;
    using DTypeKV = cutlass::float_e4m3_t;
    using AttentionType = DefaultFP8Attention;
    if (out_data_type == 1) {
        using DTypeOut = cutlass::bfloat16_t;
        FP8BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8RaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_fp8_ptr, v_fp8_ptr, out_ptr, num_qo_heads, num_kv_heads,
            head_dim, total_num_rows, total_kv_rows, sm_scale, q_scale_scalar,
            k_scale_ptr, v_scale_ptr, k_scale_scalar, v_scale_scalar, false, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(params, true, stream);
            } else {
                BatchFP8PrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(params, true, stream);
            }
        });
    } else {
        using DTypeOut = cutlass::half_t;
        FP8BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillFP8RaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_fp8_ptr, k_fp8_ptr, v_fp8_ptr, out_ptr, num_qo_heads, num_kv_heads,
            head_dim, total_num_rows, total_kv_rows, sm_scale, q_scale_scalar,
            k_scale_ptr, v_scale_ptr, k_scale_scalar, v_scale_scalar, false, workspace_int, plan_info);
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchFP8PrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(params, true, stream);
            } else {
                BatchFP8PrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(params, true, stream);
            }
        });
    }
    if (q_fp8_ptr) cudaFreeAsync(q_fp8_ptr, stream);
    if (q_scale_ptr) cudaFreeAsync(q_scale_ptr, stream);
    if (k_fp8_ptr) cudaFreeAsync(k_fp8_ptr, stream);
    if (v_fp8_ptr) cudaFreeAsync(v_fp8_ptr, stream);
#endif
}

}  // extern "C"
#endif
