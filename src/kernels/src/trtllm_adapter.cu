#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(USE_FLASHINFER) && defined(ENABLE_TRTLLM)
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>

namespace {

enum class TllmPagedAttentionMode {
  Context,
  Decode,
};

class TllmRunnerCache {
 public:
  using Key = std::tuple<Data_type, Data_type, Data_type>;

  static std::shared_ptr<TllmGenFmhaRunner> get(Data_type q_data_type, Data_type kv_data_type,
                                                Data_type o_data_type) {
    static std::unordered_map<Key, std::shared_ptr<TllmGenFmhaRunner>, KeyHash> cache;
    static std::mutex cache_mutex;
    Key key = std::make_tuple(q_data_type, kv_data_type, o_data_type);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    auto runner = std::make_shared<TllmGenFmhaRunner>(q_data_type, kv_data_type, o_data_type);
    cache.emplace(key, runner);
    return runner;
  }

 private:
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return std::hash<int>()(static_cast<int>(std::get<0>(k))) ^
             (std::hash<int>()(static_cast<int>(std::get<1>(k))) << 1) ^
             (std::hash<int>()(static_cast<int>(std::get<2>(k))) << 2);
    }
  };
};

template <typename T>
static inline T round_up(T n, T factor) {
  return ((n + factor - 1) / factor) * factor;
}

static inline Data_type to_trtllm_dtype(int32_t dtype_code) {
  // 0: FP16, 1: BF16, 2: FP8(E4M3)
  if (dtype_code == 0) {
    return Data_type::DATA_TYPE_FP16;
  }
  if (dtype_code == 1) {
    return Data_type::DATA_TYPE_BF16;
  }
  if (dtype_code == 2) {
    return Data_type::DATA_TYPE_E4M3;
  }
  return Data_type::DATA_TYPE_UNKNOWN;
}

static void trtllm_paged_attention_launcher(
    void* out, void* query, void* key_cache, void* value_cache, void* workspace_buffer,
    int32_t* block_tables, int32_t* seq_lens, int32_t* cum_seq_lens_q, int32_t* cum_seq_lens_kv,
    Data_type q_data_type, Data_type kv_data_type, Data_type o_data_type,
    TllmPagedAttentionMode mode, int64_t batch_size, int64_t sum_seq_q, int64_t max_q_len,
    int64_t max_kv_len, int64_t num_pages, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, int64_t max_num_blocks_per_seq,
    double bmm1_scale, double bmm2_scale, int64_t window_left, bool enable_pdl,
    int64_t workspace_size, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    fprintf(stderr,
            "[trtllm][paged_attention] invalid heads: num_qo_heads=%lld num_kv_heads=%lld\n",
            static_cast<long long>(num_qo_heads), static_cast<long long>(num_kv_heads));
    return;
  }

  auto fmha_runner = TllmRunnerCache::get(q_data_type, kv_data_type, o_data_type);
  TllmGenFmhaRunnerParams params{};

  params.qPtr = query;
  params.kPtr = key_cache;
  params.vPtr = value_cache;
  params.kvPageIdxPtr = block_tables;
  params.seqLensKvPtr = seq_lens;
  params.oPtr = out;
  params.mHeadDimQk = static_cast<int>(head_dim_qk);
  params.mHeadDimV = static_cast<int>(head_dim_vo);
  params.mNumHeadsQ = static_cast<int>(num_qo_heads);
  params.mNumHeadsKv = static_cast<int>(num_kv_heads);
  params.mNumHeadsQPerKv = static_cast<int>(num_qo_heads / num_kv_heads);
  params.mBatchSize = static_cast<int>(batch_size);
  params.mMaxSeqLenKv = static_cast<int>(max_kv_len);
  const bool use_paged_kv = block_tables != nullptr;
  if (use_paged_kv) {
    params.mMaxNumPagesPerSeqKv = static_cast<int>(max_num_blocks_per_seq);
    params.mNumTokensPerPage = static_cast<int>(page_size);
    params.mQkvLayout = QkvLayout::PagedKv;
  } else {
    // Non-paged prefill path: use separate contiguous Q/K/V buffers.
    params.mMaxNumPagesPerSeqKv = 0;
    params.mNumTokensPerPage = 0;
    params.mQkvLayout = QkvLayout::SeparateQkv;
    params.mSumOfSeqLensKv = static_cast<int>(sum_seq_q);
  }
  params.mMultiProcessorCount = getMultiProcessorCount();
  params.qStrideTokens = static_cast<int>(num_qo_heads * head_dim_qk);
  params.qStrideHeads = static_cast<int>(head_dim_qk);
  params.kStrideKeysValues = static_cast<int>(num_kv_heads * head_dim_qk);
  params.kStrideHeads = static_cast<int>(head_dim_qk);
  params.vStrideKeysValues = static_cast<int>(num_kv_heads * head_dim_vo);
  params.vStrideHeads = static_cast<int>(head_dim_vo);
  if (use_paged_kv) {
    // NHD page layout: [page, token, head, dim]
    params.kStrideBatch = static_cast<int>(page_size * num_kv_heads * head_dim_qk);
    params.vStrideBatch = static_cast<int>(page_size * num_kv_heads * head_dim_vo);
    params.mNumPagesInMemPool = static_cast<int>(num_pages * 2);
  } else {
    // Ragged contiguous K/V in [sum_seq, head, dim].
    params.kStrideBatch = -1;
    params.vStrideBatch = -1;
    params.mNumPagesInMemPool = 0;
  }
  params.stream = stream;
  params.outputScale = static_cast<float>(bmm2_scale);
  params.outputScalePtr = nullptr;
  params.scaleSoftmaxLog2 = static_cast<float>(bmm1_scale * M_LOG2E);
  params.scaleSoftmaxLog2Ptr = nullptr;
  params.oSfPtr = nullptr;
  params.mSfStartTokenIdx = 0;
  params.mScaleSfO = -1;
  params.mChunkedAttentionSize = std::numeric_limits<int>::max();
  params.mAttentionWindowSize = window_left == -1 ? std::numeric_limits<int>::max()
                                                  : static_cast<int>(window_left + 1);
  params.mMaxSeqLenQ = static_cast<int>(max_q_len);
  params.mSumOfSeqLensQ = static_cast<int>(sum_seq_q);
  params.ptrAttentionSinks = nullptr;
  params.enable_pdl = enable_pdl;
  params.mSparseMla = false;
  params.mSparseMlaTopK = 0;

  if (mode == TllmPagedAttentionMode::Context) {
    params.mMaskType = TrtllmGenAttentionMaskType::Causal;
    params.mKernelType = FmhaKernelType::Context;
    params.mTileScheduler = TileScheduler::Persistent;
    params.mMultiCtasKvMode = false;
    params.cumSeqLensQPtr = cum_seq_lens_q;
    params.cumSeqLensKvPtr = cum_seq_lens_kv;
  } else {
    params.mMaskType = TrtllmGenAttentionMaskType::Causal;
    params.mKernelType = FmhaKernelType::Generation;
    params.mTileScheduler = TileScheduler::Static;
    params.mMultiCtasKvMode = true;
    params.cumSeqLensQPtr = cum_seq_lens_q;
    params.cumSeqLensKvPtr = nullptr;

    size_t max_batch_size = 8192;
    size_t max_num_qo_heads = 256;
    size_t num_semaphores = round_up(max_batch_size * max_num_qo_heads, static_cast<size_t>(8));
    size_t counter_bytes = num_semaphores * sizeof(uint32_t);
    if (workspace_size <= static_cast<int64_t>(counter_bytes + 16)) {
      fprintf(stderr,
              "[trtllm][decode] workspace too small: need > %zu, got %lld bytes\n",
              counter_bytes + 16, static_cast<long long>(workspace_size));
      return;
    }
    auto* workspace_bytes = static_cast<uint8_t*>(workspace_buffer);
    params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_bytes);
    params.multiCtasKvScratchPtr = reinterpret_cast<void*>(workspace_bytes + counter_bytes);
    cudaMemsetAsync(params.multiCtasKvCounterPtr, 0, counter_bytes, stream);
  }

  auto [supported, info] = fmha_runner->isSupportedWithInfo(params);
  if (!supported) {
    fprintf(stderr, "[trtllm][paged_attention] no compatible kernel: %s\n", info.c_str());
    return;
  }
  fmha_runner->run(params);
}

}  // namespace

namespace flashinfer::trtllm_cubin_loader {
std::string getCubin(const std::string& kernel_path, const std::string& /*sha256*/) {
  std::ifstream ifs(kernel_path, std::ios::binary);
  if (!ifs.good()) {
    throw std::runtime_error("Failed to open TRTLLM cubin: " + kernel_path);
  }
  std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return std::string(buffer.begin(), buffer.end());
}
}  // namespace flashinfer::trtllm_cubin_loader

// Needed by fmhaKernels.cuh for generation kernels with multi-CTA reduction.
#include "../trtllm/fmhaReduction.cu"

#endif  // USE_FLASHINFER && ENABLE_TRTLLM

extern "C" {

void trtllm_decode_run_wrapper(
    void* out_ptr,
    void* q_ptr,
    void* k_data,
    void* v_data,
    int32_t* block_tables,
    int32_t* seq_lens,
    int32_t* cum_seq_lens_q,
    int32_t batch_size,
    int32_t sum_seq_q,
    int32_t max_q_len,
    int32_t max_kv_len,
    int32_t max_num_blocks_per_seq,
    int32_t num_pages,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float bmm1_scale,
    float bmm2_scale,
    void* workspace,
    size_t workspace_size,
    int32_t q_dtype,
    int32_t kv_dtype,
    int32_t out_dtype,
    bool enable_pdl,
    int32_t window_left,
    cudaStream_t stream) {
#if defined(USE_FLASHINFER) && defined(ENABLE_TRTLLM)
  trtllm_paged_attention_launcher(
      out_ptr, q_ptr, k_data, v_data, workspace, block_tables, seq_lens, cum_seq_lens_q, nullptr,
      to_trtllm_dtype(q_dtype), to_trtllm_dtype(kv_dtype), to_trtllm_dtype(out_dtype),
      TllmPagedAttentionMode::Decode, batch_size, sum_seq_q, max_q_len, max_kv_len, num_pages,
      num_qo_heads, num_kv_heads, head_dim, head_dim, page_size, max_num_blocks_per_seq,
      bmm1_scale, bmm2_scale, window_left, enable_pdl, workspace_size, stream);
#endif
}

void trtllm_context_run_wrapper(
    void* out_ptr,
    void* q_ptr,
    void* k_data,
    void* v_data,
    int32_t* block_tables,
    int32_t* seq_lens,
    int32_t* cum_seq_lens_q,
    int32_t* cum_seq_lens_kv,
    int32_t batch_size,
    int32_t sum_seq_q,
    int32_t max_q_len,
    int32_t max_kv_len,
    int32_t max_num_blocks_per_seq,
    int32_t num_pages,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float bmm1_scale,
    float bmm2_scale,
    void* workspace,
    size_t workspace_size,
    int32_t q_dtype,
    int32_t kv_dtype,
    int32_t out_dtype,
    bool enable_pdl,
    int32_t window_left,
    cudaStream_t stream) {
#if defined(USE_FLASHINFER) && defined(ENABLE_TRTLLM)
  trtllm_paged_attention_launcher(
      out_ptr, q_ptr, k_data, v_data, workspace, block_tables, seq_lens, cum_seq_lens_q,
      cum_seq_lens_kv, to_trtllm_dtype(q_dtype), to_trtllm_dtype(kv_dtype),
      to_trtllm_dtype(out_dtype), TllmPagedAttentionMode::Context, batch_size, sum_seq_q,
      max_q_len, max_kv_len, num_pages, num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
      max_num_blocks_per_seq, bmm1_scale, bmm2_scale, window_left, enable_pdl, workspace_size,
      stream);
#endif
}

}  // extern "C"
