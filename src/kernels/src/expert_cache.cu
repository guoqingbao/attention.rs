// MoE expert slot cache: device-side LRU + gather of missing experts.
// Both kernels use fixed launch shapes and read the miss count from device
// memory, so they can be captured into CUDA graphs.
#include <stdint.h>

#define EXPERT_EMPTY 0xFFFFFFFFu
#define LRU_THREADS 256
#define MAX_GATHER_BANKS 8

// state layout per layer: slot_of[E], expert_of[S], last_used[S], tick[1]
// misses: miss_expert[S], miss_slot[S], miss_count[1]; stats[2] = {lookups, misses}
__global__ void expert_lru_ensure_kernel(
    const uint32_t* __restrict__ ids, int n,
    uint32_t* __restrict__ slot_of, uint32_t* __restrict__ expert_of,
    uint32_t* __restrict__ last_used, uint32_t* __restrict__ tick_ptr,
    int num_slots,
    uint32_t* __restrict__ out_ids,
    uint32_t* __restrict__ miss_expert, uint32_t* __restrict__ miss_slot,
    uint32_t* __restrict__ miss_count, unsigned long long* __restrict__ stats) {
  __shared__ uint32_t s_tick;
  __shared__ uint32_t s_best_val[LRU_THREADS];
  __shared__ int s_best_idx[LRU_THREADS];
  __shared__ uint32_t s_misses;
  const int tid = threadIdx.x;
  if (tid == 0) {
    s_tick = *tick_ptr + 1;
    *tick_ptr = s_tick;
    s_misses = 0;
  }
  __syncthreads();
  const uint32_t tick = s_tick;

  // Pass 1: every resident expert of this call is marked used, so no victim is needed now.
  for (int i = tid; i < n; i += blockDim.x) {
    const uint32_t s = slot_of[ids[i]];
    if (s != EXPERT_EMPTY) last_used[s] = tick;
  }
  __syncthreads();

  // Pass 2: assign slots to misses in order; duplicates become hits once assigned.
  for (int i = 0; i < n; ++i) {
    const uint32_t e = ids[i];
    const bool need = slot_of[e] == EXPERT_EMPTY;  // uniform across the block
    __syncthreads();
    if (!need) continue;
    uint32_t best = 0xFFFFFFFFu;
    int best_idx = -1;
    for (int s = tid; s < num_slots; s += blockDim.x) {
      const uint32_t u = last_used[s];
      if (u < tick && u < best) { best = u; best_idx = s; }
    }
    s_best_val[tid] = best;
    s_best_idx[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride && s_best_val[tid + stride] < s_best_val[tid]) {
        s_best_val[tid] = s_best_val[tid + stride];
        s_best_idx[tid] = s_best_idx[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      // No victim means more distinct experts than slots: callers must stage instead.
      const int v = s_best_idx[0] < 0 ? 0 : s_best_idx[0];
      const uint32_t old = expert_of[v];
      if (old != EXPERT_EMPTY) slot_of[old] = EXPERT_EMPTY;
      expert_of[v] = e;
      slot_of[e] = v;
      last_used[v] = tick;
      if (s_misses < (uint32_t)num_slots) {
        miss_expert[s_misses] = e;
        miss_slot[s_misses] = v;
        s_misses++;
      }
    }
    __syncthreads();
  }

  // Pass 3: rewrite expert ids to slot ids.
  for (int i = tid; i < n; i += blockDim.x) out_ids[i] = slot_of[ids[i]];
  if (tid == 0) {
    *miss_count = s_misses;
    stats[0] += n;
    stats[1] += s_misses;
  }
}

struct GatherBanks {
  const uint8_t* src[MAX_GATHER_BANKS];  // device-accessible (mapped host or device)
  uint8_t* dst[MAX_GATHER_BANKS];
  unsigned long long expert_bytes[MAX_GATHER_BANKS];
};

// grid: (chunks, max_misses, num_banks); block x handles one chunk of one expert of one bank.
__global__ void expert_gather_banks_kernel(
    GatherBanks banks, const uint32_t* __restrict__ miss_expert,
    const uint32_t* __restrict__ miss_slot, const uint32_t* __restrict__ miss_count,
    unsigned long long chunk_bytes) {
  const uint32_t m = blockIdx.y;
  if (m >= *miss_count) return;
  const int b = blockIdx.z;
  const unsigned long long eb = banks.expert_bytes[b];
  const unsigned long long begin = (unsigned long long)blockIdx.x * chunk_bytes;
  if (begin >= eb) return;
  const unsigned long long end = begin + chunk_bytes < eb ? begin + chunk_bytes : eb;
  const uint8_t* src = banks.src[b] + (unsigned long long)miss_expert[m] * eb;
  uint8_t* dst = banks.dst[b] + (unsigned long long)miss_slot[m] * eb;
  const bool vec = ((eb | (uintptr_t)banks.src[b] | (uintptr_t)banks.dst[b]) & 15) == 0;
  if (vec) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src + begin);
    uint4* d4 = reinterpret_cast<uint4*>(dst + begin);
    const unsigned long long n4 = (end - begin) / 16;
    for (unsigned long long i = threadIdx.x; i < n4; i += blockDim.x) d4[i] = s4[i];
  } else {
    for (unsigned long long i = begin + threadIdx.x; i < end; i += blockDim.x) dst[i] = src[i];
  }
}

extern "C" void expert_lru_ensure(
    const uint32_t* ids, int n, uint32_t* slot_of, uint32_t* expert_of,
    uint32_t* last_used, uint32_t* tick, int num_slots, uint32_t* out_ids,
    uint32_t* miss_expert, uint32_t* miss_slot, uint32_t* miss_count,
    unsigned long long* stats, int64_t stream) {
  expert_lru_ensure_kernel<<<1, LRU_THREADS, 0, (cudaStream_t)stream>>>(
      ids, n, slot_of, expert_of, last_used, tick, num_slots, out_ids, miss_expert,
      miss_slot, miss_count, stats);
}

extern "C" void expert_gather_banks(
    const uint8_t* const* src, uint8_t* const* dst, const unsigned long long* expert_bytes,
    int num_banks, const uint32_t* miss_expert, const uint32_t* miss_slot,
    const uint32_t* miss_count, int max_misses, int64_t stream) {
  if (num_banks <= 0 || num_banks > MAX_GATHER_BANKS || max_misses <= 0) return;
  GatherBanks banks;
  unsigned long long max_eb = 0;
  for (int b = 0; b < num_banks; ++b) {
    banks.src[b] = src[b];
    banks.dst[b] = dst[b];
    banks.expert_bytes[b] = expert_bytes[b];
    if (expert_bytes[b] > max_eb) max_eb = expert_bytes[b];
  }
  const unsigned long long chunk = 64 * 1024;
  const unsigned int chunks = (unsigned int)((max_eb + chunk - 1) / chunk);
  const dim3 grid(chunks, max_misses, num_banks);
  expert_gather_banks_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
      banks, miss_expert, miss_slot, miss_count, chunk);
}
