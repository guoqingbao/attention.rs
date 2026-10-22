//! Device-side MoE expert slot cache (kernels in `kernels/src/expert_cache.cu`).
//!
//! The LRU table lives on the GPU, so a decode step can make its routed experts
//! resident without reading routing ids back to the host (CUDA-graph safe).

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};
use kernels::ffi;

/// Sentinel for "no expert" / "no slot" in the LRU tables.
pub const EMPTY: u32 = u32::MAX;
const MAX_BANKS: usize = 8;

/// Per-layer LRU tables and the miss list of the last `ensure` call.
pub struct ExpertLru {
    pub num_slots: usize,
    slot_of: Tensor,   // [E] u32
    expert_of: Tensor, // [S] u32
    last_used: Tensor, // [S] u32
    tick: Tensor,      // [1] u32
    miss_expert: Tensor,
    miss_slot: Tensor,
    miss_count: Tensor, // [1] u32
    stats: Tensor,      // [2] i64: lookups, misses
}

fn ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(c) = &*storage else {
        candle_core::bail!("expert_cache: expected a CUDA tensor");
    };
    let (base, elem) = match t.dtype() {
        DType::U32 => (*c.as_cuda_slice::<u32>()?.device_ptr(), 4),
        DType::I64 => (*c.as_cuda_slice::<i64>()?.device_ptr(), 8),
        dt => candle_core::bail!("expert_cache: unsupported dtype {dt:?}"),
    };
    Ok(base + (layout.start_offset() * elem) as u64)
}

impl ExpertLru {
    pub fn new(num_experts: usize, num_slots: usize, dev: &Device) -> Result<Self> {
        let u32s = |n: usize, v: u32| Tensor::from_vec(vec![v; n], n, dev);
        Ok(Self {
            num_slots,
            slot_of: u32s(num_experts, EMPTY)?,
            expert_of: u32s(num_slots, EMPTY)?,
            last_used: u32s(num_slots, 0)?,
            tick: u32s(1, 0)?,
            miss_expert: u32s(num_slots, 0)?,
            miss_slot: u32s(num_slots, 0)?,
            miss_count: u32s(1, 0)?,
            stats: Tensor::zeros(2, DType::I64, dev)?,
        })
    }

    /// Marks the routed experts in `ids` (u32, any shape) used, assigns slots to misses
    /// (least recently used victims) and returns the ids rewritten to slot ids.
    /// The call must touch at most `num_slots` distinct experts.
    pub fn ensure(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.contiguous()?;
        let out = Tensor::zeros(ids.shape(), DType::U32, ids.device())?;
        let stream = *ids.device().as_cuda_device()?.cu_stream() as i64;
        unsafe {
            ffi::expert_lru_ensure(
                ptr(&ids)? as *const u32,
                ids.elem_count() as i32,
                ptr(&self.slot_of)? as *mut u32,
                ptr(&self.expert_of)? as *mut u32,
                ptr(&self.last_used)? as *mut u32,
                ptr(&self.tick)? as *mut u32,
                self.num_slots as i32,
                ptr(&out)? as *mut u32,
                ptr(&self.miss_expert)? as *mut u32,
                ptr(&self.miss_slot)? as *mut u32,
                ptr(&self.miss_count)? as *mut u32,
                ptr(&self.stats)? as *mut u64,
                stream,
            );
        }
        Ok(out)
    }

    /// Copies each miss of the last `ensure` from bank `b` (`src[b] + expert * bytes[b]`,
    /// device-accessible memory, e.g. mapped pinned host) to `dst[b] + slot * bytes[b]`.
    pub fn gather(&self, dev: &Device, src: &[u64], dst: &[u64], bytes: &[u64]) -> Result<()> {
        if src.len() != dst.len() || src.len() != bytes.len() || src.len() > MAX_BANKS {
            candle_core::bail!("expert_cache: invalid bank list");
        }
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        unsafe {
            ffi::expert_gather_banks(
                src.as_ptr() as *const *const u8,
                dst.as_ptr() as *const *mut u8,
                bytes.as_ptr(),
                src.len() as i32,
                ptr(&self.miss_expert)? as *const u32,
                ptr(&self.miss_slot)? as *const u32,
                ptr(&self.miss_count)? as *const u32,
                self.num_slots as i32,
                stream,
            );
        }
        Ok(())
    }

    /// Expert held by each slot (`EMPTY` when free). Synchronizes.
    pub fn expert_of_slots(&self) -> Result<Vec<u32>> {
        self.expert_of.to_vec1()
    }

    /// (lookups, misses) since creation. Synchronizes.
    pub fn stats(&self) -> Result<(u64, u64)> {
        let v = self.stats.to_vec1::<i64>()?;
        Ok((v[0] as u64, v[1] as u64))
    }
}
