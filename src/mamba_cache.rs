// MambaCache: manages per-sequence conv and recurrent states for Mamba/GDN layers
//
// For hybrid models (e.g., Qwen3.5), GatedDeltaNet layers require:
// - conv_state:  [max_batch, d_conv, conv_kernel_size - 1] per GDN layer
// - recurrent_state: [max_batch, num_heads, head_dim, head_dim] per GDN layer
//
// The cache uses slot-based indexing: each sequence is assigned a slot index,
// and states are updated in-place during forward passes.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct ScatterRowsUpdate {
    slots: Tensor,
}

#[cfg(feature = "cuda")]
impl candle_core::InplaceOp2 for ScatterRowsUpdate {
    fn name(&self) -> &'static str {
        "mamba-scatter-rows-update"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<()> {
        candle_core::bail!("mamba-scatter-rows-update is CUDA only")
    }

    fn cuda_fwd(
        &self,
        dst: &mut candle_core::CudaStorage,
        dst_layout: &candle_core::Layout,
        src: &candle_core::CudaStorage,
        src_layout: &candle_core::Layout,
    ) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::DType;
        use kernels::ffi;

        let num_rows = src_layout.shape().dims()[0];
        if num_rows == 0 {
            return Ok(());
        }
        if src.dtype() != dst.dtype() {
            candle_core::bail!(
                "mamba scatter dtype mismatch: src={:?} dst={:?}",
                src.dtype(),
                dst.dtype()
            );
        }

        let row_elems = src_layout.shape().elem_count() / num_rows;

        let src_row_stride = src_layout.stride()[0] as i64;
        let dst_row_stride = dst_layout.stride()[0] as i64;

        let src_off = src_layout.start_offset();
        let dst_off = dst_layout.start_offset();

        let elem_size = src.dtype().size_in_bytes();
        let src_ptr = match &src.slice {
            CudaStorageSlice::BF16(s) => {
                ((*s.device_ptr() as usize) + src_off * elem_size) as *const core::ffi::c_void
            }
            CudaStorageSlice::F16(s) => {
                ((*s.device_ptr() as usize) + src_off * elem_size) as *const core::ffi::c_void
            }
            CudaStorageSlice::F32(s) => {
                ((*s.device_ptr() as usize) + src_off * elem_size) as *const core::ffi::c_void
            }
            _ => candle_core::bail!("Unsupported src dtype for mamba scatter"),
        };
        let dst_ptr = match &dst.slice {
            CudaStorageSlice::BF16(s) => {
                ((*s.device_ptr() as usize) + dst_off * elem_size) as *mut core::ffi::c_void
            }
            CudaStorageSlice::F16(s) => {
                ((*s.device_ptr() as usize) + dst_off * elem_size) as *mut core::ffi::c_void
            }
            CudaStorageSlice::F32(s) => {
                ((*s.device_ptr() as usize) + dst_off * elem_size) as *mut core::ffi::c_void
            }
            _ => candle_core::bail!("Unsupported dst dtype for mamba scatter"),
        };

        let (slots_storage, slots_layout) = self.slots.storage_and_layout();
        let slots = match &*slots_storage {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<i64>()?,
            _ => candle_core::bail!("slots tensor must be a CUDA tensor"),
        };
        let slots = slots.slice(slots_layout.start_offset()..);
        if slots_layout.shape().elem_count() != num_rows {
            candle_core::bail!(
                "slots length mismatch in mamba scatter: slots={} rows={}",
                slots_layout.shape().elem_count(),
                num_rows
            );
        }
        let slots_ptr = *slots.device_ptr() as *const core::ffi::c_long;
        let stream = *dst.device().cu_stream() as i64;

        unsafe {
            match dst.dtype() {
                DType::F16 => ffi::mamba_scatter_rows_f16(
                    src_ptr,
                    dst_ptr,
                    slots_ptr,
                    num_rows as i32,
                    row_elems as i32,
                    src_row_stride,
                    dst_row_stride,
                    stream,
                ),
                DType::BF16 => ffi::mamba_scatter_rows_bf16(
                    src_ptr,
                    dst_ptr,
                    slots_ptr,
                    num_rows as i32,
                    row_elems as i32,
                    src_row_stride,
                    dst_row_stride,
                    stream,
                ),
                DType::F32 => ffi::mamba_scatter_rows_f32(
                    src_ptr,
                    dst_ptr,
                    slots_ptr,
                    num_rows as i32,
                    row_elems as i32,
                    src_row_stride,
                    dst_row_stride,
                    stream,
                ),
                dtype => candle_core::bail!("Unsupported dtype for mamba scatter: {dtype:?}"),
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn scatter_rows_cuda(dst: &Tensor, slots: &Tensor, src: &Tensor) -> Result<()> {
    let num_slots = slots.dim(0)?;
    if num_slots == 0 {
        return Ok(());
    }
    if !matches!(dst.device(), Device::Cuda(_))
        || !matches!(src.device(), Device::Cuda(_))
        || !matches!(slots.device(), Device::Cuda(_))
    {
        candle_core::bail!("Mamba cache updates require CUDA tensors");
    }
    if slots.dtype() != DType::I64 {
        candle_core::bail!(
            "mamba scatter slot dtype mismatch: expected I64, got {:?}",
            slots.dtype()
        );
    }
    if src.dim(0)? != num_slots {
        candle_core::bail!(
            "mamba scatter source rows mismatch: rows={} slots={}",
            src.dim(0)?,
            num_slots
        );
    }
    if src.dtype() != dst.dtype() {
        candle_core::bail!(
            "mamba scatter dtype mismatch: src={:?} dst={:?}",
            src.dtype(),
            dst.dtype()
        );
    }

    let src_rows = src.dim(0)?;
    let dst_rows = dst.dim(0)?;
    let src_row_elems = src.elem_count() / src_rows;
    let dst_row_elems = dst.elem_count() / dst_rows;
    if src_row_elems != dst_row_elems {
        candle_core::bail!(
            "mamba scatter row width mismatch: src={} dst={}",
            src_row_elems,
            dst_row_elems
        );
    }

    let src_2d = src.contiguous()?.reshape((src_rows, src_row_elems))?;
    let dst_2d = dst.reshape((dst_rows, dst_row_elems))?;
    dst_2d.inplace_op2(
        &src_2d,
        &ScatterRowsUpdate {
            slots: slots.clone(),
        },
    )?;
    Ok(())
}

pub struct MambaCache {
    /// Per-layer conv states: [max_batch, d_conv, conv_kernel_size - 1]
    conv_states: Vec<Tensor>,
    /// Per-layer recurrent states: [max_batch, num_heads, head_dim, head_dim]
    recurrent_states: Vec<Tensor>,
    /// Available slot indices
    free_slots: Vec<usize>,
    /// Mapping: sequence_id → slot_index
    seq_to_slot: HashMap<usize, usize>,
    /// Maximum batch size (number of concurrent sequences)
    max_batch_size: usize,
    /// Number of GDN layers
    num_gdn_layers: usize,
    /// Max number of cached prefix-state snapshots.
    prefix_cache_capacity: usize,
    /// Prefix hash -> captured slot states.
    prefix_states: HashMap<u64, PrefixStateSnapshot>,
    /// LRU queue for prefix-state eviction.
    prefix_lru: VecDeque<u64>,
}

#[derive(Clone)]
struct PrefixStateSnapshot {
    conv_states: Vec<Tensor>,
    recurrent_states: Vec<Tensor>,
}

impl MambaCache {
    /// Create a new MambaCache for GDN/Mamba layers
    ///
    /// Arguments:
    /// - num_gdn_layers: number of GDN layers in the model
    /// - max_batch_size: maximum number of concurrent sequences
    /// - d_conv: convolution dimension (typically intermediate_size)
    /// - conv_kernel_size: convolution kernel size (typically 4)
    /// - num_heads: number of GDN attention heads
    /// - head_k_dim: key head dimension for GDN recurrence state
    /// - head_v_dim: value head dimension for GDN recurrence state
    /// - conv_dtype: data type for conv state tensors
    /// - recurrent_dtype: data type for recurrent state tensors
    /// - device: computation device
    pub fn new(
        num_gdn_layers: usize,
        max_batch_size: usize,
        d_conv: usize,
        conv_kernel_size: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_dtype: DType,
        recurrent_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut conv_states = Vec::with_capacity(num_gdn_layers);
        let mut recurrent_states = Vec::with_capacity(num_gdn_layers);

        for _ in 0..num_gdn_layers {
            conv_states.push(Tensor::zeros(
                (max_batch_size, d_conv, conv_kernel_size - 1),
                conv_dtype,
                device,
            )?);
            recurrent_states.push(Tensor::zeros(
                (max_batch_size, num_heads, head_k_dim, head_v_dim),
                recurrent_dtype,
                device,
            )?);
        }

        let free_slots: Vec<usize> = (0..max_batch_size).rev().collect();

        Ok(Self {
            conv_states,
            recurrent_states,
            free_slots,
            seq_to_slot: HashMap::new(),
            max_batch_size,
            num_gdn_layers,
            prefix_cache_capacity: 0,
            prefix_states: HashMap::new(),
            prefix_lru: VecDeque::new(),
        })
    }

    /// Allocate a cache slot for a new sequence
    /// Returns the slot index, or an error if no slots are available
    pub fn allocate_slot(&mut self, seq_id: usize) -> Result<usize> {
        if let Some(&existing) = self.seq_to_slot.get(&seq_id) {
            return Ok(existing);
        }
        if self.free_slots.is_empty() {
            candle_core::bail!(
                "MambaCache: no free slots (max_batch_size={}), increase preallocated capacity",
                self.max_batch_size
            );
        }
        let slot = self.free_slots.pop().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "MambaCache: no free slots (max_batch_size={})",
                self.max_batch_size
            ))
        })?;
        self.seq_to_slot.insert(seq_id, slot);
        Ok(slot)
    }

    fn expand_capacity(&mut self, new_max_batch_size: usize) -> Result<()> {
        if new_max_batch_size <= self.max_batch_size {
            return Ok(());
        }
        let old_max_batch_size = self.max_batch_size;
        for layer_idx in 0..self.num_gdn_layers {
            let conv = &self.conv_states[layer_idx];
            let conv_dim = conv.dim(1)?;
            let conv_window = conv.dim(2)?;
            let mut expanded_conv = Tensor::zeros(
                (new_max_batch_size, conv_dim, conv_window),
                conv.dtype(),
                &conv.device(),
            )?;
            expanded_conv = expanded_conv
                .slice_assign(&[0..old_max_batch_size, 0..conv_dim, 0..conv_window], conv)?;
            self.conv_states[layer_idx] = expanded_conv;

            let rec = &self.recurrent_states[layer_idx];
            let rec_heads = rec.dim(1)?;
            let rec_h = rec.dim(2)?;
            let rec_w = rec.dim(3)?;
            let mut expanded_rec = Tensor::zeros(
                (new_max_batch_size, rec_heads, rec_h, rec_w),
                rec.dtype(),
                &rec.device(),
            )?;
            expanded_rec = expanded_rec.slice_assign(
                &[0..old_max_batch_size, 0..rec_heads, 0..rec_h, 0..rec_w],
                rec,
            )?;
            self.recurrent_states[layer_idx] = expanded_rec;
        }
        self.free_slots
            .extend((old_max_batch_size..new_max_batch_size).rev());
        self.max_batch_size = new_max_batch_size;
        Ok(())
    }

    /// Ensure the cache is preallocated for at least `new_max_batch_size` sequences.
    /// This is intended to be called during engine/model initialization so that
    /// inference can reuse preallocated state buffers.
    pub fn reserve_capacity(&mut self, new_max_batch_size: usize) -> Result<()> {
        self.expand_capacity(new_max_batch_size)
    }

    pub fn ensure_slots_for_sequences(&mut self, seq_ids: &[usize]) -> Result<Vec<usize>> {
        for &seq_id in seq_ids {
            if self.seq_to_slot.contains_key(&seq_id) {
                continue;
            }
            self.allocate_slot(seq_id)?;
        }
        seq_ids
            .iter()
            .map(|id| {
                self.seq_to_slot.get(id).copied().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "MambaCache: sequence {} not found in cache",
                        id
                    ))
                })
            })
            .collect()
    }

    /// Resolve slots for known sequences without allocating new slots.
    pub fn get_slots_for_sequences(&self, seq_ids: &[usize]) -> Result<Vec<usize>> {
        seq_ids
            .iter()
            .map(|id| {
                self.seq_to_slot.get(id).copied().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "MambaCache: sequence {} not found in cache",
                        id
                    ))
                })
            })
            .collect()
    }

    /// Free a cache slot when a sequence is done
    pub fn free_slot(&mut self, seq_id: usize) {
        if let Some(slot) = self.seq_to_slot.remove(&seq_id) {
            // Zero out the state for this slot
            let _ = self.reset_slot_states(slot);
            self.free_slots.push(slot);
        }
    }

    /// Reset (zero out) all states for a given slot
    fn reset_slot_states(&mut self, slot: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let device = self
                .conv_states
                .first()
                .map(|t| t.device().clone())
                .or_else(|| self.recurrent_states.first().map(|t| t.device().clone()))
                .ok_or_else(|| candle_core::Error::Msg("MambaCache has no layers".to_string()))?;
            let slot_tensor = Tensor::from_vec(vec![slot as i64], (1,), &device)?;

            for layer_idx in 0..self.num_gdn_layers {
                let conv_dim = self.conv_states[layer_idx].dim(1)?;
                let conv_window = self.conv_states[layer_idx].dim(2)?;
                let conv_zeros = Tensor::zeros(
                    (1, conv_dim, conv_window),
                    self.conv_states[layer_idx].dtype(),
                    self.conv_states[layer_idx].device(),
                )?;
                scatter_rows_cuda(&self.conv_states[layer_idx], &slot_tensor, &conv_zeros)?;

                let rec_heads = self.recurrent_states[layer_idx].dim(1)?;
                let rec_h = self.recurrent_states[layer_idx].dim(2)?;
                let rec_w = self.recurrent_states[layer_idx].dim(3)?;
                let rec_zeros = Tensor::zeros(
                    (1, rec_heads, rec_h, rec_w),
                    self.recurrent_states[layer_idx].dtype(),
                    self.recurrent_states[layer_idx].device(),
                )?;
                scatter_rows_cuda(&self.recurrent_states[layer_idx], &slot_tensor, &rec_zeros)?;
            }
            return Ok(());
        }

        #[cfg(not(feature = "cuda"))]
        {
            for layer_idx in 0..self.num_gdn_layers {
                let conv_dim = self.conv_states[layer_idx].dim(1)?;
                let conv_window = self.conv_states[layer_idx].dim(2)?;
                let conv_zeros = Tensor::zeros(
                    (1, conv_dim, conv_window),
                    self.conv_states[layer_idx].dtype(),
                    self.conv_states[layer_idx].device(),
                )?;
                let conv_updated = self.conv_states[layer_idx]
                    .slice_assign(&[slot..slot + 1, 0..conv_dim, 0..conv_window], &conv_zeros)?;
                self.conv_states[layer_idx] = conv_updated;

                let rec_heads = self.recurrent_states[layer_idx].dim(1)?;
                let rec_h = self.recurrent_states[layer_idx].dim(2)?;
                let rec_w = self.recurrent_states[layer_idx].dim(3)?;
                let rec_zeros = Tensor::zeros(
                    (1, rec_heads, rec_h, rec_w),
                    self.recurrent_states[layer_idx].dtype(),
                    self.recurrent_states[layer_idx].device(),
                )?;
                let rec_updated = self.recurrent_states[layer_idx].slice_assign(
                    &[slot..slot + 1, 0..rec_heads, 0..rec_h, 0..rec_w],
                    &rec_zeros,
                )?;
                self.recurrent_states[layer_idx] = rec_updated;
            }
            Ok(())
        }
    }

    /// Get the conv state tensor for a given GDN layer and slot
    /// Returns a view of shape [d_conv, conv_kernel_size - 1]
    pub fn get_conv_state(&self, gdn_layer_idx: usize, slot: usize) -> Result<Tensor> {
        self.conv_states[gdn_layer_idx].i(slot)
    }

    /// Get the recurrent state tensor for a given GDN layer and slot
    /// Returns a view of shape [num_heads, head_dim, head_dim]
    pub fn get_recurrent_state(&self, gdn_layer_idx: usize, slot: usize) -> Result<Tensor> {
        self.recurrent_states[gdn_layer_idx].i(slot)
    }

    /// Get mutable reference to the full conv state tensor for a layer
    /// Shape: [max_batch, d_conv, conv_kernel_size - 1]
    pub fn conv_state_mut(&mut self, gdn_layer_idx: usize) -> &mut Tensor {
        &mut self.conv_states[gdn_layer_idx]
    }

    /// Get mutable reference to the full recurrent state tensor for a layer
    /// Shape: [max_batch, num_heads, head_dim, head_dim]
    pub fn recurrent_state_mut(&mut self, gdn_layer_idx: usize) -> &mut Tensor {
        &mut self.recurrent_states[gdn_layer_idx]
    }

    /// Get reference to the full conv state tensor for a layer
    pub fn conv_state(&self, gdn_layer_idx: usize) -> &Tensor {
        &self.conv_states[gdn_layer_idx]
    }

    /// Get reference to the full recurrent state tensor for a layer
    pub fn recurrent_state(&self, gdn_layer_idx: usize) -> &Tensor {
        &self.recurrent_states[gdn_layer_idx]
    }

    pub fn get_batch_conv_state(&self, gdn_layer_idx: usize, slots: &Tensor) -> Result<Tensor> {
        if slots.dim(0)? == 0 {
            candle_core::bail!("MambaCache: empty slot list for conv state");
        }
        if slots.dtype() != DType::I64 {
            candle_core::bail!(
                "MambaCache: conv slot tensor must be I64, got {:?}",
                slots.dtype()
            );
        }
        self.conv_states[gdn_layer_idx].index_select(slots, 0)
    }

    pub fn set_batch_conv_state(
        &mut self,
        gdn_layer_idx: usize,
        slots: &Tensor,
        batch_state: &Tensor,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            return scatter_rows_cuda(&self.conv_states[gdn_layer_idx], slots, batch_state);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (gdn_layer_idx, slots, batch_state);
            candle_core::bail!("Mamba cache updates require building attention-rs with CUDA");
        }
    }

    pub fn get_batch_recurrent_state(
        &self,
        gdn_layer_idx: usize,
        slots: &Tensor,
    ) -> Result<Tensor> {
        if slots.dim(0)? == 0 {
            candle_core::bail!("MambaCache: empty slot list for recurrent state");
        }
        if slots.dtype() != DType::I64 {
            candle_core::bail!(
                "MambaCache: recurrent slot tensor must be I64, got {:?}",
                slots.dtype()
            );
        }
        self.recurrent_states[gdn_layer_idx].index_select(slots, 0)
    }

    pub fn set_batch_recurrent_state(
        &mut self,
        gdn_layer_idx: usize,
        slots: &Tensor,
        batch_state: &Tensor,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            return scatter_rows_cuda(&self.recurrent_states[gdn_layer_idx], slots, batch_state);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (gdn_layer_idx, slots, batch_state);
            candle_core::bail!("Mamba cache updates require building attention-rs with CUDA");
        }
    }

    /// Get the slot index for a sequence
    pub fn get_slot(&self, seq_id: usize) -> Option<usize> {
        self.seq_to_slot.get(&seq_id).copied()
    }

    /// Get slotted states for a batch of sequences (for CUDA kernel calls)
    /// Returns tensors indexed by the slot indices for the given sequence IDs
    pub fn get_batch_indices(&self, seq_ids: &[usize]) -> Result<Vec<usize>> {
        seq_ids
            .iter()
            .map(|id| {
                self.seq_to_slot.get(id).copied().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "MambaCache: sequence {} not found in cache",
                        id
                    ))
                })
            })
            .collect()
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn num_gdn_layers(&self) -> usize {
        self.num_gdn_layers
    }

    pub fn num_active_sequences(&self) -> usize {
        self.seq_to_slot.len()
    }

    pub fn set_prefix_cache_capacity(&mut self, capacity: usize) {
        self.prefix_cache_capacity = capacity;
        if capacity == 0 {
            self.prefix_states.clear();
            self.prefix_lru.clear();
            return;
        }
        self.evict_prefix_states_if_needed();
    }

    pub fn has_prefix_state(&self, hash: u64) -> bool {
        self.prefix_cache_capacity > 0 && self.prefix_states.contains_key(&hash)
    }

    fn touch_prefix_state(&mut self, hash: u64) {
        self.prefix_lru.retain(|&h| h != hash);
        self.prefix_lru.push_back(hash);
    }

    fn evict_prefix_states_if_needed(&mut self) {
        while self.prefix_states.len() > self.prefix_cache_capacity {
            let Some(hash) = self.prefix_lru.pop_front() else {
                break;
            };
            if self.prefix_states.remove(&hash).is_some() {
                break;
            }
        }
    }

    pub fn capture_prefix_state(&mut self, seq_id: usize, hash: u64) -> Result<bool> {
        if self.prefix_cache_capacity == 0 {
            return Ok(false);
        }
        let Some(slot) = self.get_slot(seq_id) else {
            return Ok(false);
        };
        if self.num_gdn_layers == 0 {
            return Ok(false);
        }

        let device = self.conv_states[0].device();
        let slot_tensor = Tensor::from_vec(vec![slot as i64], (1,), device)?;
        let mut conv_states = Vec::with_capacity(self.num_gdn_layers);
        let mut recurrent_states = Vec::with_capacity(self.num_gdn_layers);
        for layer_idx in 0..self.num_gdn_layers {
            conv_states.push(self.get_batch_conv_state(layer_idx, &slot_tensor)?);
            recurrent_states.push(self.get_batch_recurrent_state(layer_idx, &slot_tensor)?);
        }

        self.prefix_states.insert(
            hash,
            PrefixStateSnapshot {
                conv_states,
                recurrent_states,
            },
        );
        self.touch_prefix_state(hash);
        self.evict_prefix_states_if_needed();
        Ok(true)
    }

    pub fn restore_prefix_state(&mut self, seq_id: usize, hash: u64) -> Result<bool> {
        if self.prefix_cache_capacity == 0 {
            return Ok(false);
        }
        let Some(snapshot) = self.prefix_states.get(&hash).cloned() else {
            return Ok(false);
        };
        let Some(slot) = self.get_slot(seq_id) else {
            return Ok(false);
        };
        if self.num_gdn_layers == 0 {
            return Ok(false);
        }
        if snapshot.conv_states.len() != self.num_gdn_layers
            || snapshot.recurrent_states.len() != self.num_gdn_layers
        {
            candle_core::bail!(
                "MambaCache prefix snapshot layers mismatch: got conv={} rec={} expected={}",
                snapshot.conv_states.len(),
                snapshot.recurrent_states.len(),
                self.num_gdn_layers
            );
        }

        let device = self.conv_states[0].device();
        let slot_tensor = Tensor::from_vec(vec![slot as i64], (1,), device)?;
        for layer_idx in 0..self.num_gdn_layers {
            self.set_batch_conv_state(layer_idx, &slot_tensor, &snapshot.conv_states[layer_idx])?;
            self.set_batch_recurrent_state(
                layer_idx,
                &slot_tensor,
                &snapshot.recurrent_states[layer_idx],
            )?;
        }
        self.touch_prefix_state(hash);
        Ok(true)
    }

    pub fn reset_all(&mut self) -> Result<()> {
        for layer_idx in 0..self.num_gdn_layers {
            self.conv_states[layer_idx].zero_()?;
            self.recurrent_states[layer_idx].zero_()?;
        }
        self.seq_to_slot.clear();
        self.free_slots = (0..self.max_batch_size).rev().collect();
        self.prefix_states.clear();
        self.prefix_lru.clear();
        Ok(())
    }
}
