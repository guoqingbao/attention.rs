// MambaCache: manages per-sequence conv and recurrent states for Mamba/GDN layers
//
// For hybrid models (e.g., Qwen3.5), GatedDeltaNet layers require:
// - conv_state:  [max_batch, d_conv, conv_kernel_size - 1] per GDN layer
// - recurrent_state: [max_batch, num_heads, head_dim, head_dim] per GDN layer
//
// The cache uses slot-based indexing: each sequence is assigned a slot index,
// and states are updated in-place during forward passes.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::{HashMap, HashSet};

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
    /// - dtype: data type for state tensors
    /// - device: computation device
    pub fn new(
        num_gdn_layers: usize,
        max_batch_size: usize,
        d_conv: usize,
        conv_kernel_size: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut conv_states = Vec::with_capacity(num_gdn_layers);
        let mut recurrent_states = Vec::with_capacity(num_gdn_layers);

        for _ in 0..num_gdn_layers {
            conv_states.push(Tensor::zeros(
                (max_batch_size, d_conv, conv_kernel_size - 1),
                dtype,
                device,
            )?);
            recurrent_states.push(Tensor::zeros(
                (max_batch_size, num_heads, head_k_dim, head_v_dim),
                dtype,
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
        })
    }

    /// Allocate a cache slot for a new sequence
    /// Returns the slot index, or an error if no slots are available
    pub fn allocate_slot(&mut self, seq_id: usize) -> Result<usize> {
        if let Some(&existing) = self.seq_to_slot.get(&seq_id) {
            return Ok(existing);
        }
        if self.free_slots.is_empty() {
            let new_capacity = (self.max_batch_size * 2).max(1);
            self.expand_capacity(new_capacity)?;
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

    pub fn ensure_slots_for_sequences(&mut self, seq_ids: &[usize]) -> Result<Vec<usize>> {
        let active: HashSet<usize> = seq_ids.iter().copied().collect();
        for &seq_id in seq_ids {
            if self.seq_to_slot.contains_key(&seq_id) {
                continue;
            }
            if self.allocate_slot(seq_id).is_ok() {
                continue;
            }

            // Best effort reclamation for stale sequences if slot pressure appears.
            let stale_ids = self
                .seq_to_slot
                .keys()
                .copied()
                .filter(|id| !active.contains(id))
                .collect::<Vec<_>>();
            for stale_id in stale_ids {
                self.free_slot(stale_id);
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
        for layer_idx in 0..self.num_gdn_layers {
            let conv = &self.conv_states[layer_idx];
            let zeros_conv = Tensor::zeros(&conv.dims()[1..], conv.dtype(), &conv.device())?;
            self.conv_states[layer_idx] = self.conv_states[layer_idx].slice_assign(
                &[slot..slot + 1, 0..conv.dim(1)?, 0..conv.dim(2)?],
                &zeros_conv.unsqueeze(0)?,
            )?;

            let rec = &self.recurrent_states[layer_idx];
            let zeros_rec = Tensor::zeros(&rec.dims()[1..], rec.dtype(), &rec.device())?;
            self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].slice_assign(
                &[
                    slot..slot + 1,
                    0..rec.dim(1)?,
                    0..rec.dim(2)?,
                    0..rec.dim(3)?,
                ],
                &zeros_rec.unsqueeze(0)?,
            )?;
        }
        Ok(())
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

    pub fn get_batch_conv_state(&self, gdn_layer_idx: usize, slots: &[usize]) -> Result<Tensor> {
        if slots.is_empty() {
            candle_core::bail!("MambaCache: empty slot list for conv state");
        }
        let slot_ids = slots.iter().map(|&s| s as u32).collect::<Vec<_>>();
        let index = Tensor::from_vec(
            slot_ids,
            (slots.len(),),
            self.conv_states[gdn_layer_idx].device(),
        )?;
        self.conv_states[gdn_layer_idx].index_select(&index, 0)
    }

    pub fn set_batch_conv_state(
        &mut self,
        gdn_layer_idx: usize,
        slots: &[usize],
        batch_state: &Tensor,
    ) -> Result<()> {
        for (batch_idx, &slot) in slots.iter().enumerate() {
            let state = batch_state.i(batch_idx)?.unsqueeze(0)?.contiguous()?;
            let dst = self.conv_states[gdn_layer_idx].narrow(0, slot, 1)?;
            dst.copy_(&state, 0)?;
        }
        Ok(())
    }

    pub fn get_batch_recurrent_state(
        &self,
        gdn_layer_idx: usize,
        slots: &[usize],
    ) -> Result<Tensor> {
        if slots.is_empty() {
            candle_core::bail!("MambaCache: empty slot list for recurrent state");
        }
        let slot_ids = slots.iter().map(|&s| s as u32).collect::<Vec<_>>();
        let index = Tensor::from_vec(
            slot_ids,
            (slots.len(),),
            self.recurrent_states[gdn_layer_idx].device(),
        )?;
        self.recurrent_states[gdn_layer_idx].index_select(&index, 0)
    }

    pub fn set_batch_recurrent_state(
        &mut self,
        gdn_layer_idx: usize,
        slots: &[usize],
        batch_state: &Tensor,
    ) -> Result<()> {
        for (batch_idx, &slot) in slots.iter().enumerate() {
            let state = batch_state.i(batch_idx)?.unsqueeze(0)?.contiguous()?;
            let dst = self.recurrent_states[gdn_layer_idx].narrow(0, slot, 1)?;
            dst.copy_(&state, 0)?;
        }
        Ok(())
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
}
