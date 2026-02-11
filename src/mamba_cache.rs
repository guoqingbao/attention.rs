// MambaCache: manages per-sequence conv and recurrent states for Mamba/GDN layers
//
// For hybrid models (e.g., Qwen3.5), GatedDeltaNet layers require:
// - conv_state:  [max_batch, d_conv, conv_kernel_size - 1] per GDN layer
// - recurrent_state: [max_batch, num_heads, head_dim, head_dim] per GDN layer
//
// The cache uses slot-based indexing: each sequence is assigned a slot index,
// and states are updated in-place during forward passes.

use candle_core::{DType, Device, Result, Tensor, IndexOp};
use std::collections::HashMap;

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
    /// - head_dim: head dimension for GDN
    /// - dtype: data type for state tensors
    /// - device: computation device
    pub fn new(
        num_gdn_layers: usize,
        max_batch_size: usize,
        d_conv: usize,
        conv_kernel_size: usize,
        num_heads: usize,
        head_dim: usize,
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
                (max_batch_size, num_heads, head_dim, head_dim),
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
        let slot = self.free_slots.pop().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "MambaCache: no free slots (max_batch_size={})",
                self.max_batch_size
            ))
        })?;
        self.seq_to_slot.insert(seq_id, slot);
        Ok(slot)
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
            let zeros_conv = Tensor::zeros(
                &conv.dims()[1..],
                conv.dtype(),
                &conv.device(),
            )?;
            self.conv_states[layer_idx] = self.conv_states[layer_idx]
                .slice_assign(&[slot..slot + 1, 0..conv.dim(1)?, 0..conv.dim(2)?], &zeros_conv.unsqueeze(0)?)?;

            let rec = &self.recurrent_states[layer_idx];
            let zeros_rec = Tensor::zeros(
                &rec.dims()[1..],
                rec.dtype(),
                &rec.device(),
            )?;
            self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx]
                .slice_assign(&[slot..slot + 1, 0..rec.dim(1)?, 0..rec.dim(2)?, 0..rec.dim(3)?], &zeros_rec.unsqueeze(0)?)?;
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
