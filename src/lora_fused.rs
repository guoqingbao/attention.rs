//! The fused batched LoRA shrink/expand GEMM (the multi-tenant serving).
//!
//! This is the attention-rs side of the xinfer LoRA fused kernel. It computes
//! the low-rank delta `delta = scale * (x @ A) @ B` for a batch of rows where
//! each row may belong to a different LoRA tenant (slot), in two fused phases
//! over a COMPACTED set of active slots:
//!
//! - `lora_shrink`: `buf[row, s, k] = scale[slot[s]] * sum_c x[row, c] * A[slot[s], c, k]`
//! - `lora_expand`: `out[row, o] = sum_k buf[row, s, k] * B[slot[s], k, o]`
//!
//! where `s = slot_map[row]` is the compacted slot index (0..num_active) and
//! `slot[s] = active_slots[s]` is the actual bank slot. A row whose `slot_map`
//! entry is the `BASE_SLOT` sentinel gets a zero delta (the base model).
//!
//! The A/B are the stacked banks: A is `[num_slots, in_features, rank]`, B is
//! `[num_slots, rank, out_features]`. The `scale` is `[num_slots]` (f32). The
//! scratch `buf` is `[rows, num_active, rank]` f32 (compacted, not the full
//! bank width). The accumulation is f32 (no precision loss in the low-rank
//! product), matching the vLLM Punica choice. The x/A/B may be f16/bf16/f32
//! (the model compute dtype); the `out` is written in its own dtype.
//!
//! The xinfer side (the `lora` branch) calls these from the
//! `attention_rs::lora_fused::cuda` module, falling back to the per-layer candle
//! matmul when the kernel is unavailable.

#[cfg(feature = "cuda")]
pub mod cuda {
    use candle_core::{bail, DType, Error, Result, Storage, Tensor};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;
    use kernels::ffi;

    /// The sentinel `slot_map` value for a base-model row (no LoRA delta). Must
    /// match the `LORA_FUSED_BASE_SLOT` in the kernel.
    pub const BASE_SLOT: u32 = 0xFFFFFFFF;

    /// Compute the LoRA shrink phase: `buf[row, s, k] = scale[slot[s]] *
    /// sum_c x[row, c] * A[slot[s], c, k]`. The `slot_map` is the `[rows]` u32
    /// compacted slot index (the `BASE_SLOT` sentinel = the base model). The
    /// `active_slots` is the `[num_active]` u32 map from the compacted index to
    /// the actual bank slot. The `buf` is the f32 scratch `[rows, num_active,
    /// rank]`, pre-allocated by the caller.
    pub fn lora_shrink(
        x: &Tensor,
        a: &Tensor,
        scale: &Tensor,
        slot_map: &Tensor,
        active_slots: &Tensor,
        buf: &Tensor,
    ) -> Result<()> {
        lora_shrink_impl(x, a, scale, slot_map, active_slots, buf)
    }

    /// Compute the LoRA expand phase: `out[row, o] = sum_k buf[row, s, k] *
    /// B[slot[s], k, o]`. The `out` is the fresh delta tensor (the caller adds
    /// it to the base output). The `buf` is the f32 scratch from the shrink phase.
    pub fn lora_expand(
        buf: &Tensor,
        b: &Tensor,
        slot_map: &Tensor,
        active_slots: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        lora_expand_impl(buf, b, slot_map, active_slots, out)
    }

    /// Compute the stacked (additive multi-adapter) shrink phase. For each
    /// (row, j) pair where j < stack_count[row]:
    ///   buf[row, j, k] = scale[bank_slot] * sum_c x[row, c] * A[bank_slot, c, k]
    /// where bank_slot = active_slots[slot_ids[slot_offsets[row] + j]].
    /// The `slot_offsets` is the `[rows+1]` u32 CSR row pointer; the `slot_ids`
    /// is the `[total]` u32 compacted slot index per (row, adapter) pair; the
    /// `buf` is the `[rows, max_stack, rank]` f32 scratch.
    pub fn lora_shrink_stacked(
        x: &Tensor,
        a: &Tensor,
        scale: &Tensor,
        slot_offsets: &Tensor,
        slot_ids: &Tensor,
        active_slots: &Tensor,
        buf: &Tensor,
    ) -> Result<()> {
        lora_shrink_stacked_impl(x, a, scale, slot_offsets, slot_ids, active_slots, buf)
    }

    /// Compute the stacked (additive multi-adapter) expand phase:
    ///   out[row, o] = sum_j sum_k buf[row, j, k] * B[bank_slot(j), k, o]
    /// which is the additive composition base + sum_j scaling_j * (x @ A_j) @ B_j.
    /// The `out` is the fresh delta tensor (the caller adds it to the base output).
    pub fn lora_expand_stacked(
        buf: &Tensor,
        b: &Tensor,
        slot_offsets: &Tensor,
        slot_ids: &Tensor,
        active_slots: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        lora_expand_stacked_impl(buf, b, slot_offsets, slot_ids, active_slots, out)
    }

    // The dtype dispatch: select the f16/bf16/f32 launcher based on the input
    // dtype (the model compute dtype). The A/B/x/y all share the dtype; the
    // scale + the buf are f32; the slot_map + the active_slots are u32 (the
    // index convention used throughout the attention-rs kernels).
    fn lora_shrink_impl(
        x: &Tensor,
        a: &Tensor,
        scale: &Tensor,
        slot_map: &Tensor,
        active_slots: &Tensor,
        buf: &Tensor,
    ) -> Result<()> {
        let dev = x.device();
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let rows = x.dims()[0];
        let in_features = x.dims()[1];
        let rank = a.dims()[2];
        let num_active = active_slots.dims()[0];

        let raw = |t: &Tensor, dt: &str| -> Result<*const std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *const std::ffi::c_void)
                }
                _ => bail!("lora_shrink: expected a CUDA tensor"),
            }
        };
        let raw_mut = |t: &Tensor, dt: &str| -> Result<*mut std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *mut std::ffi::c_void)
                }
                _ => bail!("lora_shrink: expected a CUDA tensor"),
            }
        };

        let x_tag = dtype_tag(x)?;
        let (x_ptr, a_ptr, scale_ptr, slot_ptr, active_ptr, buf_ptr) = (
            raw(x, x_tag)?,
            raw(a, x_tag)?,
            raw(scale, "f32")?,
            raw(slot_map, "u32")?,
            raw(active_slots, "u32")?,
            raw_mut(buf, "f32")?,
        );

        match x_tag {
            "f16" => unsafe {
                ffi::lora_shrink_f16(
                    x_ptr, a_ptr, scale_ptr, slot_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, num_active as i32, stream,
                );
            },
            "bf16" => unsafe {
                ffi::lora_shrink_bf16(
                    x_ptr, a_ptr, scale_ptr, slot_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, num_active as i32, stream,
                );
            },
            _ => unsafe {
                ffi::lora_shrink_f32(
                    x_ptr, a_ptr, scale_ptr, slot_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, num_active as i32, stream,
                );
            },
        }
        Ok(())
    }

    fn lora_expand_impl(
        buf: &Tensor,
        b: &Tensor,
        slot_map: &Tensor,
        active_slots: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::Storage;
        use kernels::ffi;

        let dev = out.device();
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let rows = out.dims()[0];
        let out_features = out.dims()[1];
        let rank = b.dims()[1];
        let num_active = active_slots.dims()[0];

        let raw = |t: &Tensor, dt: &str| -> Result<*const std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *const std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_expand: expected a CUDA tensor"),
            }
        };
        let raw_mut = |t: &Tensor, dt: &str| -> Result<*mut std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *mut std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_expand: expected a CUDA tensor"),
            }
        };

        let out_tag = dtype_tag(out)?;
        let (buf_ptr, b_ptr, slot_ptr, active_ptr, out_ptr) = (
            raw(buf, "f32")?,
            raw(b, out_tag)?,
            raw(slot_map, "u32")?,
            raw(active_slots, "u32")?,
            raw_mut(out, out_tag)?,
        );

        match out_tag {
            "f16" => unsafe {
                ffi::lora_expand_f16(
                    buf_ptr, b_ptr, slot_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, num_active as i32, stream,
                );
            },
            "bf16" => unsafe {
                ffi::lora_expand_bf16(
                    buf_ptr, b_ptr, slot_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, num_active as i32, stream,
                );
            },
            _ => unsafe {
                ffi::lora_expand_f32(
                    buf_ptr, b_ptr, slot_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, num_active as i32, stream,
                );
            },
        }
        Ok(())
    }

    // The dtype tag of a tensor (the "f16"/"bf16"/"f32"), for the launcher dispatch.
    fn dtype_tag(t: &Tensor) -> Result<&'static str> {
        match t.dtype() {
            DType::F16 => Ok("f16"),
            DType::BF16 => Ok("bf16"),
            DType::F32 => Ok("f32"),
            _ => Err(candle_core::Error::msg(format!(
                "lora_fused: unsupported dtype {:?}",
                t.dtype()
            ))),
        }
    }

    // The stacked shrink: the buf is [rows, max_stack, rank] f32; the CSR
    // slot_offsets/slot_ids carry the per-row adapter list.
    fn lora_shrink_stacked_impl(
        x: &Tensor,
        a: &Tensor,
        scale: &Tensor,
        slot_offsets: &Tensor,
        slot_ids: &Tensor,
        active_slots: &Tensor,
        buf: &Tensor,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::Storage;
        use kernels::ffi;

        let dev = x.device();
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let rows = x.dims()[0];
        let in_features = x.dims()[1];
        let rank = a.dims()[2];
        let max_stack = buf.dims()[1];

        let raw = |t: &Tensor, dt: &str| -> Result<*const std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *const std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_shrink_stacked: expected a CUDA tensor"),
            }
        };
        let raw_mut = |t: &Tensor, dt: &str| -> Result<*mut std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *mut std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_shrink_stacked: expected a CUDA tensor"),
            }
        };

        let x_tag = dtype_tag(x)?;
        let (x_ptr, a_ptr, scale_ptr, off_ptr, ids_ptr, active_ptr, buf_ptr) = (
            raw(x, x_tag)?,
            raw(a, x_tag)?,
            raw(scale, "f32")?,
            raw(slot_offsets, "u32")?,
            raw(slot_ids, "u32")?,
            raw(active_slots, "u32")?,
            raw_mut(buf, "f32")?,
        );

        match x_tag {
            "f16" => unsafe {
                ffi::lora_shrink_stacked_f16(
                    x_ptr, a_ptr, scale_ptr, off_ptr, ids_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, max_stack as i32, stream,
                );
            },
            "bf16" => unsafe {
                ffi::lora_shrink_stacked_bf16(
                    x_ptr, a_ptr, scale_ptr, off_ptr, ids_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, max_stack as i32, stream,
                );
            },
            _ => unsafe {
                ffi::lora_shrink_stacked_f32(
                    x_ptr, a_ptr, scale_ptr, off_ptr, ids_ptr, active_ptr, buf_ptr,
                    rows as i32, in_features as i32, rank as i32, max_stack as i32, stream,
                );
            },
        }
        Ok(())
    }

    // The stacked expand: the out is [rows, out_features]; the buf is
    // [rows, max_stack, rank] f32.
    fn lora_expand_stacked_impl(
        buf: &Tensor,
        b: &Tensor,
        slot_offsets: &Tensor,
        slot_ids: &Tensor,
        active_slots: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::Storage;
        use kernels::ffi;

        let dev = out.device();
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let rows = out.dims()[0];
        let out_features = out.dims()[1];
        let rank = b.dims()[1];
        let max_stack = buf.dims()[1];

        let raw = |t: &Tensor, dt: &str| -> Result<*const std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *const std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_expand_stacked: expected a CUDA tensor"),
            }
        };
        let raw_mut = |t: &Tensor, dt: &str| -> Result<*mut std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let ptr = match dt {
                        "f16" => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        "bf16" => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        "u32" => *c.as_cuda_slice::<u32>()?.device_ptr(),
                        _ => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    };
                    Ok(ptr as *mut std::ffi::c_void)
                }
                _ => candle_core::bail!("lora_expand_stacked: expected a CUDA tensor"),
            }
        };

        let out_tag = dtype_tag(out)?;
        let (buf_ptr, b_ptr, off_ptr, ids_ptr, active_ptr, out_ptr) = (
            raw(buf, "f32")?,
            raw(b, out_tag)?,
            raw(slot_offsets, "u32")?,
            raw(slot_ids, "u32")?,
            raw(active_slots, "u32")?,
            raw_mut(out, out_tag)?,
        );

        match out_tag {
            "f16" => unsafe {
                ffi::lora_expand_stacked_f16(
                    buf_ptr, b_ptr, off_ptr, ids_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, max_stack as i32, stream,
                );
            },
            "bf16" => unsafe {
                ffi::lora_expand_stacked_bf16(
                    buf_ptr, b_ptr, off_ptr, ids_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, max_stack as i32, stream,
                );
            },
            _ => unsafe {
                ffi::lora_expand_stacked_f32(
                    buf_ptr, b_ptr, off_ptr, ids_ptr, active_ptr, out_ptr,
                    rows as i32, rank as i32, out_features as i32, max_stack as i32, stream,
                );
            },
        }
        Ok(())
    }
}