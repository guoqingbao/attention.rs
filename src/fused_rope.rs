//! Fused Rotary Position Embedding (RoPE) CUDA kernel interface
//!
//! This module provides a high-performance fused rotary embedding implementation
//! that fuses two operations:
//!   1. Position-based cos/sin selection (eliminates index_select kernel)
//!   2. Rotary position embedding application
//!
//! Supports Grouped Query Attention (GQA) where Q and K have different head counts.

use candle_core::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use kernels::ffi;

#[derive(Clone, Copy)]
enum RopeLayout {
    BatchMajor {
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
    },
    TokenMajor {
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
    },
}

#[allow(dead_code)]
impl RopeLayout {
    fn positions_len(self) -> usize {
        match self {
            Self::BatchMajor { seq_len, .. } => seq_len as usize,
            Self::TokenMajor { num_tokens, .. } => num_tokens as usize,
        }
    }

    fn q_bh(self) -> u32 {
        match self {
            Self::BatchMajor { q_bh, .. } => q_bh,
            Self::TokenMajor { q_heads, .. } => q_heads,
        }
    }

    fn k_bh(self) -> u32 {
        match self {
            Self::BatchMajor { k_bh, .. } => k_bh,
            Self::TokenMajor { k_heads, .. } => k_heads,
        }
    }

    fn seq_len(self) -> u32 {
        match self {
            Self::BatchMajor { seq_len, .. } => seq_len,
            Self::TokenMajor { num_tokens, .. } => num_tokens,
        }
    }

    fn d(self) -> u32 {
        match self {
            Self::BatchMajor { d, .. } => d,
            Self::TokenMajor { d, .. } => d,
        }
    }

    fn is_token_major(self) -> bool {
        matches!(self, Self::TokenMajor { .. })
    }
}

fn resolve_rope_layout(q: &Tensor, k: &Tensor) -> Result<RopeLayout> {
    match (q.dims().len(), k.dims().len()) {
        (4, 4) => {
            let (b, q_h, seq_len, d) = q.dims4()?;
            let (kb, k_h, k_seq_len, kd) = k.dims4()?;
            if b != kb || seq_len != k_seq_len || d != kd {
                candle_core::bail!(
                    "Q and K batch/seq_len/head_dim must match, got Q: {:?}, K: {:?}",
                    q.shape(),
                    k.shape()
                );
            }
            Ok(RopeLayout::BatchMajor {
                q_bh: (b * q_h) as u32,
                k_bh: (b * k_h) as u32,
                seq_len: seq_len as u32,
                d: d as u32,
            })
        }
        (3, 3) => {
            let (num_tokens, q_heads, d) = q.dims3()?;
            let (k_num_tokens, k_heads, kd) = k.dims3()?;
            if num_tokens != k_num_tokens || d != kd {
                candle_core::bail!(
                    "Q and K num_tokens/head_dim must match, got Q: {:?}, K: {:?}",
                    q.shape(),
                    k.shape()
                );
            }
            Ok(RopeLayout::TokenMajor {
                num_tokens: num_tokens as u32,
                q_heads: q_heads as u32,
                k_heads: k_heads as u32,
                d: d as u32,
            })
        }
        _ => candle_core::bail!(
            "FusedRope expects Q and K to be both 4D [batch, heads, seq, dim] or both 3D [tokens, heads, dim], got Q: {:?}, K: {:?}",
            q.shape(),
            k.shape()
        ),
    }
}

#[cfg(not(feature = "cuda"))]
fn launch_fused_rope_metal(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_interleaved: bool,
    rotary_dim: usize,
) -> Result<()> {
    use candle_core::backend::BackendStorage;

    let layout = resolve_rope_layout(q, k)?;
    let expected_positions_len = layout.positions_len();
    let pos_shape = positions.dims();
    if pos_shape.len() != 1 || pos_shape[0] != expected_positions_len {
        candle_core::bail!(
            "positions should be [{}], got {:?}",
            expected_positions_len,
            pos_shape
        );
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "rotary_dim must be an even positive integer, got {}",
            rotary_dim
        );
    }
    if rotary_dim > layout.d() as usize {
        candle_core::bail!(
            "rotary_dim {} exceeds head_dim {} for Q {:?}, K {:?}",
            rotary_dim,
            layout.d(),
            q.shape(),
            k.shape()
        );
    }

    let positions = if positions.dtype() != DType::I64 {
        positions.to_dtype(DType::I64)?
    } else {
        positions.clone()
    };

    if !q.is_contiguous()
        || !k.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || !positions.is_contiguous()
    {
        candle_core::bail!("All tensors (q, k, cos, sin, positions) must be contiguous");
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
        candle_core::bail!(
            "Q, K, cos, sin must have same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
            q.dtype(),
            k.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let (pos_storage, pos_layout) = positions.storage_and_layout();

    let q_metal = match &*q_storage {
        candle_core::Storage::Metal(s) => s,
        _ => candle_core::bail!("Q must be on Metal device"),
    };
    let k_metal = match &*k_storage {
        candle_core::Storage::Metal(s) => s,
        _ => candle_core::bail!("K must be on Metal device"),
    };
    let cos_metal = match &*cos_storage {
        candle_core::Storage::Metal(s) => s,
        _ => candle_core::bail!("cos must be on Metal device"),
    };
    let sin_metal = match &*sin_storage {
        candle_core::Storage::Metal(s) => s,
        _ => candle_core::bail!("sin must be on Metal device"),
    };
    let pos_metal = match &*pos_storage {
        candle_core::Storage::Metal(s) => s,
        _ => candle_core::bail!("positions must be on Metal device"),
    };

    let device = q_metal.device();
    let command_buffer = device.command_buffer()?;
    let kernels = metal_kernels::Kernels::default();

    metal_kernels::call_fused_rope(
        device.device(),
        &*command_buffer,
        kernels,
        dtype,
        q_metal.buffer(),
        q_layout.start_offset() * dtype.size_in_bytes(),
        k_metal.buffer(),
        k_layout.start_offset() * dtype.size_in_bytes(),
        cos_metal.buffer(),
        cos_layout.start_offset() * dtype.size_in_bytes(),
        sin_metal.buffer(),
        sin_layout.start_offset() * dtype.size_in_bytes(),
        pos_metal.buffer(),
        pos_layout.start_offset() * std::mem::size_of::<i64>(),
        layout.q_bh(),
        layout.k_bh(),
        layout.seq_len(),
        layout.d(),
        rotary_dim as u32,
        is_interleaved,
        layout.is_token_major(),
    )
    .map_err(|e| candle_core::Error::Msg(format!("Metal fused_rope error: {:?}", e)))?;

    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_fused_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_interleaved: bool,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    let layout = resolve_rope_layout(q, k)?;
    let expected_positions_len = layout.positions_len();
    let pos_shape = positions.dims();
    if pos_shape.len() != 1 || pos_shape[0] != expected_positions_len {
        candle_core::bail!(
            "positions should be [{}], got {:?}",
            expected_positions_len,
            pos_shape
        );
    }

    let positions = if positions.dtype() != DType::I64 {
        positions.to_dtype(DType::I64)?
    } else {
        positions.clone()
    };

    if !q.is_contiguous()
        || !k.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || !positions.is_contiguous()
    {
        candle_core::bail!("All tensors (q, k, cos, sin, positions) must be contiguous");
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
        candle_core::bail!(
            "Q, K, cos, sin must have same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
            q.dtype(),
            k.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let dev = q.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let q_storage = q.storage_and_layout().0;
    let k_storage = k.storage_and_layout().0;
    let cos_storage = cos.storage_and_layout().0;
    let sin_storage = sin.storage_and_layout().0;
    let pos_storage = positions.storage_and_layout().0;

    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("Q must be on CUDA"),
    };
    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("K must be on CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cos must be on CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("sin must be on CUDA"),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("positions must be on CUDA"),
    };

    let pos_ptr = match &pos_cuda.slice {
        CudaStorageSlice::I64(s) => *s.device_ptr() as *const i64,
        _ => candle_core::bail!("positions must be I64"),
    };

    match dtype {
        DType::F32 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                _ => candle_core::bail!("Expected F32"),
            };

            unsafe {
                match layout {
                    RopeLayout::BatchMajor {
                        q_bh,
                        k_bh,
                        seq_len,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_f32(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        } else {
                            ffi::fused_rope_f32(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        }
                    }
                    RopeLayout::TokenMajor {
                        num_tokens,
                        q_heads,
                        k_heads,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_tok_major_f32(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        } else {
                            ffi::fused_rope_tok_major_f32(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        }
                    }
                }
            }
        }
        DType::F16 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };

            unsafe {
                match layout {
                    RopeLayout::BatchMajor {
                        q_bh,
                        k_bh,
                        seq_len,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_f16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        } else {
                            ffi::fused_rope_f16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        }
                    }
                    RopeLayout::TokenMajor {
                        num_tokens,
                        q_heads,
                        k_heads,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_tok_major_f16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        } else {
                            ffi::fused_rope_tok_major_f16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        }
                    }
                }
            }
        }
        DType::BF16 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };

            unsafe {
                match layout {
                    RopeLayout::BatchMajor {
                        q_bh,
                        k_bh,
                        seq_len,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_bf16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        } else {
                            ffi::fused_rope_bf16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d,
                                stream,
                            );
                        }
                    }
                    RopeLayout::TokenMajor {
                        num_tokens,
                        q_heads,
                        k_heads,
                        d,
                    } => {
                        if is_interleaved {
                            ffi::fused_rope_i_tok_major_bf16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        } else {
                            ffi::fused_rope_tok_major_bf16(
                                q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads,
                                k_heads, d, stream,
                            );
                        }
                    }
                }
            }
        }
        _ => candle_core::bail!("FusedRope only supports F32, F16, BF16, got {:?}", dtype),
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_fused_rope_partial_token_major(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_interleaved: bool,
    rotary_dim: usize,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    let (num_tokens, q_heads, full_d) = q.dims3()?;
    let (k_num_tokens, k_heads, k_d) = k.dims3()?;
    if num_tokens != k_num_tokens || full_d != k_d {
        candle_core::bail!(
            "Q and K num_tokens/head_dim must match, got Q: {:?}, K: {:?}",
            q.shape(),
            k.shape()
        );
    }
    if rotary_dim == 0 || rotary_dim > full_d || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "partial fused rope requires even rotary_dim in 1..={}, got {}",
            full_d,
            rotary_dim
        );
    }
    if positions.dims() != [num_tokens] {
        candle_core::bail!(
            "positions should be [{}], got {:?}",
            num_tokens,
            positions.dims()
        );
    }
    if cos.dims().len() != 2 || sin.dims().len() != 2 {
        candle_core::bail!(
            "cos/sin should be 2D full tables, got cos {:?}, sin {:?}",
            cos.shape(),
            sin.shape()
        );
    }

    let positions = if positions.dtype() != DType::I64 {
        positions.to_dtype(DType::I64)?
    } else {
        positions.clone()
    };

    if !q.is_contiguous()
        || !k.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || !positions.is_contiguous()
    {
        candle_core::bail!("All tensors (q, k, cos, sin, positions) must be contiguous");
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || cos.dtype() != dtype || sin.dtype() != dtype {
        candle_core::bail!(
            "Q, K, cos, sin must have same dtype, got Q: {:?}, K: {:?}, cos: {:?}, sin: {:?}",
            q.dtype(),
            k.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let dev = q.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let q_storage = q.storage_and_layout().0;
    let k_storage = k.storage_and_layout().0;
    let cos_storage = cos.storage_and_layout().0;
    let sin_storage = sin.storage_and_layout().0;
    let pos_storage = positions.storage_and_layout().0;

    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("Q must be on CUDA"),
    };
    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("K must be on CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cos must be on CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("sin must be on CUDA"),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("positions must be on CUDA"),
    };

    let pos_ptr = match &pos_cuda.slice {
        CudaStorageSlice::I64(s) => *s.device_ptr() as *const i64,
        _ => candle_core::bail!("positions must be I64"),
    };

    match dtype {
        DType::F32 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *mut f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                _ => candle_core::bail!("Expected F32"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::F32(s) => *s.device_ptr() as *const f32,
                _ => candle_core::bail!("Expected F32"),
            };
            unsafe {
                if is_interleaved {
                    ffi::fused_rope_i_partial_tok_major_f32(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                } else {
                    ffi::fused_rope_partial_tok_major_f32(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                }
            }
        }
        DType::F16 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::F16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected F16"),
            };
            unsafe {
                if is_interleaved {
                    ffi::fused_rope_i_partial_tok_major_f16(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                } else {
                    ffi::fused_rope_partial_tok_major_f16(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                }
            }
        }
        DType::BF16 => {
            let q_ptr = match &q_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let k_ptr = match &k_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let cos_ptr = match &cos_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            let sin_ptr = match &sin_cuda.slice {
                CudaStorageSlice::BF16(s) => *s.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("Expected BF16"),
            };
            unsafe {
                if is_interleaved {
                    ffi::fused_rope_i_partial_tok_major_bf16(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                } else {
                    ffi::fused_rope_partial_tok_major_bf16(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        pos_ptr,
                        num_tokens as u32,
                        q_heads as u32,
                        k_heads as u32,
                        rotary_dim as u32,
                        full_d as u32,
                        stream,
                    );
                }
            }
        }
        _ => candle_core::bail!("FusedRope only supports F32, F16, BF16, got {:?}", dtype),
    }

    Ok(())
}

/// Fused Rotary Position Embedding
///
/// Applies rotary position embedding to Q and K tensors using optimized CUDA kernels.
/// Fuses the position-based cos/sin selection with the RoPE computation.
pub struct FusedRope;

impl FusedRope {
    /// Apply fused rotary embedding with position-based cos/sin selection.
    ///
    /// This fuses index_select + RoPE into a single kernel, eliminating one kernel launch.
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape [batch, num_q_heads, seq_len, head_dim]
    ///   or packed [num_tokens, num_q_heads, head_dim]
    /// * `k` - Key tensor, shape [batch, num_kv_heads, seq_len, head_dim]
    ///   or packed [num_tokens, num_kv_heads, head_dim]
    /// * `cos` - FULL cosine table, shape [max_seq_len, head_dim/2]
    /// * `sin` - FULL sine table, shape [max_seq_len, head_dim/2]
    /// * `positions` - Position indices, shape [seq_len] for 4D inputs or
    ///   [num_tokens] for packed token-major inputs
    /// * `is_interleaved` - If true, uses interleaved layout (adjacent pairs)
    ///
    /// # Returns
    /// Result with (q_embed, k_embed) tensors with rotary embedding applied
    #[cfg(feature = "cuda")]
    pub fn apply(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<(Tensor, Tensor)> {
        launch_fused_rope(q, k, cos, sin, positions, is_interleaved)?;
        Ok((q.to_owned(), k.to_owned()))
    }

    /// Apply fused rotary embedding in-place.
    ///
    /// Same as `apply` but modifies Q and K directly.
    #[cfg(feature = "cuda")]
    pub fn apply_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<()> {
        launch_fused_rope(q, k, cos, sin, positions, is_interleaved)
    }

    /// Apply fused rotary embedding in-place to only the leading `rotary_dim`
    /// channels of packed token-major Q/K tensors.
    #[cfg(feature = "cuda")]
    pub fn apply_inplace_partial(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
        rotary_dim: usize,
    ) -> Result<()> {
        launch_fused_rope_partial_token_major(q, k, cos, sin, positions, is_interleaved, rotary_dim)
    }

    /// Convenience: non-interleaved RoPE
    #[cfg(feature = "cuda")]
    pub fn apply_rope(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, true)
    }

    /// Convenience: non-interleaved RoPE in-place
    #[cfg(feature = "cuda")]
    pub fn apply_rope_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE in-place
    #[cfg(feature = "cuda")]
    pub fn apply_rope_i_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, true)
    }

    // ========================================================================
    // Metal implementations
    // ========================================================================

    /// Apply fused rotary embedding in-place (Metal version)
    #[cfg(not(feature = "cuda"))]
    #[allow(unused)]
    pub fn apply_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<()> {
        let layout = resolve_rope_layout(q, k)?;
        launch_fused_rope_metal(
            q,
            k,
            cos,
            sin,
            positions,
            is_interleaved,
            layout.d() as usize,
        )
    }

    /// Apply fused rotary embedding in-place to only the leading `rotary_dim`
    /// channels of Q/K tensors on non-CUDA backends.
    #[cfg(not(feature = "cuda"))]
    pub fn apply_inplace_partial(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
        rotary_dim: usize,
    ) -> Result<()> {
        launch_fused_rope_metal(q, k, cos, sin, positions, is_interleaved, rotary_dim)
    }

    /// Apply fused rotary embedding (Metal version) - returns new tensors
    #[cfg(not(feature = "cuda"))]
    pub fn apply(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        is_interleaved: bool,
    ) -> Result<(Tensor, Tensor)> {
        // Clone tensors (Metal will modify in-place)
        let q_out = q.contiguous()?.clone();
        let k_out = k.contiguous()?.clone();
        Self::apply_inplace(&q_out, &k_out, cos, sin, positions, is_interleaved)?;
        Ok((q_out, k_out))
    }

    /// Convenience: non-interleaved RoPE (Metal)
    #[cfg(not(feature = "cuda"))]
    pub fn apply_rope(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE (Metal)
    #[cfg(not(feature = "cuda"))]
    pub fn apply_rope_i(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Self::apply(q, k, cos, sin, positions, true)
    }

    /// Convenience: non-interleaved RoPE in-place (Metal)
    #[cfg(not(feature = "cuda"))]
    pub fn apply_rope_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, false)
    }

    /// Convenience: interleaved RoPE in-place (Metal)
    #[cfg(not(feature = "cuda"))]
    pub fn apply_rope_i_inplace(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        Self::apply_inplace(q, k, cos, sin, positions, true)
    }
}
