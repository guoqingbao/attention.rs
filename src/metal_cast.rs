//! Fused dtype cast + layout transpose used by the Metal attention fast path.
//!
//! Projections produce `(seq, heads, dim)`; the attention kernel wants
//! `(heads, seq, dim)`. Doing this as `to_dtype` + `transpose` + `contiguous`
//! costs two full passes and a strided copy per tensor, which is a measurable
//! slice of long-prompt prefill. These kernels move each element exactly once.

#[cfg(feature = "metal")]
use candle_core::{Result, Tensor};

#[cfg(feature = "metal")]
fn type_tag(dtype: candle_core::DType) -> Option<&'static str> {
    use candle_core::DType;
    match dtype {
        DType::F32 => Some("float"),
        DType::F16 => Some("half"),
        DType::BF16 => Some("bfloat"),
        _ => None,
    }
}

/// `(seq, heads, dim)` → `(heads, seq, dim)` with a dtype cast.
/// Returns `None` when the fast kernel does not apply.
#[cfg(feature = "metal")]
pub fn seq_heads_to_heads_seq(
    x: &Tensor,
    seq: usize,
    heads: usize,
    dim: usize,
    out_dtype: candle_core::DType,
) -> Option<Result<Tensor>> {
    let launched = launch(
        x,
        seq,
        heads,
        dim,
        out_dtype,
        &format!(
            "cast_transpose_{}_to_{}_shd_hsd",
            type_tag(x.dtype())?,
            type_tag(out_dtype)?
        ),
        true,
    );
    launched
}

/// `(heads, seq, dim)` → `(seq, heads, dim)` with a dtype cast.
#[cfg(feature = "metal")]
pub fn heads_seq_to_seq_heads(
    x: &Tensor,
    seq: usize,
    heads: usize,
    dim: usize,
    out_dtype: candle_core::DType,
) -> Option<Result<Tensor>> {
    launch(
        x,
        seq,
        heads,
        dim,
        out_dtype,
        &format!(
            "cast_transpose_{}_to_{}_hsd_shd",
            type_tag(x.dtype())?,
            type_tag(out_dtype)?
        ),
        false,
    )
}

#[cfg(feature = "metal")]
fn launch(
    x: &Tensor,
    seq: usize,
    heads: usize,
    dim: usize,
    out_dtype: candle_core::DType,
    kernel_name: &str,
    shd_to_hsd: bool,
) -> Option<Result<Tensor>> {
    use candle_core::Device;
    use metal::MTLSize;

    if dim % 4 != 0 {
        return None;
    }
    let device = match x.device() {
        Device::Metal(dev) => dev,
        _ => return None,
    };
    let (x_storage, x_layout) = x.storage_and_layout();
    let x_ms = match &*x_storage {
        candle_core::Storage::Metal(s) => s,
        _ => return None,
    };
    if !x_layout.is_contiguous() {
        return None;
    }

    let out_shape = if shd_to_hsd {
        candle_core::Shape::from_dims(&[heads, seq, dim])
    } else {
        candle_core::Shape::from_dims(&[seq, heads, dim])
    };
    let out = match unsafe { Tensor::empty_(out_shape, out_dtype, x.device()) } {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };

    let kernels = crate::metal_kernels::Kernels::default();
    // Returns None on missing kernels (e.g. bf16 on older GPUs) so the caller
    // falls back to the generic transpose/cast path.
    let pipeline = kernels
        .load_pipeline(device.device(), kernel_name.to_string())
        .ok()?;

    let command_buffer = match device.command_buffer() {
        Ok(cb) => cb,
        Err(e) => return Some(Err(e)),
    };
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    {
        let (out_storage, _) = out.storage_and_layout();
        let out_ms = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => return None,
        };
        let x_offset = (x_layout.start_offset() * x.dtype().size_in_bytes()) as u64;
        encoder.set_buffer(0, Some(x_ms.buffer()), x_offset);
        encoder.set_buffer(1, Some(out_ms.buffer()), 0);
        for (i, v) in [seq as i32, heads as i32, dim as i32]
            .into_iter()
            .enumerate()
        {
            encoder.set_bytes(2 + i as u64, 4, &v.to_ne_bytes() as *const _ as *const _);
        }
        let total = (seq * heads * dim / 4) as u64;
        let threads = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total.div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(groups, threads);
    }
    encoder.end_encoding();
    Some(Ok(out))
}
