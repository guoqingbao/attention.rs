//! Moet-style sign-symmetric 2-bit MoE expert planes.
//!
//! Codebook `{−4,−1,+1,+4}` (codes 0..3). Packers match vLLM-Moet; device
//! unpack emits row-major e4m3 + F32 UE8M0 block-32 scales for `moe_gemm_fp8`.

use candle_core::{Device, Result, Tensor};

/// PRMT LUT word (LE bytes = e4m3 encodings of {-4,-1,+1,+4}).
pub const PRMT_LUT_WORD: u32 = 0x4838_B8C8;

pub const W2_SCALE_BLOCK_K: usize = 32;
pub const W2_ACTIVATION_GROUP_K: usize = 128;

/// DeepSeek V4 SwiGLU activation for the W2 down projection.  The kernel
/// matches the reference implementation's asymmetric gate clamp and BF16
/// rounding before the activation quantizer consumes the result.
#[cfg(feature = "cuda")]
pub fn moe_w2_swiglu_clamp_bf16(gate_up: &Tensor, hidden: usize, limit: f32) -> Result<Tensor> {
    use crate::kernels;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Device, Storage};

    let (rows, fused_hidden) = gate_up.dims2()?;
    if fused_hidden != hidden * 2 {
        candle_core::bail!(
            "moe_w2_swiglu_clamp_bf16: expected fused hidden {}, got {}",
            hidden * 2,
            fused_hidden
        );
    }
    let gate_up = gate_up.contiguous()?;
    let Device::Cuda(dev) = gate_up.device() else {
        candle_core::bail!("moe_w2_swiglu_clamp_bf16 requires CUDA");
    };
    if gate_up.dtype() != DType::BF16 {
        candle_core::bail!("moe_w2_swiglu_clamp_bf16 requires BF16 input");
    }
    let output = Tensor::zeros((rows, hidden), DType::BF16, gate_up.device())?;
    let (in_storage, in_layout) = gate_up.storage_and_layout();
    let (out_storage, out_layout) = output.storage_and_layout();
    let in_ptr = match &*in_storage {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<half::bf16>()?
            .slice(in_layout.start_offset()..)
            .device_ptr() as *const std::ffi::c_void,
        _ => candle_core::bail!("gate_up must be CUDA"),
    };
    let out_ptr = match &*out_storage {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..)
            .device_ptr() as *mut std::ffi::c_void,
        _ => candle_core::bail!("output must be CUDA"),
    };
    unsafe {
        kernels::ffi::moe_w2_swiglu_clamp_bf16(
            in_ptr,
            out_ptr,
            rows as i32,
            hidden as i32,
            limit,
            *dev.cu_stream() as i64,
        );
    }
    drop((in_storage, out_storage));
    Ok(output)
}

#[cfg(not(feature = "cuda"))]
pub fn moe_w2_swiglu_clamp_bf16(gate_up: &Tensor, hidden: usize, limit: f32) -> Result<Tensor> {
    let _ = (gate_up, hidden, limit);
    candle_core::bail!("moe_w2_swiglu_clamp_bf16 requires cuda feature")
}

/// e4m3 byte encodings for codebook levels (matches PRMT_LUT_WORD).
fn code_to_e4m3(code: u8) -> u8 {
    ((PRMT_LUT_WORD >> (8 * (code as u32))) & 0xFF) as u8
}

fn scale_from_ue8m0(s: u8) -> f32 {
    if s == 0 {
        return 0.0;
    }
    2.0f32.powi(s as i32 - 127)
}

/// Decode row-major plane → e4m3 bytes `[N*K]` (host fallback for non-CUDA).
fn unpack_row_major_to_e4m3(plane: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    if plane.len() < n * k / 4 {
        candle_core::bail!("unpack_row_major_to_e4m3: plane too small");
    }
    let mut out = vec![0u8; n * k];
    for i in 0..(n * k / 4) {
        let b = plane[i];
        out[4 * i] = code_to_e4m3(b & 3);
        out[4 * i + 1] = code_to_e4m3((b >> 2) & 3);
        out[4 * i + 2] = code_to_e4m3((b >> 4) & 3);
        out[4 * i + 3] = code_to_e4m3((b >> 6) & 3);
    }
    Ok(out)
}

/// UE8M0 scales `[N*(K/32)]` → F32 block scales (host fallback).
fn ue8m0_to_f32_scales(scales: &[u8]) -> Vec<f32> {
    scales.iter().map(|&s| scale_from_ue8m0(s)).collect()
}

/// GPU: unpack row-major W2 planes `[E, N*K/4]` + UE8M0 `[E, N*(K/32)]`
/// → e4m3 U8 `[E, N, K]` + F32 scales `[E, N, K/32]`.
pub fn moe_w2_unpack_to_fp8(
    planes: &Tensor,
    scales_ue8m0: &Tensor,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let device = planes.device();
    let e = planes.dim(0)?;
    if k % 4 != 0 || k % W2_SCALE_BLOCK_K != 0 {
        candle_core::bail!("moe_w2_unpack_to_fp8: invalid K={k}");
    }

    #[cfg(feature = "cuda")]
    {
        if matches!(device, Device::Cuda(_)) {
            return moe_w2_unpack_to_fp8_cuda(planes, scales_ue8m0, e, n, k);
        }
    }

    // Host fallback (CPU / Metal path for bring-up)
    let plane_flat = planes.flatten_all()?.to_vec1::<u8>()?;
    let scale_flat = scales_ue8m0.flatten_all()?.to_vec1::<u8>()?;
    let plane_stride = n * k / 4;
    let scale_stride = n * (k / W2_SCALE_BLOCK_K);
    let mut w_out = Vec::with_capacity(e * n * k);
    let mut s_out = Vec::with_capacity(e * scale_stride);
    for ei in 0..e {
        let p = &plane_flat[ei * plane_stride..(ei + 1) * plane_stride];
        w_out.extend(unpack_row_major_to_e4m3(p, n, k)?);
        let s = &scale_flat[ei * scale_stride..(ei + 1) * scale_stride];
        s_out.extend(ue8m0_to_f32_scales(s));
    }
    let w = Tensor::from_vec(w_out, (e, n, k), device)?;
    let s = Tensor::from_vec(s_out, (e, n, k / W2_SCALE_BLOCK_K), device)?;
    Ok((w, s))
}

#[cfg(feature = "cuda")]
fn moe_w2_unpack_to_fp8_cuda(
    planes: &Tensor,
    scales_ue8m0: &Tensor,
    e: usize,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use crate::kernels;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Storage};

    let device = planes.device();
    let out_w = Tensor::zeros((e, n, k), DType::U8, device)?;
    let out_s = Tensor::zeros((e, n, k / W2_SCALE_BLOCK_K), DType::F32, device)?;

    let stream = match device {
        Device::Cuda(d) => *d.cu_stream() as i64,
        _ => candle_core::bail!("moe_w2_unpack_to_fp8_cuda requires CUDA"),
    };

    let (planes_s, planes_l) = planes.storage_and_layout();
    let (scales_s, scales_l) = scales_ue8m0.storage_and_layout();
    let (out_w_s, out_w_l) = out_w.storage_and_layout();
    let (out_s_s, out_s_l) = out_s.storage_and_layout();

    let planes_ptr = match &*planes_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(planes_l.start_offset()..)
            .device_ptr() as *const u8,
        _ => candle_core::bail!("planes must be CUDA"),
    };
    let scales_ptr = match &*scales_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(scales_l.start_offset()..)
            .device_ptr() as *const u8,
        _ => candle_core::bail!("scales must be CUDA"),
    };
    let out_w_ptr = match &*out_w_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(out_w_l.start_offset()..)
            .device_ptr() as *mut u8,
        _ => candle_core::bail!("out_w must be CUDA"),
    };
    let out_s_ptr = match &*out_s_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<f32>()?
            .slice(out_s_l.start_offset()..)
            .device_ptr() as *mut f32,
        _ => candle_core::bail!("out_s must be CUDA"),
    };

    unsafe {
        kernels::ffi::moe_w2_unpack_to_fp8(
            planes_ptr, scales_ptr, out_w_ptr, out_s_ptr, e as i32, n as i32, k as i32, stream,
        );
    }
    drop((planes_s, scales_s, out_w_s, out_s_s));
    Ok((out_w, out_s))
}

/// Unpack only the experts listed in `expert_ids` [U] (I32) → `[U, N, K]` e4m3 + F32 scales.
#[cfg(feature = "cuda")]
pub fn moe_w2_unpack_by_ids_to_fp8(
    planes: &Tensor,
    scales_ue8m0: &Tensor,
    expert_ids: &Tensor,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use crate::kernels;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Storage};

    let device = planes.device();
    let u = expert_ids.elem_count();
    if k % 4 != 0 || k % W2_SCALE_BLOCK_K != 0 {
        candle_core::bail!("moe_w2_unpack_by_ids_to_fp8: invalid K={k}");
    }
    if u == 0 {
        candle_core::bail!("moe_w2_unpack_by_ids_to_fp8: empty expert_ids");
    }

    let Device::Cuda(d) = device else {
        candle_core::bail!("moe_w2_unpack_by_ids_to_fp8 requires CUDA");
    };
    let stream = *d.cu_stream() as i64;

    let expert_ids = if expert_ids.dtype() == DType::U32 && expert_ids.is_contiguous() {
        expert_ids.clone()
    } else {
        expert_ids.to_dtype(DType::U32)?.contiguous()?
    };

    let out_w = Tensor::zeros((u, n, k), DType::U8, device)?;
    let out_s = Tensor::zeros((u, n, k / W2_SCALE_BLOCK_K), DType::F32, device)?;

    let (planes_s, planes_l) = planes.storage_and_layout();
    let (scales_s, scales_l) = scales_ue8m0.storage_and_layout();
    let (ids_s, ids_l) = expert_ids.storage_and_layout();
    let (out_w_s, out_w_l) = out_w.storage_and_layout();
    let (out_s_s, out_s_l) = out_s.storage_and_layout();

    let planes_ptr = match &*planes_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(planes_l.start_offset()..)
            .device_ptr() as *const u8,
        _ => candle_core::bail!("planes must be CUDA"),
    };
    let scales_ptr = match &*scales_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(scales_l.start_offset()..)
            .device_ptr() as *const u8,
        _ => candle_core::bail!("scales must be CUDA"),
    };
    // Kernel takes int*; U32 expert ids are non-negative so bit-identical reinterpret is safe.
    let ids_ptr = match &*ids_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u32>()?
            .slice(ids_l.start_offset()..)
            .device_ptr() as *const i32,
        _ => candle_core::bail!("expert_ids must be CUDA"),
    };
    let out_w_ptr = match &*out_w_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<u8>()?
            .slice(out_w_l.start_offset()..)
            .device_ptr() as *mut u8,
        _ => candle_core::bail!("out_w must be CUDA"),
    };
    let out_s_ptr = match &*out_s_s {
        Storage::Cuda(c) => *c
            .as_cuda_slice::<f32>()?
            .slice(out_s_l.start_offset()..)
            .device_ptr() as *mut f32,
        _ => candle_core::bail!("out_s must be CUDA"),
    };

    unsafe {
        kernels::ffi::moe_w2_unpack_by_ids_to_fp8(
            planes_ptr, scales_ptr, ids_ptr, out_w_ptr, out_s_ptr, u as i32, n as i32, k as i32,
            stream,
        );
    }
    drop((planes_s, scales_s, ids_s, out_w_s, out_s_s));
    Ok((out_w, out_s))
}

/// Apply the W2 reference activation quantization numerics and return the
/// BF16 dequantized activations consumed by the generic grouped GEMM.  W2 is
/// trained as W2A8 with per-token group-128 FP8 activations; feeding the raw
/// BF16 tensor into the fallback GEMM changes the model's QAT distribution.
#[cfg(feature = "cuda")]
pub fn moe_w2_quantize_dequantize_activation(input: &Tensor) -> Result<Tensor> {
    use crate::kernels;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Device, Storage};

    let (rows, k) = input.dims2()?;
    if k % W2_ACTIVATION_GROUP_K != 0 {
        candle_core::bail!(
            "moe_w2 activation K={} must be divisible by {}",
            k,
            W2_ACTIVATION_GROUP_K
        );
    }
    let input = input.contiguous()?;
    let Device::Cuda(dev) = input.device() else {
        candle_core::bail!("moe_w2 activation quantization requires CUDA");
    };
    if input.dtype() != DType::BF16 {
        candle_core::bail!("moe_w2 activation quantization requires BF16 input");
    }
    let q = Tensor::zeros((rows, k), DType::U8, input.device())?;
    let scales = Tensor::zeros(
        (rows, k / W2_ACTIVATION_GROUP_K),
        DType::F32,
        input.device(),
    )?;
    let out = Tensor::zeros((rows, k), DType::BF16, input.device())?;
    let stream = *dev.cu_stream() as i64;

    let ptrs = |tensor: &Tensor| -> Result<(*const u8, usize)> {
        let (storage, layout) = tensor.storage_and_layout();
        let ptr = match &*storage {
            Storage::Cuda(c) => *c
                .as_cuda_slice::<u8>()?
                .slice(layout.start_offset()..)
                .device_ptr(),
            _ => candle_core::bail!("expected CUDA tensor"),
        };
        Ok((ptr as *const u8, layout.start_offset()))
    };
    let input_ptr = {
        let (storage, layout) = input.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => *c
                .as_cuda_slice::<half::bf16>()?
                .slice(layout.start_offset()..)
                .device_ptr() as *const std::ffi::c_void,
            _ => candle_core::bail!("expected CUDA input"),
        }
    };
    let q_ptr = ptrs(&q)?.0 as *mut std::ffi::c_void;
    let scale_ptr = {
        let (storage, layout) = scales.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => *c
                .as_cuda_slice::<f32>()?
                .slice(layout.start_offset()..)
                .device_ptr() as *mut f32,
            _ => candle_core::bail!("expected CUDA scales"),
        }
    };
    let out_ptr = {
        let (storage, layout) = out.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => *c
                .as_cuda_slice::<half::bf16>()?
                .slice(layout.start_offset()..)
                .device_ptr() as *mut std::ffi::c_void,
            _ => candle_core::bail!("expected CUDA output"),
        }
    };
    unsafe {
        kernels::ffi::fp8_quantize_per_token_group_launch(
            input_ptr,
            q_ptr,
            scale_ptr,
            (rows * (k / W2_ACTIVATION_GROUP_K)) as i32,
            W2_ACTIVATION_GROUP_K as i32,
            (k / W2_ACTIVATION_GROUP_K) as i32,
            (k / W2_ACTIVATION_GROUP_K) as i32,
            false,
            false,
            stream,
        );
        kernels::ffi::moe_w2_dequantize_activation_fp8(
            q_ptr as *const u8,
            scale_ptr as *const f32,
            out_ptr,
            rows as i32,
            k as i32,
            stream,
        );
    }
    Ok(out)
}

/// GPU: pack MXFP4/e2m1 `[E,N,K/2]` + UE8M0 scales → W2 planes `[E,N*K/4]` + scales.
/// Entirely device-side (no host D2H). Scales are copied byte-identical.
#[cfg(feature = "cuda")]
pub fn moe_w2_pack_from_mxfp4(
    packed: &Tensor,
    scales: &Tensor,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use crate::kernels;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Storage};

    let device = packed.device();
    let (e, n0, k_half) = packed.dims3()?;
    if n0 != n || k_half * 2 != k {
        candle_core::bail!(
            "moe_w2_pack_from_mxfp4: expected packed [E,{n},{}], got {:?}",
            k / 2,
            packed.dims()
        );
    }
    if k % 4 != 0 || k % W2_SCALE_BLOCK_K != 0 {
        candle_core::bail!("moe_w2_pack_from_mxfp4: invalid K={k}");
    }
    let Device::Cuda(d) = device else {
        candle_core::bail!("moe_w2_pack_from_mxfp4 requires CUDA");
    };
    let stream = *d.cu_stream() as i64;

    let plane_stride = n * k / 4;
    let scale_stride = n * (k / W2_SCALE_BLOCK_K);
    let planes = Tensor::zeros((e, plane_stride), DType::U8, device)?;
    let scales_out = Tensor::zeros((e, scale_stride), DType::U8, device)?;

    let packed = packed.contiguous()?;
    let scales = scales.contiguous()?;

    let (ps, pl) = packed.storage_and_layout();
    let (ss, sl) = scales.storage_and_layout();
    let (os, ol) = planes.storage_and_layout();
    let (sos, sol) = scales_out.storage_and_layout();

    let as_u8_ptr = |s: &Storage, start: usize, dtype: DType| -> Result<*const u8> {
        match (s, dtype) {
            (Storage::Cuda(c), DType::U8 | DType::F8E8M0 | DType::F8E4M3) => {
                Ok(*c.as_cuda_slice::<u8>()?.slice(start..).device_ptr() as *const u8)
            }
            _ => candle_core::bail!("moe_w2_pack: expected CUDA 1-byte tensor, got {dtype:?}"),
        }
    };

    let packed_ptr = as_u8_ptr(&ps, pl.start_offset(), packed.dtype())?;
    let scales_ptr = as_u8_ptr(&ss, sl.start_offset(), scales.dtype())?;
    let planes_ptr = as_u8_ptr(&os, ol.start_offset(), DType::U8)? as *mut u8;
    let scales_out_ptr = as_u8_ptr(&sos, sol.start_offset(), DType::U8)? as *mut u8;

    unsafe {
        kernels::ffi::moe_w2_pack_from_mxfp4(
            packed_ptr,
            scales_ptr,
            planes_ptr,
            scales_out_ptr,
            e as i32,
            n as i32,
            k as i32,
            stream,
        );
    }
    drop((ps, ss, os, sos));
    Ok((planes, scales_out))
}

#[cfg(not(feature = "cuda"))]
pub fn moe_w2_pack_from_mxfp4(
    packed: &Tensor,
    scales: &Tensor,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let _ = (packed, scales, n, k);
    candle_core::bail!("moe_w2_pack_from_mxfp4 requires cuda feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_host_unpack() {
        let n = 16;
        let k = 64;
        // Row-major packing: 4 codes per byte, little-endian nibbles of 2 bits.
        let mut plane = vec![0u8; n * k / 4];
        for i in 0..(n * k / 4) {
            let b0 = ((4 * i) % 4) as u8;
            let b1 = ((4 * i + 1) % 4) as u8;
            let b2 = ((4 * i + 2) % 4) as u8;
            let b3 = ((4 * i + 3) % 4) as u8;
            plane[i] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
        }
        let e4 = unpack_row_major_to_e4m3(&plane, n, k).unwrap();
        assert_eq!(e4.len(), n * k);
        assert_eq!(e4[0], code_to_e4m3(0));
        assert_eq!(e4[1], code_to_e4m3(1));
        assert_eq!(e4[2], code_to_e4m3(2));
        assert_eq!(e4[3], code_to_e4m3(3));
    }

    #[test]
    fn prmt_lut_codebook_bytes() {
        // {-4,-1,+1,+4} as e4m3 bytes in PRMT_LUT_WORD little-endian order.
        assert_eq!(code_to_e4m3(0), 0xC8);
        assert_eq!(code_to_e4m3(1), 0xB8);
        assert_eq!(code_to_e4m3(2), 0x38);
        assert_eq!(code_to_e4m3(3), 0x48);
    }
}
