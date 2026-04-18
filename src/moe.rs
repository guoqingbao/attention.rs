#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::get_moe_cutlass_workspace;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::quantized::QTensor;
use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RoutedRowsMetadata {
    sorted_token_ids: Vec<u32>,
    scatter_ids: Vec<u32>,
    expert_offsets: Vec<u32>,
    expert_counts: Vec<u32>,
    sf_offsets: Vec<u32>,
    problem_sizes: Vec<u32>,
    total_sf_rows: usize,
    total_expanded: usize,
}

#[cfg(test)]
fn build_routed_rows_metadata(
    indices_cpu: &[Vec<u32>],
    num_experts: usize,
    n: usize,
    k: usize,
    input_has_topk_dim: bool,
) -> Result<RoutedRowsMetadata> {
    let num_tokens = indices_cpu.len();
    let topk = indices_cpu.first().map_or(0, Vec::len);
    for (token_idx, row) in indices_cpu.iter().enumerate() {
        if row.len() != topk {
            candle_core::bail!(
                "moe_gemm_nvfp4: inconsistent topk width at token {}: expected {}, got {}",
                token_idx,
                topk,
                row.len()
            );
        }
    }

    let total_expanded = num_tokens * topk;
    let mut expanded: Vec<(u32, usize, u32)> = Vec::with_capacity(total_expanded);
    for (token_idx, row) in indices_cpu.iter().enumerate() {
        for (slot_idx, &expert_id) in row.iter().enumerate() {
            if expert_id as usize >= num_experts {
                candle_core::bail!(
                    "moe_gemm_nvfp4: expert index {} out of range for {} experts",
                    expert_id,
                    num_experts
                );
            }
            let expanded_row = token_idx * topk + slot_idx;
            let source_row = if input_has_topk_dim {
                expanded_row as u32
            } else {
                token_idx as u32
            };
            expanded.push((expert_id, expanded_row, source_row));
        }
    }

    expanded.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

    let sorted_token_ids: Vec<u32> = expanded.iter().map(|&(_, _, src)| src).collect();
    let scatter_ids: Vec<u32> = expanded.iter().map(|&(_, orig, _)| orig as u32).collect();

    let mut expert_offsets = vec![0u32; num_experts];
    let mut expert_counts = vec![0u32; num_experts];
    for &(expert_id, _, _) in &expanded {
        expert_counts[expert_id as usize] += 1;
    }
    let mut offset = 0u32;
    for expert_idx in 0..num_experts {
        expert_offsets[expert_idx] = offset;
        offset += expert_counts[expert_idx];
    }

    let mut problem_sizes = vec![0u32; num_experts * 3];
    let mut sf_offsets = vec![0u32; num_experts];
    let mut total_sf_rows = 0usize;
    for expert_idx in 0..num_experts {
        problem_sizes[expert_idx * 3] = expert_counts[expert_idx];
        problem_sizes[expert_idx * 3 + 1] = n as u32;
        problem_sizes[expert_idx * 3 + 2] = k as u32;
        sf_offsets[expert_idx] = total_sf_rows as u32;
        total_sf_rows += pad_to(expert_counts[expert_idx] as usize, 128);
    }

    Ok(RoutedRowsMetadata {
        sorted_token_ids,
        scatter_ids,
        expert_offsets,
        expert_counts,
        sf_offsets,
        problem_sizes,
        total_sf_rows,
        total_expanded,
    })
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_dtype_code(dtype: candle_core::DType) -> Result<i32> {
    match dtype {
        candle_core::DType::F16 => Ok(0),
        candle_core::DType::BF16 => Ok(1),
        _ => candle_core::bail!("only f16/bf16 are supported for flashinfer fused moe"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_f16bf16(t: &Tensor) -> Result<*const core::ffi::c_void> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F16) => {
            Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr() as *const core::ffi::c_void)
        }
        (candle::Storage::Cuda(c), candle_core::DType::BF16) => {
            Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const core::ffi::c_void)
        }
        _ => candle_core::bail!("expected CUDA f16/bf16 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_u8(t: &Tensor) -> Result<*const u8> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::U8) => {
            Ok(*c.as_cuda_slice::<u8>()?.device_ptr() as *const u8)
        }
        _ => candle_core::bail!("expected CUDA u8 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_f32(t: &Tensor) -> Result<*const f32> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F32) => {
            Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *const f32)
        }
        _ => candle_core::bail!("expected CUDA f32 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_topk_ids_i32(t: &Tensor) -> Result<*const i32> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::U32) => {
            Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *const i32)
        }
        _ => candle_core::bail!("expected CUDA u32 tensor for topk ids"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_mut_ptr_f16bf16(t: &Tensor) -> Result<*mut core::ffi::c_void> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F16) => {
            Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr() as *mut core::ffi::c_void)
        }
        (candle::Storage::Cuda(c), candle_core::DType::BF16) => {
            Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr() as *mut core::ffi::c_void)
        }
        _ => candle_core::bail!("expected CUDA f16/bf16 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_fused_moe(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    down_weights: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "trtllm")]
    crate::trtllm_cubin_loader::init_cubin_loader();

    let (num_tokens, hidden_size) = input.dims2()?;
    let (num_experts, gate_up_n, gate_up_k) = gate_up_weights.dims3()?;
    let (down_experts, down_n, down_k) = down_weights.dims3()?;
    let (topk_tokens, top_k) = topk_ids.dims2()?;
    let (topk_w_tokens, topk_w_k) = topk_weights.dims2()?;
    if !input.is_contiguous()
        || !topk_ids.is_contiguous()
        || !topk_weights.is_contiguous()
        || !gate_up_weights.is_contiguous()
        || !down_weights.is_contiguous()
    {
        candle_core::bail!("flashinfer fused moe expects contiguous tensors");
    }
    if topk_tokens != num_tokens || topk_w_tokens != num_tokens || topk_w_k != top_k {
        candle_core::bail!("flashinfer fused moe: invalid topk tensors");
    }
    if gate_up_k != hidden_size || down_experts != num_experts || down_n != hidden_size {
        candle_core::bail!("flashinfer fused moe: invalid tensor shapes for moe weights");
    }
    if gate_up_n % 2 != 0 {
        candle_core::bail!("flashinfer fused moe: gate_up second dim must be even");
    }
    if down_k * 2 != gate_up_n {
        candle_core::bail!("flashinfer fused moe: gate_up/down intermediate dims mismatch");
    }
    let input_dtype = cuda_dtype_code(input.dtype())?;
    let weight_dtype = cuda_dtype_code(gate_up_weights.dtype())?;
    if input_dtype != weight_dtype {
        candle_core::bail!("flashinfer fused moe: input and weight dtype must match");
    }
    let dev = input.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let output = Tensor::zeros((num_tokens, hidden_size), input.dtype(), input.device())?;
    let status = unsafe {
        ffi::flashinfer_fused_moe_bf16(
            cuda_ptr_f16bf16(input)?,
            cuda_ptr_topk_ids_i32(topk_ids)?,
            cuda_ptr_f32(topk_weights)?,
            cuda_ptr_f16bf16(gate_up_weights)?,
            cuda_ptr_f16bf16(down_weights)?,
            cuda_mut_ptr_f16bf16(&output)?,
            num_tokens as i32,
            hidden_size as i32,
            down_k as i32,
            num_experts as i32,
            top_k as i32,
            input_dtype,
            weight_dtype,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("flashinfer fused moe bf16 kernel failed with status {status}");
    }
    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_fused_moe_fp8(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    gate_up_scales: &Tensor,
    down_weights: &Tensor,
    down_scales: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "trtllm")]
    crate::trtllm_cubin_loader::init_cubin_loader();

    let (num_tokens, hidden_size) = input.dims2()?;
    let (num_experts, gate_up_n, gate_up_k) = gate_up_weights.dims3()?;
    let (down_experts, down_n, down_k) = down_weights.dims3()?;
    let (gate_up_scale_experts, gate_up_scale_n, gate_up_scale_k) = gate_up_scales.dims3()?;
    let (down_scale_experts, down_scale_n, down_scale_k) = down_scales.dims3()?;
    let (topk_tokens, top_k) = topk_ids.dims2()?;
    let (topk_w_tokens, topk_w_k) = topk_weights.dims2()?;
    if !input.is_contiguous()
        || !topk_ids.is_contiguous()
        || !topk_weights.is_contiguous()
        || !gate_up_weights.is_contiguous()
        || !gate_up_scales.is_contiguous()
        || !down_weights.is_contiguous()
        || !down_scales.is_contiguous()
    {
        candle_core::bail!("flashinfer fused moe fp8 expects contiguous tensors");
    }
    if gate_up_weights.dtype() != candle_core::DType::U8
        || down_weights.dtype() != candle_core::DType::U8
        || gate_up_scales.dtype() != candle_core::DType::F32
        || down_scales.dtype() != candle_core::DType::F32
    {
        candle_core::bail!("flashinfer fused moe fp8 expects u8 weights and f32 scales");
    }
    if topk_tokens != num_tokens || topk_w_tokens != num_tokens || topk_w_k != top_k {
        candle_core::bail!("flashinfer fused moe fp8: invalid topk tensors");
    }
    if gate_up_k != hidden_size || down_experts != num_experts || down_n != hidden_size {
        candle_core::bail!("flashinfer fused moe fp8: invalid tensor shapes for moe weights");
    }
    if gate_up_n % 2 != 0 || down_k * 2 != gate_up_n {
        candle_core::bail!("flashinfer fused moe fp8: gate_up/down intermediate dims mismatch");
    }
    if gate_up_scale_experts != num_experts || down_scale_experts != num_experts {
        candle_core::bail!("flashinfer fused moe fp8: scale tensor expert dim mismatch");
    }
    if hidden_size % 128 != 0 || down_k % 128 != 0 {
        candle_core::bail!(
            "flashinfer fused moe fp8: hidden/intermediate dims must be divisible by 128"
        );
    }
    let expected_gate_up_scale_n = gate_up_n / 128;
    let expected_gate_up_scale_k = hidden_size / 128;
    let expected_down_scale_n = hidden_size / 128;
    let expected_down_scale_k = down_k / 128;
    if gate_up_scale_n != expected_gate_up_scale_n
        || gate_up_scale_k != expected_gate_up_scale_k
        || down_scale_n != expected_down_scale_n
        || down_scale_k != expected_down_scale_k
    {
        candle_core::bail!(
            "flashinfer fused moe fp8: invalid scale tensor shapes, expected gate_up=[{num_experts}, {expected_gate_up_scale_n}, {expected_gate_up_scale_k}], down=[{num_experts}, {expected_down_scale_n}, {expected_down_scale_k}]"
        );
    }
    let input_dtype = cuda_dtype_code(input.dtype())?;
    let dev = input.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let output = Tensor::zeros(
        (num_tokens, hidden_size),
        candle_core::DType::BF16,
        input.device(),
    )?;
    let status = unsafe {
        ffi::flashinfer_fused_moe_fp8(
            cuda_ptr_f16bf16(input)?,
            cuda_ptr_topk_ids_i32(topk_ids)?,
            cuda_ptr_f32(topk_weights)?,
            cuda_ptr_u8(gate_up_weights)?,
            cuda_ptr_f32(gate_up_scales)?,
            cuda_ptr_u8(down_weights)?,
            cuda_ptr_f32(down_scales)?,
            cuda_mut_ptr_f16bf16(&output)?,
            num_tokens as i32,
            hidden_size as i32,
            down_k as i32,
            num_experts as i32,
            top_k as i32,
            input_dtype,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("flashinfer fused moe fp8 kernel failed with status {status}");
    }
    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_mxfp4_fused_moe(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    gate_up_scales: &Tensor,
    down_weights: &Tensor,
    down_scales: &Tensor,
    num_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    use candle_core::Storage;

    #[cfg(feature = "trtllm")]
    crate::trtllm_cubin_loader::init_cubin_loader();

    let dev = input.device();
    let dtype = input.dtype();

    let sm_version = crate::cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0) as usize;
    if sm_version < 100 {
        candle_core::bail!("flashinfer_mxfp4_fused_moe requires Blackwell (sm100+)");
    }

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let input_dtype_code: i32 = match dtype {
        candle_core::DType::F16 => 0,
        candle_core::DType::BF16 => 1,
        _ => candle_core::bail!(
            "flashinfer_mxfp4_fused_moe: unsupported input dtype {:?}",
            dtype
        ),
    };

    let output = Tensor::zeros((num_tokens, hidden_size), dtype, dev)?;

    fn cuda_ptr(s: &Storage, dtype: candle_core::DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                candle_core::DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                candle_core::DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                candle_core::DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                candle_core::DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                candle_core::DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    {
        let (input_s, _) = input.storage_and_layout();
        let (topk_ids_s, _) = topk_ids.storage_and_layout();
        let (topk_weights_s, _) = topk_weights.storage_and_layout();
        let (gate_up_w_s, _) = gate_up_weights.storage_and_layout();
        let (gate_up_s_s, _) = gate_up_scales.storage_and_layout();
        let (down_w_s, _) = down_weights.storage_and_layout();
        let (down_s_s, _) = down_scales.storage_and_layout();
        let (output_s, _) = output.storage_and_layout();

        let status = unsafe {
            ffi::flashinfer_fused_moe_mxfp4(
                cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                cuda_ptr(&topk_ids_s, candle_core::DType::U32)? as *const i32,
                cuda_ptr(&topk_weights_s, candle_core::DType::F32)? as *const f32,
                cuda_ptr(&gate_up_w_s, candle_core::DType::U8)? as *const u8,
                cuda_ptr(&gate_up_s_s, candle_core::DType::U8)? as *const u8,
                cuda_ptr(&down_w_s, candle_core::DType::U8)? as *const u8,
                cuda_ptr(&down_s_s, candle_core::DType::U8)? as *const u8,
                cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                num_tokens as i32,
                hidden_size as i32,
                intermediate_size as i32,
                num_experts as i32,
                top_k as i32,
                input_dtype_code,
                stream,
            )
        };

        if status != 0 {
            candle_core::bail!("flashinfer_fused_moe_mxfp4 returned error code {}", status);
        }
    }

    Ok(output)
}

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (input_rows, size_k1) = input.dims2()?;
        let size_m = if topk_weights.is_none() {
            input_rows * topk
        } else {
            input_rows
        };
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_wmma only accept f16/bf16 inputs!")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            if is_prefill || size_m > 128 {
                let expert_counts = dev.alloc::<u32>(num_experts).w()?;
                let expert_offsets = dev.alloc::<u32>(num_experts + 1).w()?;
                ffi::moe_gemm_wmma(
                    *input.device_ptr() as *const c_void,   // [size_m, size_k]
                    *weights.device_ptr() as *const c_void, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    *expert_counts.device_ptr() as *mut i32, // pre-allocated buffer [num_experts]
                    *expert_offsets.device_ptr() as *mut i32, // pre-allocated buffer [num_experts + 1]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32, // 0=float16, 1=bf16 (for input/output)
                    is_prefill,
                    stream as i64,
                );
            } else {
                ffi::moe_gemv(
                    *input.device_ptr() as *const c_void,   // [size_m, size_k]
                    *weights.device_ptr() as *const c_void, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32, // 0=float16, 1=bf16 (for input/output)
                    stream as i64,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm only accept f16/bf16 inputs!")
        }
    }
}

/// MoE GEMM with FP8 weights and block-wise scales.
///
/// # Arguments
/// * `input` - Input tensor [size_m, size_k] in F16/BF16
/// * `weights` - FP8 weights as U8 tensor [num_experts, size_n, size_k]
/// * `weight_scales` - Block-wise scales [num_experts, scale_n_dim, scale_k_dim] in F32
/// * `topk_weights` - Optional per-token gating weights [size_m]
/// * `sorted_token_ids` - Sorted token indices [size_m]
/// * `experts_ids` - Expert IDs [size_m]
/// * `topk` - Number of experts per token
/// * `block_size_n` - Block size in N dimension for scales
/// * `block_size_k` - Block size in K dimension for scales
/// * `is_prefill` - Whether this is prefill (uses WMMA) or decode (uses GEMV)
#[cfg(feature = "cuda")]
pub fn moe_gemm_fp8(
    input: &Tensor,
    weights: &Tensor,       // U8 tensor for FP8 weights
    weight_scales: &Tensor, // F32 tensor for scales
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    block_size_n: usize,
    block_size_k: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        weight_scales: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        block_size_n: usize,
        block_size_k: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (input_rows, size_k1) = input.dims2()?;
        let size_m = if topk_weights.is_none() {
            input_rows * topk
        } else {
            input_rows
        };
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );

        // Validate weight dtype is U8 (FP8)
        assert!(
            weights.dtype() == DType::U8,
            "moe_gemm_fp8 expects U8 weights for FP8, got {:?}",
            weights.dtype()
        );

        assert!(
            weight_scales.dtype() == DType::F32,
            "moe_gemm_fp8 expects f32 scales, got {:?}",
            weight_scales.dtype()
        );

        #[cfg(feature = "cutlass")]
        let device = input.device().clone();
        let input_dtype = input.dtype();
        let dev = input.device().as_cuda_device()?;
        #[cfg(feature = "cutlass")]
        let sm_version = crate::cuda_utils::sm_version(dev).unwrap_or(0) as i32;
        let data_type = match input_dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_fp8 only accepts f16/bf16 inputs!")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
            _ => candle::bail!("weights must be a cuda tensor"),
        };

        let (weight_scales, _) = weight_scales.storage_and_layout();
        let weight_scales = match &*weight_scales {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("weight_scales must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        #[cfg(feature = "cutlass")]
        let use_cutlass = sm_version >= 90 && block_size_n == 128 && block_size_k == 128;
        #[cfg(not(feature = "cutlass"))]
        let use_cutlass = false;

        if use_cutlass && (is_prefill || size_m > 128) {
            #[cfg(feature = "cutlass")]
            {
                let k_blocks = (size_k + block_size_k - 1) / block_size_k;
                let num_groups_per_row = k_blocks;
                let num_groups = (input_rows * num_groups_per_row) as i32;

                // SM100+ (Blackwell) requires column-major scale layout (UMMA::Major::MN)
                // SM90 (Hopper) requires row-major scale layout (GMMA::Major::K)
                let is_column_major_scales = sm_version >= 100;

                let input_q = Tensor::zeros((input_rows, size_k), DType::U8, &device)?;
                let input_scale = if is_column_major_scales {
                    // Column-major: allocate transposed and transpose for column-major view
                    Tensor::zeros((k_blocks, input_rows), DType::F32, &device)?.t()?
                } else {
                    // Row-major: standard contiguous layout
                    Tensor::zeros((input_rows, k_blocks), DType::F32, &device)?
                };
                let rep_a_q = Tensor::zeros((size_m, size_k), DType::U8, &device)?;
                let rep_a_scales = if is_column_major_scales {
                    Tensor::zeros((k_blocks, size_m), DType::F32, &device)?.t()?
                } else {
                    Tensor::zeros((size_m, k_blocks), DType::F32, &device)?
                };
                let rep_out = Tensor::zeros((size_m, size_n), input_dtype, &device)?;
                let output = Tensor::zeros((size_m, size_n), input_dtype, &device)?;
                let map_divisor = if topk_weights.is_none() {
                    topk as i32
                } else {
                    1
                };

                // Get scale stride for quantization kernel
                let input_scale_stride = if is_column_major_scales {
                    input_rows as i32 // Column-major stride
                } else {
                    num_groups_per_row as i32 // Row-major stride
                };

                let (input_q, _) = input_q.storage_and_layout();
                let input_q = match &*input_q {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                    _ => candle::bail!("input_q must be a cuda tensor"),
                };
                let (input_scale, _) = input_scale.storage_and_layout();
                let input_scale = match &*input_scale {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input_scale must be a cuda tensor"),
                };
                let (rep_a_q, _) = rep_a_q.storage_and_layout();
                let rep_a_q = match &*rep_a_q {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                    _ => candle::bail!("rep_a_q must be a cuda tensor"),
                };
                let (rep_a_scales, _) = rep_a_scales.storage_and_layout();
                let rep_a_scales = match &*rep_a_scales {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("rep_a_scales must be a cuda tensor"),
                };
                let (rep_out, _) = rep_out.storage_and_layout();
                let rep_out = match &*rep_out {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("rep_out must be a cuda tensor"),
                };
                let (output, _) = output.storage_and_layout();
                let output = match &*output {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("output must be a cuda tensor"),
                };

                let expert_counts = unsafe { dev.alloc::<i32>(num_experts).w()? };
                let expert_offsets = unsafe { dev.alloc::<i32>(num_experts + 1).w()? };

                let stream = *dev.cu_stream() as i64;
                use core::ffi::c_void;
                unsafe {
                    ffi::fp8_quantize_per_token_group_launch(
                        *input.device_ptr() as *const c_void,
                        *input_q.device_ptr() as *mut c_void,
                        *input_scale.device_ptr() as *mut f32,
                        num_groups as i32,
                        128,
                        num_groups_per_row as i32,
                        input_scale_stride,
                        data_type == 0,
                        is_column_major_scales,
                        stream as i64,
                    );

                    ffi::moe_fp8_shuffle_rows_u8(
                        *input_q.device_ptr() as *const u8,
                        *sorted_token_ids.device_ptr() as *const i32,
                        *rep_a_q.device_ptr() as *mut u8,
                        input_rows as i64,
                        size_m as i64,
                        size_k as i64,
                        map_divisor,
                        stream as i64,
                    );

                    // Use strided shuffle for column-major scales (SM100+ Blackwell)
                    // or regular shuffle for row-major scales (SM90)
                    if is_column_major_scales {
                        ffi::moe_fp8_shuffle_rows_f32_strided(
                            *input_scale.device_ptr() as *const f32,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *rep_a_scales.device_ptr() as *mut f32,
                            input_rows as i64,
                            size_m as i64,
                            num_groups_per_row as i64,
                            input_rows as i64, // src_row_stride (column-major)
                            size_m as i64,     // dst_row_stride (column-major)
                            map_divisor,
                            stream as i64,
                        );
                    } else {
                        ffi::moe_fp8_shuffle_rows_f32(
                            *input_scale.device_ptr() as *const f32,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *rep_a_scales.device_ptr() as *mut f32,
                            input_rows as i64,
                            size_m as i64,
                            num_groups_per_row as i64,
                            map_divisor,
                            stream as i64,
                        );
                    }

                    ffi::calculate_expert_offsets(
                        *experts_ids.device_ptr() as *const i32,
                        *expert_counts.device_ptr() as *mut i32,
                        *expert_offsets.device_ptr() as *mut i32,
                        num_experts as i32,
                        size_m as i32,
                        stream as i64,
                    );

                    if data_type == 0 {
                        ffi::moe_fp8_grouped_gemm_f16(
                            *rep_a_q.device_ptr() as *const u8,
                            *weights.device_ptr() as *const u8,
                            *rep_a_scales.device_ptr() as *const f32,
                            *weight_scales.device_ptr() as *const f32,
                            *expert_offsets.device_ptr() as *const i32,
                            num_experts as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            block_size_n as i32,
                            block_size_k as i32,
                            sm_version as i32,
                            *rep_out.device_ptr() as *mut c_void,
                            stream as i64,
                        );
                        ffi::moe_fp8_scatter_rows_f16(
                            *rep_out.device_ptr() as *const c_void,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *output.device_ptr() as *mut c_void,
                            size_m as i64,
                            size_m as i64,
                            size_n as i64,
                            topk_weights_ptr,
                            stream as i64,
                        );
                    } else {
                        ffi::moe_fp8_grouped_gemm_bf16(
                            *rep_a_q.device_ptr() as *const u8,
                            *weights.device_ptr() as *const u8,
                            *rep_a_scales.device_ptr() as *const f32,
                            *weight_scales.device_ptr() as *const f32,
                            *expert_offsets.device_ptr() as *const i32,
                            num_experts as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            block_size_n as i32,
                            block_size_k as i32,
                            sm_version as i32,
                            *rep_out.device_ptr() as *mut c_void,
                            stream as i64,
                        );
                        ffi::moe_fp8_scatter_rows_bf16(
                            *rep_out.device_ptr() as *const c_void,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *output.device_ptr() as *mut c_void,
                            size_m as i64,
                            size_m as i64,
                            size_n as i64,
                            topk_weights_ptr,
                            stream as i64,
                        );
                    }
                }

                let output = candle::CudaStorage::wrap_cuda_slice(output.clone(), dev.clone());
                let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;
                return Ok(output);
            }
        }

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            if is_prefill || size_m > 128 {
                let expert_counts = dev.alloc::<u32>(num_experts).w()?;
                let expert_offsets = dev.alloc::<u32>(num_experts + 1).w()?;
                ffi::moe_gemm_wmma_fp8(
                    *input.device_ptr() as *const c_void,
                    *weights.device_ptr() as *const u8,
                    *weight_scales.device_ptr() as *const f32,
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void,
                    *expert_counts.device_ptr() as *mut i32,
                    *expert_offsets.device_ptr() as *mut i32,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    block_size_n as i32,
                    block_size_k as i32,
                    data_type as i32,
                    is_prefill,
                    stream as i64,
                );
            } else {
                ffi::moe_gemv_fp8(
                    *input.device_ptr() as *const c_void,
                    *weights.device_ptr() as *const u8,
                    *weight_scales.device_ptr() as *const f32,
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    block_size_n as i32,
                    block_size_k as i32,
                    data_type as i32,
                    stream as i64,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            block_size_n,
            block_size_k,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            block_size_n,
            block_size_k,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm_fp8 only accepts f16/bf16 inputs!")
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm_fp8(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: usize,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_fp8 is not implemented on this platform!")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_nvfp4(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
    input_scales: Option<&Tensor>,
    biases: Option<&Tensor>,
    indices: &Tensor,
    pre_sorted: Option<(&Tensor, &Tensor)>,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle_core::{DType, Storage};

    fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = if weights.is_contiguous() {
        weights.clone()
    } else {
        weights.contiguous()?
    };
    let weight_scales = if weight_scales.is_contiguous() {
        weight_scales.clone()
    } else {
        weight_scales.contiguous()?
    };
    let weight_global_scales = if weight_global_scales.is_contiguous() {
        weight_global_scales.clone()
    } else {
        weight_global_scales.contiguous()?
    };
    let indices = if indices.is_contiguous() {
        indices.clone()
    } else {
        indices.contiguous()?
    };

    let indices_dims = indices.dims();
    if indices_dims.len() != 2 {
        candle_core::bail!(
            "moe_gemm_nvfp4: expected indices rank 2 [num_tokens, topk], got {:?}",
            indices_dims
        );
    }
    let num_tokens = indices_dims[0];
    let topk = indices_dims[1];

    let input_dims = input.dims();
    let (k, input_has_topk_dim) = match input_dims {
        [t, kk] => {
            if *t != num_tokens {
                candle_core::bail!(
                    "moe_gemm_nvfp4: input/indices mismatch: input tokens={t}, indices tokens={num_tokens}"
                );
            }
            (*kk, false)
        }
        [t, tk, kk] => {
            if *t != num_tokens || *tk != topk {
                candle_core::bail!(
                    "moe_gemm_nvfp4: input/indices mismatch: input={input_dims:?}, indices={indices_dims:?}"
                );
            }
            (*kk, true)
        }
        _ => candle_core::bail!(
            "moe_gemm_nvfp4: expected input rank 2 or 3, got {:?}",
            input_dims
        ),
    };

    if k % crate::nvfp4_linear::NVFP4_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "moe_gemm_nvfp4: K must be divisible by {}, got K={k}",
            crate::nvfp4_linear::NVFP4_BLOCK_SIZE
        );
    }

    let w_dims = weights.dims();
    if w_dims.len() != 3 {
        candle_core::bail!(
            "moe_gemm_nvfp4: expected weights rank 3 [E, N, K/2], got {:?}",
            w_dims
        );
    }
    let num_experts = w_dims[0];
    let n = w_dims[1];
    if w_dims[2] != k / 2 {
        candle_core::bail!(
            "moe_gemm_nvfp4: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            w_dims
        );
    }

    let dtype = input.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) {
        candle_core::bail!("moe_gemm_nvfp4 only accepts f16/bf16 inputs");
    }

    let dev = input.device();
    let cuda_dev = dev.as_cuda_device()?;

    // During prefill, sort by expert and dispatch to grouped kernels:
    // hardware FP4 (SM100+) or software WMMA (SM80+).
    // Prefill never uses CUDA graphs, so thrust-based expert offset is safe.
    if is_prefill {
        use crate::sort::ArgSortOp;

        let (sorted_expert_ids, sorted_token_ids) = if let Some((stids, seids)) = pre_sorted {
            (seids.clone(), stids.clone())
        } else {
            let flat_indices = indices.flatten_all()?.contiguous()?;
            flat_indices.sort(true)?
        };
        let total_slots = num_tokens * topk;

        #[cfg(feature = "cutlass")]
        {
            let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
            if sm >= 100 && n % 32 == 0 && k % 32 == 0 {
                let routed_input = if input_has_topk_dim {
                    input.reshape((num_tokens * topk, k))?
                } else {
                    input.clone()
                };

                let output = moe_gemm_nvfp4_hardware(
                    &routed_input,
                    &weights,
                    &weight_scales,
                    &weight_global_scales,
                    input_scales,
                    &None,
                    &sorted_token_ids,
                    &sorted_expert_ids,
                    topk,
                    is_prefill,
                )?;
                return output.reshape((num_tokens, topk, n));
            }
        }

        // WMMA grouped MoE path (SM80+): compute expert offsets on GPU.
        let sorted_expert_ids_u32 = sorted_expert_ids.to_dtype(DType::U32)?;
        let sorted_token_ids_u32 = sorted_token_ids.to_dtype(DType::U32)?;
        let expert_counts_t = Tensor::zeros((num_experts,), DType::U32, dev)?;
        let expert_offsets_t = Tensor::zeros((num_experts + 1,), DType::U32, dev)?;
        {
            let stream = *cuda_dev.cu_stream() as i64;
            let (seids_s, _) = sorted_expert_ids_u32.storage_and_layout();
            let (ec_s, _) = expert_counts_t.storage_and_layout();
            let (eo_s, _) = expert_offsets_t.storage_and_layout();
            unsafe {
                ffi::calculate_expert_offsets(
                    cuda_ptr(&seids_s, DType::U32)? as *const i32,
                    cuda_ptr(&ec_s, DType::U32)? as *mut i32,
                    cuda_ptr(&eo_s, DType::U32)? as *mut i32,
                    num_experts as i32,
                    total_slots as i32,
                    stream,
                );
            }
        }

        let routed_input = if input_has_topk_dim {
            input.reshape((num_tokens * topk, k))?
        } else {
            input.clone()
        };

        let output = Tensor::zeros((num_tokens * topk, n), dtype, dev)?;
        {
            let stream = *cuda_dev.cu_stream() as i64;
            let (input_s, _) = routed_input.storage_and_layout();
            let (weights_s, _) = weights.storage_and_layout();
            let (scales_s, _) = weight_scales.storage_and_layout();
            let (gscales_s, _) = weight_global_scales.storage_and_layout();
            let (stids_s, _) = sorted_token_ids_u32.storage_and_layout();
            let (eoffs_s, _) = expert_offsets_t.storage_and_layout();
            let (output_s, _) = output.storage_and_layout();

            unsafe {
                match dtype {
                    DType::F16 => ffi::nvfp4_moe_gemm_wmma_f16(
                        cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                        cuda_ptr(&weights_s, DType::U8)? as *const u8,
                        cuda_ptr(&scales_s, DType::U8)? as *const u8,
                        cuda_ptr(&gscales_s, DType::F32)? as *const f32,
                        cuda_ptr(&stids_s, DType::U32)? as *const i32,
                        cuda_ptr(&eoffs_s, DType::U32)? as *const i32,
                        std::ptr::null(),
                        cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                        num_experts as i32,
                        topk as i32,
                        total_slots as i32,
                        n as i32,
                        k as i32,
                        input_has_topk_dim,
                        stream,
                    ),
                    DType::BF16 => ffi::nvfp4_moe_gemm_wmma_bf16(
                        cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                        cuda_ptr(&weights_s, DType::U8)? as *const u8,
                        cuda_ptr(&scales_s, DType::U8)? as *const u8,
                        cuda_ptr(&gscales_s, DType::F32)? as *const f32,
                        cuda_ptr(&stids_s, DType::U32)? as *const i32,
                        cuda_ptr(&eoffs_s, DType::U32)? as *const i32,
                        std::ptr::null(),
                        cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                        num_experts as i32,
                        topk as i32,
                        total_slots as i32,
                        n as i32,
                        k as i32,
                        input_has_topk_dim,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
        }

        return output.reshape((num_tokens, topk, n));
    }

    let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;
    {
        let stream = *cuda_dev.cu_stream() as i64;

        let (input_s, _) = input.storage_and_layout();
        let (weights_s, _) = weights.storage_and_layout();
        let (scales_s, _) = weight_scales.storage_and_layout();
        let (gscales_s, _) = weight_global_scales.storage_and_layout();
        let (indices_s, _) = indices.storage_and_layout();
        let (output_s, _) = output.storage_and_layout();

        let biases_ptr = if let Some(b) = biases {
            let (b_s, _) = b.storage_and_layout();
            cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };

        unsafe {
            match dtype {
                DType::F16 => ffi::nvfp4_indexed_moe_gemm_f16(
                    cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&weights_s, DType::U8)? as *const u8,
                    cuda_ptr(&scales_s, DType::U8)? as *const u8,
                    cuda_ptr(&gscales_s, DType::F32)? as *const f32,
                    biases_ptr,
                    cuda_ptr(&indices_s, DType::U32)? as *const u32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    num_tokens as i32,
                    topk as i32,
                    num_experts as i32,
                    n as i32,
                    k as i32,
                    biases.is_some(),
                    input_has_topk_dim,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_indexed_moe_gemm_bf16(
                    cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&weights_s, DType::U8)? as *const u8,
                    cuda_ptr(&scales_s, DType::U8)? as *const u8,
                    cuda_ptr(&gscales_s, DType::F32)? as *const f32,
                    biases_ptr,
                    cuda_ptr(&indices_s, DType::U32)? as *const u32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    num_tokens as i32,
                    topk as i32,
                    num_experts as i32,
                    n as i32,
                    k as i32,
                    biases.is_some(),
                    input_has_topk_dim,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    Ok(output)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_nvfp4(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&Tensor>,
    _: Option<&Tensor>,
    _: &Tensor,
    _: Option<(&Tensor, &Tensor)>,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_nvfp4 is not implemented on this platform!")
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
#[allow(clippy::too_many_arguments)]
/// Routed NVFP4 MoE GEMM.
///
/// This mirrors the FP8 MoE contract: the caller provides expert-grouped
/// `sorted_token_ids` and `experts_ids` so the kernel wrapper does not need to
/// rebuild routing from raw `topk_ids`.
pub fn moe_gemm_nvfp4_hardware(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
    input_scales: Option<&Tensor>,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    _is_prefill: bool,
) -> Result<Tensor> {
    use candle_core::{DType, Storage};

    fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = if weights.is_contiguous() {
        weights.clone()
    } else {
        weights.contiguous()?
    };
    let weight_scales = if weight_scales.is_contiguous() {
        weight_scales.clone()
    } else {
        weight_scales.contiguous()?
    };
    let weight_global_scales = if weight_global_scales.is_contiguous() {
        weight_global_scales.clone()
    } else {
        weight_global_scales.contiguous()?
    };
    let sorted_token_ids = if sorted_token_ids.is_contiguous() {
        sorted_token_ids.clone()
    } else {
        sorted_token_ids.contiguous()?
    };
    let experts_ids = if experts_ids.is_contiguous() {
        experts_ids.clone()
    } else {
        experts_ids.contiguous()?
    };

    let (input_rows, k) = input.dims2()?;
    let (num_experts, n, packed_k) = weights.dims3()?;
    if packed_k != k / 2 {
        candle_core::bail!(
            "moe_gemm_nvfp4: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            weights.dims()
        );
    }
    if k % crate::nvfp4_linear::NVFP4_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "moe_gemm_nvfp4: K must be divisible by {}, got K={k}",
            crate::nvfp4_linear::NVFP4_BLOCK_SIZE
        );
    }

    let size_m = sorted_token_ids.elem_count();
    if sorted_token_ids.elem_count() != size_m || experts_ids.elem_count() != size_m {
        candle_core::bail!(
            "moe_gemm_nvfp4: routed tensors must have {} elements, got sorted={} experts={}",
            size_m,
            sorted_token_ids.elem_count(),
            experts_ids.elem_count()
        );
    }
    let map_divisor = if input_rows == size_m {
        1
    } else if input_rows * topk == size_m {
        topk as i32
    } else {
        candle_core::bail!(
            "moe_gemm_nvfp4_hardware: input rows {} are incompatible with routed size {} and topk {}",
            input_rows,
            size_m,
            topk
        );
    };

    let dtype = input.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) {
        candle_core::bail!("moe_gemm_nvfp4_hardware only accepts f16/bf16 inputs");
    }

    let dev = input.device();
    let cuda_dev = dev.as_cuda_device()?;
    let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
    if sm < 100 {
        candle_core::bail!(
            "moe_gemm_nvfp4_hardware requires Blackwell (sm100+), got sm{}",
            sm
        );
    }

    let stream = *cuda_dev.cu_stream() as i64;
    let expert_counts_t = Tensor::zeros((num_experts,), DType::U32, dev)?;
    let expert_offsets_t = Tensor::zeros((num_experts + 1,), DType::U32, dev)?;
    let sf_offsets_t = Tensor::zeros((num_experts,), DType::U32, dev)?;
    let problem_sizes_t = Tensor::zeros((num_experts * 3,), DType::U32, dev)?;
    let alphas_t = Tensor::zeros((num_experts,), DType::F32, dev)?;
    let input_scale_invs_t = Tensor::zeros((num_experts,), DType::F32, dev)?;

    {
        let (experts_ids_s, _) = experts_ids.storage_and_layout();
        let (expert_counts_s, _) = expert_counts_t.storage_and_layout();
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        unsafe {
            ffi::calculate_expert_offsets(
                cuda_ptr(&experts_ids_s, DType::U32)? as *const i32,
                cuda_ptr(&expert_counts_s, DType::U32)? as *mut i32,
                cuda_ptr(&expert_offsets_s, DType::U32)? as *mut i32,
                num_experts as i32,
                size_m as i32,
                stream,
            );
        }
    }

    {
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        let (weight_global_scales_s, _) = weight_global_scales.storage_and_layout();
        let (sf_offsets_s, _) = sf_offsets_t.storage_and_layout();
        let (problem_sizes_s, _) = problem_sizes_t.storage_and_layout();
        let (alphas_s, _) = alphas_t.storage_and_layout();
        let (input_scale_invs_s, _) = input_scale_invs_t.storage_and_layout();
        let input_scales_ptr = if let Some(scales) = input_scales {
            let (scales_s, _) = scales.storage_and_layout();
            cuda_ptr(&scales_s, DType::F32)? as *const f32
        } else {
            std::ptr::null()
        };
        unsafe {
            ffi::nvfp4_moe_build_metadata(
                cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                cuda_ptr(&weight_global_scales_s, DType::F32)? as *const f32,
                input_scales_ptr,
                cuda_ptr(&sf_offsets_s, DType::U32)? as *mut i32,
                cuda_ptr(&problem_sizes_s, DType::U32)? as *mut i32,
                cuda_ptr(&alphas_s, DType::F32)? as *mut f32,
                cuda_ptr(&input_scale_invs_s, DType::F32)? as *mut f32,
                num_experts as i32,
                n as i32,
                k as i32,
                stream,
            );
        }
    }

    let gathered = Tensor::zeros((size_m, k), dtype, dev)?;
    {
        let (input_s, _) = input.storage_and_layout();
        let (gathered_s, _) = gathered.storage_and_layout();
        let (sorted_s, _) = sorted_token_ids.storage_and_layout();
        unsafe {
            match dtype {
                DType::F16 => ffi::nvfp4_moe_gather_f16(
                    cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&gathered_s, dtype)? as *mut std::ffi::c_void,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    size_m as i32,
                    k as i32,
                    map_divisor,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_moe_gather_bf16(
                    cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&gathered_s, dtype)? as *mut std::ffi::c_void,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    size_m as i32,
                    k as i32,
                    map_divisor,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    let k_scale = k / crate::nvfp4_linear::NVFP4_BLOCK_SIZE;
    let k_scale_padded = pad_to(k_scale, 4);
    let act_packed = Tensor::zeros((size_m, k / 2), DType::U8, dev)?;
    let total_sf_rows_capacity = size_m + 127 * num_experts;
    let act_scales_swizzled =
        Tensor::zeros((total_sf_rows_capacity, k_scale_padded), DType::U8, dev)?;
    {
        let (gathered_s, _) = gathered.storage_and_layout();
        let (act_packed_s, _) = act_packed.storage_and_layout();
        let (act_scales_sw_s, _) = act_scales_swizzled.storage_and_layout();
        let (input_scale_invs_s, _) = input_scale_invs_t.storage_and_layout();
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        let (sf_offsets_s, _) = sf_offsets_t.storage_and_layout();
        unsafe {
            match dtype {
                DType::F16 => ffi::nvfp4_quantize_activation_grouped_f16(
                    cuda_ptr(&gathered_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&act_packed_s, DType::U8)? as *mut std::ffi::c_void,
                    cuda_ptr(&act_scales_sw_s, DType::U8)? as *mut std::ffi::c_void,
                    cuda_ptr(&input_scale_invs_s, DType::F32)? as *const f32,
                    cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                    cuda_ptr(&sf_offsets_s, DType::U32)? as *const i32,
                    size_m as i32,
                    num_experts as i32,
                    k as i32,
                    k_scale_padded as i32,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_quantize_activation_grouped_bf16(
                    cuda_ptr(&gathered_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&act_packed_s, DType::U8)? as *mut std::ffi::c_void,
                    cuda_ptr(&act_scales_sw_s, DType::U8)? as *mut std::ffi::c_void,
                    cuda_ptr(&input_scale_invs_s, DType::F32)? as *const f32,
                    cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                    cuda_ptr(&sf_offsets_s, DType::U32)? as *const i32,
                    size_m as i32,
                    num_experts as i32,
                    k as i32,
                    k_scale_padded as i32,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    let n_padded = pad_to(n, 128);
    let weight_scales_swizzled =
        Tensor::zeros((num_experts, n_padded, k_scale_padded), DType::U8, dev)?;
    {
        let (ws_s, _) = weight_scales.storage_and_layout();
        let (wss_s, _) = weight_scales_swizzled.storage_and_layout();
        let ws_base = cuda_ptr(&ws_s, DType::U8)?;
        let wss_base = cuda_ptr(&wss_s, DType::U8)?;
        for e in 0..num_experts {
            let src_offset = (e * n * k_scale) as u64;
            let dst_offset = (e * n_padded * k_scale_padded) as u64;
            unsafe {
                ffi::nvfp4_swizzle_weight_scales(
                    (ws_base + src_offset) as *const std::ffi::c_void,
                    (wss_base + dst_offset) as *mut std::ffi::c_void,
                    n as i32,
                    k_scale as i32,
                    n_padded as i32,
                    k_scale_padded as i32,
                    stream,
                );
            }
        }
    }

    let rep_out = Tensor::zeros((size_m, n), dtype, dev)?;
    {
        let (act_packed_s, _) = act_packed.storage_and_layout();
        let (weights_s, _) = weights.storage_and_layout();
        let (act_scales_sw_s, _) = act_scales_swizzled.storage_and_layout();
        let (wss_s, _) = weight_scales_swizzled.storage_and_layout();
        let (alphas_s, _) = alphas_t.storage_and_layout();
        let (eo_s, _) = expert_offsets_t.storage_and_layout();
        let (sfo_s, _) = sf_offsets_t.storage_and_layout();
        let (ps_s, _) = problem_sizes_t.storage_and_layout();
        let (out_s, _) = rep_out.storage_and_layout();
        let (workspace_ptr, workspace_bytes) = get_moe_cutlass_workspace(cuda_dev, 0)?;
        unsafe {
            let ret = match dtype {
                DType::F16 => ffi::nvfp4_cutlass_moe_gemm_f16(
                    cuda_ptr(&out_s, dtype)? as *mut std::ffi::c_void,
                    cuda_ptr(&act_packed_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&weights_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&act_scales_sw_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&wss_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&alphas_s, DType::F32)? as *const f32,
                    cuda_ptr(&eo_s, DType::U32)? as *const i32,
                    cuda_ptr(&sfo_s, DType::U32)? as *const i32,
                    cuda_ptr(&ps_s, DType::U32)? as *const i32,
                    num_experts as i32,
                    size_m as i32,
                    n as i32,
                    k as i32,
                    workspace_ptr,
                    workspace_bytes as i64,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_cutlass_moe_gemm_bf16(
                    cuda_ptr(&out_s, dtype)? as *mut std::ffi::c_void,
                    cuda_ptr(&act_packed_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&weights_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&act_scales_sw_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&wss_s, DType::U8)? as *const std::ffi::c_void,
                    cuda_ptr(&alphas_s, DType::F32)? as *const f32,
                    cuda_ptr(&eo_s, DType::U32)? as *const i32,
                    cuda_ptr(&sfo_s, DType::U32)? as *const i32,
                    cuda_ptr(&ps_s, DType::U32)? as *const i32,
                    num_experts as i32,
                    size_m as i32,
                    n as i32,
                    k as i32,
                    workspace_ptr,
                    workspace_bytes as i64,
                    stream,
                ),
                _ => unreachable!(),
            };
            if ret != 0 {
                candle_core::bail!("nvfp4_cutlass_moe_gemm failed with error code {}", ret);
            }
        }
    }

    let output = Tensor::zeros((size_m, n), dtype, dev)?;
    {
        let (rep_out_s, _) = rep_out.storage_and_layout();
        let (sorted_s, _) = sorted_token_ids.storage_and_layout();
        let (output_s, _) = output.storage_and_layout();
        let weights_ptr = if let Some(t) = topk_weights {
            let (s, _) = t.storage_and_layout();
            cuda_ptr(&s, DType::F32)? as *const f32
        } else {
            std::ptr::null()
        };
        unsafe {
            match dtype {
                DType::F16 => ffi::moe_fp8_scatter_rows_f16(
                    cuda_ptr(&rep_out_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    size_m as i64,
                    size_m as i64,
                    n as i64,
                    weights_ptr,
                    stream,
                ),
                DType::BF16 => ffi::moe_fp8_scatter_rows_bf16(
                    cuda_ptr(&rep_out_s, dtype)? as *const std::ffi::c_void,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    size_m as i64,
                    size_m as i64,
                    n as i64,
                    weights_ptr,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    Ok(output)
}

#[cfg(not(all(feature = "cuda", feature = "cutlass")))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_nvfp4_hardware(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&Tensor>,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_nvfp4_hardware is not implemented on this platform!")
}

#[allow(clippy::too_many_arguments)]
/// MXFP4 MoE GEMM.
///
/// `is_prefill` selects the kernel path: prefill uses the grouped WMMA GEMM
/// (which internally allocates scratch memory via `cudaMallocAsync` and is
/// therefore incompatible with CUDA graph capture), while decode uses the
/// indexed dot-product kernel that is graph-safe.
pub fn moe_gemm_mxfp4(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    biases: Option<&Tensor>,
    indices: &Tensor,
    is_prefill: bool,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = if weights.is_contiguous() {
        weights.clone()
    } else {
        weights.contiguous()?
    };
    let weight_scales = if weight_scales.is_contiguous() {
        weight_scales.clone()
    } else {
        weight_scales.contiguous()?
    };
    let indices = if indices.is_contiguous() {
        indices.clone()
    } else {
        indices.contiguous()?
    };

    let indices_dims = indices.dims();
    if indices_dims.len() != 2 {
        candle_core::bail!(
            "moe_gemm_mxfp4: expected indices rank 2 [num_tokens, topk], got {:?}",
            indices_dims
        );
    }
    let num_tokens = indices_dims[0];
    let topk = indices_dims[1];

    let input_dims = input.dims();
    let (k, input_has_topk_dim) = match input_dims {
        [t, kk] => {
            if *t != num_tokens {
                candle_core::bail!(
                    "moe_gemm_mxfp4: input/indices mismatch: input tokens={t}, indices tokens={num_tokens}"
                );
            }
            (*kk, false)
        }
        [t, tk, kk] => {
            if *t != num_tokens || *tk != topk {
                candle_core::bail!(
                    "moe_gemm_mxfp4: input/indices mismatch: input={input_dims:?}, indices={indices_dims:?}"
                );
            }
            (*kk, true)
        }
        _ => candle_core::bail!(
            "moe_gemm_mxfp4: expected input rank 2 or 3, got {:?}",
            input_dims
        ),
    };

    if k % crate::mxfp4_linear::MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "moe_gemm_mxfp4: K must be divisible by {}, got K={k}",
            crate::mxfp4_linear::MXFP4_BLOCK_SIZE
        );
    }

    let w_dims = weights.dims();
    if w_dims.len() != 3 {
        candle_core::bail!(
            "moe_gemm_mxfp4: expected weights rank 3 [E, N, K/2], got {:?}",
            w_dims
        );
    }
    let num_experts = w_dims[0];
    let n = w_dims[1];

    if w_dims[2] != k / 2 {
        candle_core::bail!(
            "moe_gemm_mxfp4: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            w_dims
        );
    }

    let dev = input.device();
    let dtype = input.dtype();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::{DType, Storage};

            fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
                match s {
                    Storage::Cuda(c) => match dtype {
                        DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                        DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                        DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                        DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                        _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                    },
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            }

            let has_bias = biases.is_some();
            let use_fused = is_prefill;
            let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;

            {
                let (input_s, _) = input.storage_and_layout();
                let (weights_s, _) = weights.storage_and_layout();
                let (scales_s, _) = weight_scales.storage_and_layout();
                let (indices_s, _) = indices.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weights_ptr = cuda_ptr(&weights_s, DType::U8)? as *const u8;
                let scales_ptr = cuda_ptr(&scales_s, DType::U8)? as *const u8;
                let indices_ptr = cuda_ptr(&indices_s, DType::U32)? as *const u32;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let biases_ptr = if let Some(b) = biases {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let stream = *cuda_dev.cu_stream() as i64;

                unsafe {
                    match dtype {
                        DType::F16 => {
                            if use_fused {
                                ffi::mxfp4_moe_grouped_gemm_wmma_f16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            } else {
                                ffi::mxfp4_indexed_moe_gemm_f16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            }
                        }
                        DType::BF16 => {
                            if use_fused {
                                ffi::mxfp4_moe_grouped_gemm_wmma_bf16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            } else {
                                ffi::mxfp4_indexed_moe_gemm_bf16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            }
                        }
                        _ => {
                            candle_core::bail!("moe_gemm_mxfp4 CUDA: unsupported dtype {:?}", dtype)
                        }
                    }
                }
            }

            Ok(output)
        }

        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::{DType, Storage};

            let reuse_topk = !input_has_topk_dim && topk <= 8;
            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();
            let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;

            let (input_s, input_l) = input.storage_and_layout();
            let input_ms = match &*input_s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("input must be metal"),
            };
            let (weights_s, weights_l) = weights.storage_and_layout();
            let weights_ms = match &*weights_s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("weights must be metal"),
            };
            let (scales_s, scales_l) = weight_scales.storage_and_layout();
            let scales_ms = match &*scales_s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("weight_scales must be metal"),
            };
            let (indices_s, indices_l) = indices.storage_and_layout();
            let indices_ms = match &*indices_s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("indices must be metal"),
            };
            let (output_s, _) = output.storage_and_layout();
            let output_ms = match &*output_s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("output must be metal"),
            };

            let x = (
                input_ms.buffer(),
                input_l.start_offset() * dtype.size_in_bytes(),
            );
            let w = (
                weights_ms.buffer(),
                weights_l.start_offset() * weights.dtype().size_in_bytes(),
            );
            let sc = (
                scales_ms.buffer(),
                scales_l.start_offset() * weight_scales.dtype().size_in_bytes(),
            );
            let idx = (
                indices_ms.buffer(),
                indices_l.start_offset() * indices.dtype().size_in_bytes(),
            );

            if let Some(biases) = biases {
                let biases = if biases.is_contiguous() {
                    biases.clone()
                } else {
                    biases.contiguous()?
                };
                let (bias_s, bias_l) = biases.storage_and_layout();
                let bias_ms = match &*bias_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("biases must be metal"),
                };
                let bias_buf = (
                    bias_ms.buffer(),
                    bias_l.start_offset() * biases.dtype().size_in_bytes(),
                );

                metal_kernels::call_mxfp4_moe_gemm(
                    metal_dev.device(),
                    command_buffer_ref,
                    metal_kernels::Kernels::default(),
                    dtype,
                    x,
                    w,
                    sc,
                    bias_buf,
                    idx,
                    output_ms.buffer(),
                    num_tokens,
                    topk,
                    num_experts,
                    n,
                    k,
                    true,
                    input_has_topk_dim,
                    reuse_topk,
                )
                .map_err(candle_core::Error::wrap)?;
            } else {
                let dummy_biases = (input_ms.buffer(), 0usize);

                metal_kernels::call_mxfp4_moe_gemm(
                    metal_dev.device(),
                    command_buffer_ref,
                    metal_kernels::Kernels::default(),
                    dtype,
                    x,
                    w,
                    sc,
                    dummy_biases,
                    idx,
                    output_ms.buffer(),
                    num_tokens,
                    topk,
                    num_experts,
                    n,
                    k,
                    false,
                    input_has_topk_dim,
                    reuse_topk,
                )
                .map_err(candle_core::Error::wrap)?;
            }

            Ok(output)
        }
        _ => candle_core::bail!("moe_gemm_mxfp4: unsupported backend (need CUDA or Metal)"),
    }
}

#[cfg(feature = "metal")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle_core as candle;
    use candle_core::{DType, Storage};

    let (input_rows, size_k1) = input.dims2()?;
    let size_m = if topk_weights.is_none() {
        input_rows * topk
    } else {
        input_rows
    };
    let (num_experts, size_n, size_k) = weights.dims3()?;
    assert!(
        size_k == size_k1,
        "input {:?} and weight {:?} last dim mismatch!",
        size_k1,
        size_k
    );

    let dtype = input.dtype();
    if dtype != DType::F16 && dtype != DType::BF16 {
        candle_core::bail!(
            "moe_gemm on Metal only supports f16/bf16 inputs, got {:?}",
            dtype
        );
    }

    let dev = input.device();
    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer = command_buffer.as_ref();

    let output = Tensor::zeros((size_m, size_n), dtype, dev)?;

    {
        let (input_s, input_l) = input.storage_and_layout();
        let input_s = match &*input_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("input must be a metal tensor"),
        };

        let (weights_s, weights_l) = weights.storage_and_layout();
        let weights_s = match &*weights_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("weights must be a metal tensor"),
        };

        let (sti_s, sti_l) = sorted_token_ids.storage_and_layout();
        let sti_s = match &*sti_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("sorted_token_ids must be a metal tensor"),
        };

        let (eid_s, eid_l) = experts_ids.storage_and_layout();
        let eid_s = match &*eid_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("experts_ids must be a metal tensor"),
        };

        let topk_weights_buf = if let Some(tw) = topk_weights {
            let (tw_s, tw_l) = tw.storage_and_layout();
            let tw_s = match &*tw_s {
                Storage::Metal(s) => s,
                _ => candle::bail!("topk_weights must be a metal tensor"),
            };
            Some((
                tw_s.buffer().clone(),
                tw_l.start_offset() * tw.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let (output_s, output_l) = output.storage_and_layout();
        let output_s = match &*output_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("output must be a metal tensor"),
        };

        let tw_ref = topk_weights_buf
            .as_ref()
            .map(|(buf, off)| (buf as &metal::Buffer, *off));

        metal_kernels::call_moe_gemm(
            metal_dev.device(),
            command_buffer,
            metal_kernels::Kernels::default(),
            dtype,
            input_s.buffer(),
            input_l.start_offset() * dtype.size_in_bytes(),
            weights_s.buffer(),
            weights_l.start_offset() * dtype.size_in_bytes(),
            sti_s.buffer(),
            sti_l.start_offset() * sorted_token_ids.dtype().size_in_bytes(),
            eid_s.buffer(),
            eid_l.start_offset() * experts_ids.dtype().size_in_bytes(),
            tw_ref,
            output_s.buffer(),
            output_l.start_offset() * dtype.size_in_bytes(),
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            is_prefill,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok(output)
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm is not implemented on this platform!")
}

#[cfg(feature = "cuda")]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::quantized::GgmlDType;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle_core::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }.w()?;
        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let (input_ptr, input_dtype) = match &*input {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            (*c.as_cuda_slice::<f16>()?.device_ptr() as *const c_void, 0)
                        } else {
                            (*c.as_cuda_slice::<bf16>()?.device_ptr() as *const c_void, 1)
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,               // [size_m or size_m/topk, size_k]
                    weight_ptr as *const u8, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream as i64,
                );
            } else {
                let (input, _) = input.storage_and_layout();
                let input = match &*input {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };

                // Use optimized small-M kernel for batch size < 8 (decode scenarios)
                if size_m <= 8 {
                    ffi::moe_gemm_gguf_small_m(
                        *input.device_ptr() as *const f32, // [size_m or size_m/topk, size_k]
                        weight_ptr as *const c_void,       // [num_experts, size_n, size_k]
                        *sorted_token_ids.device_ptr() as *const i32,
                        *experts_ids.device_ptr() as *const i32,
                        topk_weights_ptr,
                        *output.device_ptr() as *mut c_void, // [size_m, size_n]
                        num_experts as i32,
                        topk as i32,
                        size_m as i32,
                        size_n as i32,
                        size_k as i32,
                        gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                        stream as i64,
                    );
                } else {
                    ffi::moe_gemm_gguf(
                        *input.device_ptr() as *const f32, // [size_m or size_m/topk, size_k]
                        weight_ptr as *const c_void,       // [num_experts, size_n, size_k]
                        *sorted_token_ids.device_ptr() as *const i32,
                        *experts_ids.device_ptr() as *const i32,
                        topk_weights_ptr,
                        *output.device_ptr() as *mut c_void, // [size_m, size_n]
                        num_experts as i32,
                        topk as i32,
                        size_m as i32,
                        size_n as i32,
                        size_k as i32,
                        gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                        stream as i64,
                    );
                }
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle_core::bail!("moe_gemm_gguf only accept f16/bf16 inputs!")
        }
    }
}

#[cfg(feature = "metal")]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    use candle_core::DType;

    let _shape = weights.shape().dims3()?;
    let dequant = weights.dequantize_f16(input.device())?;

    let compute_dtype = if dtype == DType::F16 || dtype == DType::BF16 {
        dtype
    } else {
        DType::F16
    };

    let input_cast = if input.dtype() != compute_dtype {
        input.to_dtype(compute_dtype)?
    } else {
        input.clone()
    };

    let dequant = if dequant.dtype() != compute_dtype {
        dequant.to_dtype(compute_dtype)?
    } else {
        dequant
    };

    let result = moe_gemm(
        &input_cast,
        &dequant,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        topk,
        is_prefill,
    )?;

    if result.dtype() != DType::F32 {
        result.to_dtype(DType::F32)
    } else {
        Ok(result)
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle_core::DType,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_gguf is not implemented on this platform!")
}

#[cfg(test)]
mod tests {
    use super::{build_routed_rows_metadata, RoutedRowsMetadata};

    fn assert_metadata_eq(actual: RoutedRowsMetadata, expected: RoutedRowsMetadata) {
        assert_eq!(actual.sorted_token_ids, expected.sorted_token_ids);
        assert_eq!(actual.scatter_ids, expected.scatter_ids);
        assert_eq!(actual.expert_offsets, expected.expert_offsets);
        assert_eq!(actual.expert_counts, expected.expert_counts);
        assert_eq!(actual.sf_offsets, expected.sf_offsets);
        assert_eq!(actual.problem_sizes, expected.problem_sizes);
        assert_eq!(actual.total_sf_rows, expected.total_sf_rows);
        assert_eq!(actual.total_expanded, expected.total_expanded);
    }

    #[test]
    fn routed_rows_metadata_for_gate_up_input_uses_token_rows() {
        let actual =
            build_routed_rows_metadata(&[vec![2, 0], vec![1, 2]], 3, 64, 32, false).unwrap();
        let expected = RoutedRowsMetadata {
            sorted_token_ids: vec![0, 1, 0, 1],
            scatter_ids: vec![1, 2, 0, 3],
            expert_offsets: vec![0, 1, 2],
            expert_counts: vec![1, 1, 2],
            sf_offsets: vec![0, 128, 256],
            problem_sizes: vec![1, 64, 32, 1, 64, 32, 2, 64, 32],
            total_sf_rows: 384,
            total_expanded: 4,
        };
        assert_metadata_eq(actual, expected);
    }

    #[test]
    fn routed_rows_metadata_for_down_proj_input_uses_routed_rows() {
        let actual =
            build_routed_rows_metadata(&[vec![2, 0], vec![1, 2]], 3, 64, 32, true).unwrap();
        let expected = RoutedRowsMetadata {
            sorted_token_ids: vec![1, 2, 0, 3],
            scatter_ids: vec![1, 2, 0, 3],
            expert_offsets: vec![0, 1, 2],
            expert_counts: vec![1, 1, 2],
            sf_offsets: vec![0, 128, 256],
            problem_sizes: vec![1, 64, 32, 1, 64, 32, 2, 64, 32],
            total_sf_rows: 384,
            total_expanded: 4,
        };
        assert_metadata_eq(actual, expected);
    }
}
