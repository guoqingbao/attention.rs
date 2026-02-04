use candle_core::{DType, MetalStorage};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use once_cell::sync::OnceCell;
use std::sync::{OnceLock, RwLock};
use std::{collections::HashMap, ffi::c_void};

pub mod utils;
use utils::EncoderProvider;

#[derive(Debug)]
pub enum PagedAttentionDType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
}

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention.metallib"));
#[cfg(target_os = "ios")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention_ios.metallib"));

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("dtype mismatch, got {got:?}, expected {expected:?}")]
    DTypeMismatch { expected: Vec<DType>, got: DType },
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipelineState>;

#[derive(Debug)]
pub struct Kernels {
    pipelines: RwLock<Pipelines>,
}

pub(crate) static G_KERNEL: OnceCell<Kernels> = OnceCell::new();
pub(crate) static LIBRARY: OnceLock<Library> = OnceLock::new();

impl Kernels {
    pub fn default() -> &'static Kernels {
        G_KERNEL.get_or_init(Kernels::new)
    }

    pub fn new() -> Self {
        let pipelines = RwLock::new(Pipelines::new());
        Self { pipelines }
    }

    pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
        if let Some(lib) = LIBRARY.get() {
            Ok(lib.clone())
        } else {
            let source_data = KERNELS;
            let lib = {
                device.new_library_with_data(source_data).map_err(|e| {
                    MetalKernelError::LoadLibraryError(format!(
                        "Metal requires macosx > 13.0 or higher, cannot load candle metal library: {e}"
                    ))
                })?
            };
            Ok(LIBRARY.get_or_init(|| lib).clone())
        }
    }

    fn load_function(
        &self,
        device: &Device,
        name: String,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device)?
            .get_function(&name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        name: String,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(
                device,
                name.clone(),
                constants.as_ref().map(|c| c.function_constant_values()),
            )?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        name: String,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, name, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy_blocks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    key_cache: &Buffer,
    key_cache_offset: usize,
    value_cache: &Buffer,
    value_cache_offset: usize,
    block_mapping: &Buffer,
    block_mapping_offset: usize,
    num_pairs: u64,
    numel_per_block: u64,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "copy_blocks_float",
        DType::BF16 => "copy_blocks_bfloat16_t",
        DType::F16 => "copy_blocks_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (key_cache, key_cache_offset),
            (value_cache, value_cache_offset),
            (block_mapping, block_mapping_offset),
            numel_per_block
        )
    );

    let thread_groups_count = MTLSize {
        width: num_pairs,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: numel_per_block.min(1024),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reshape_and_cache(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    key: &Buffer,
    key_offset: usize,
    value: &Buffer,
    value_offset: usize,
    key_cache: &Buffer,
    key_cache_offset: usize,
    value_cache: &Buffer,
    value_cache_offset: usize,
    slot_mapping: &Buffer,
    slot_mapping_offset: usize,
    num_tokens: i32,
    num_heads: i32,
    head_size: i32,
    block_size: i32,
    x: i32,
    key_stride: i32,
    value_stride: i32,
    k_scale: Option<MetalStorage>,
    v_scale: Option<MetalStorage>,
) -> Result<(), MetalKernelError> {
    let quantized_cache = k_scale.is_some() && v_scale.is_some();
    let name = match ty {
        PagedAttentionDType::F32 => {
            if quantized_cache {
                "reshape_and_cache_float_uint8_t"
            } else {
                "reshape_and_cache_float_float"
            }
        }
        PagedAttentionDType::BF16 => {
            if quantized_cache {
                "reshape_and_cache_bfloat16_t_uint8_t"
            } else {
                "reshape_and_cache_bfloat16_t_bfloat16_t"
            }
        }
        PagedAttentionDType::F16 => {
            if quantized_cache {
                "reshape_and_cache_half_uint8_t"
            } else {
                "reshape_and_cache_half_half"
            }
        }
    };
    let pipeline = kernels.load_pipeline(device, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (key, key_offset),
            (value, value_offset),
            (key_cache, key_cache_offset),
            (value_cache, value_cache_offset),
            (slot_mapping, slot_mapping_offset),
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x,
            k_scale,
            v_scale
        )
    );

    let thread_groups_count = MTLSize {
        width: num_tokens as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: (num_heads * head_size).min(512) as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum Value {
    Bool(bool),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v1(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    context_lens: &Buffer,
    context_lens_offset: usize,
    alibi_storage_and_offset: Option<(MetalStorage, usize)>,
    output: &Buffer,
    num_kv_heads: i32,
    scale: f32,
    softcapping: f32,
    block_size: i32,
    max_context_len: i32,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    k_scale: Option<MetalStorage>,
    v_scale: Option<MetalStorage>,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: u64 = 256;
    const NUM_SIMD_LANES: u64 = 32;
    let quantized_cache = k_scale.is_some() && v_scale.is_some();

    let name = match ty {
        PagedAttentionDType::F32 => {
            if quantized_cache {
                "paged_attention_float_uint8_t"
            } else {
                "paged_attention_float_float"
            }
        }
        PagedAttentionDType::BF16 => {
            if quantized_cache {
                "paged_attention_bfloat16_t_uint8_t"
            } else {
                "paged_attention_bfloat16_t_bfloat16_t"
            }
        }
        PagedAttentionDType::F16 => {
            if quantized_cache {
                "paged_attention_half_uint8_t"
            } else {
                "paged_attention_half_half"
            }
        }
    };
    let mut name = name.to_string();
    name.push_str(&format!("_hs{head_size}"));
    name.push_str(&format!("_bs{block_size}"));
    name.push_str(&format!("_nt{NUM_THREADS}"));
    name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
    // v1 has no partition
    name.push_str(&format!("_ps{}", 0));

    // v1 has no partition.
    // Handle alibi
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* use_partitioning */ false)),
        (
            20,
            Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
        ),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

    let num_simds = NUM_THREADS / NUM_SIMD_LANES;
    let padded_max_context_len = ((max_context_len + block_size - 1) / block_size) * block_size;
    let logits_size = padded_max_context_len * std::mem::size_of::<f32>() as i32;
    let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
    let shared_mem_size = logits_size.max(outputs_size);
    encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

    encoder.set_buffer(2, Some(output), 0 as NSUInteger);
    encoder.set_buffer(3, Some(q), q_offset as NSUInteger);
    encoder.set_buffer(4, Some(k_cache), k_cache_offset as NSUInteger);
    encoder.set_buffer(5, Some(v_cache), v_cache_offset as NSUInteger);
    encoder.set_bytes(
        6,
        core::mem::size_of_val(&num_kv_heads) as u64,
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(
        7,
        core::mem::size_of_val(&scale) as u64,
        &scale as *const _ as *const c_void,
    );
    encoder.set_bytes(
        8,
        core::mem::size_of_val(&softcapping) as u64,
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_buffer(9, Some(block_tables), block_tables_offset as NSUInteger);
    encoder.set_buffer(10, Some(context_lens), context_lens_offset as NSUInteger);
    encoder.set_bytes(
        11,
        core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
        &max_num_blocks_per_seq as *const _ as *const c_void,
    );
    if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
        encoder.set_buffer(12, Some(alibi.buffer()), alibi_offset as NSUInteger);
    }
    encoder.set_bytes(
        13,
        core::mem::size_of_val(&q_stride) as u64,
        &q_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        14,
        core::mem::size_of_val(&kv_block_stride) as u64,
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        15,
        core::mem::size_of_val(&kv_head_stride) as u64,
        &kv_head_stride as *const _ as *const c_void,
    );

    if let Some(k_scale) = k_scale {
        encoder.set_buffer(16, Some(k_scale.buffer()), 0);
    }

    if let Some(v_scale) = v_scale {
        encoder.set_buffer(17, Some(v_scale.buffer()), 0);
    }

    let thread_groups_count = MTLSize {
        width: num_heads as u64,
        height: num_seqs as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: NUM_THREADS,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    exp_sums: &Buffer,
    max_logits: &Buffer,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    context_lens: &Buffer,
    context_lens_offset: usize,
    alibi_storage_and_offset: Option<(MetalStorage, usize)>,
    tmp_out: &Buffer,
    output: &Buffer,
    num_kv_heads: i32,
    scale: f32,
    softcapping: f32,
    block_size: i32,
    max_context_len: i32,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    k_scale: Option<MetalStorage>,
    v_scale: Option<MetalStorage>,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: u64 = 256;
    const PARTITION_SIZE: u64 = 512;
    const NUM_SIMD_LANES: u64 = 32;
    let quantized_cache = k_scale.is_some() && v_scale.is_some();
    // Initial paged attention kernel
    {
        let name = match ty {
            PagedAttentionDType::F32 => {
                if quantized_cache {
                    "paged_attention_float_uint8_t"
                } else {
                    "paged_attention_float_float"
                }
            }
            PagedAttentionDType::BF16 => {
                if quantized_cache {
                    "paged_attention_bfloat16_t_uint8_t"
                } else {
                    "paged_attention_bfloat16_t_bfloat16_t"
                }
            }
            PagedAttentionDType::F16 => {
                if quantized_cache {
                    "paged_attention_half_uint8_t"
                } else {
                    "paged_attention_half_half"
                }
            }
        };
        let mut name = name.to_string();
        name.push_str(&format!("_hs{head_size}"));
        name.push_str(&format!("_bs{block_size}"));
        name.push_str(&format!("_nt{NUM_THREADS}"));
        name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
        // v2 has partition.
        name.push_str(&format!("_ps{}", PARTITION_SIZE));

        // v2 has partition.
        // Handle alibi
        let constants = Some(ConstantValues::new(vec![
            (10, Value::Bool(/* use_partitioning */ true)),
            (
                20,
                Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
            ),
        ]));

        let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

        let num_simds = NUM_THREADS / NUM_SIMD_LANES;
        let max_num_partitions =
            (max_context_len + PARTITION_SIZE as i32 - 1) / PARTITION_SIZE as i32;
        let logits_size = PARTITION_SIZE as i32 * std::mem::size_of::<f32>() as i32;
        let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
        let shared_mem_size = logits_size.max(outputs_size);
        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        encoder.set_buffer(0, Some(exp_sums), 0 as NSUInteger);
        encoder.set_buffer(1, Some(max_logits), 0 as NSUInteger);
        encoder.set_buffer(2, Some(tmp_out), 0 as NSUInteger);
        encoder.set_buffer(3, Some(q), q_offset as NSUInteger);
        encoder.set_buffer(4, Some(k_cache), k_cache_offset as NSUInteger);
        encoder.set_buffer(5, Some(v_cache), v_cache_offset as NSUInteger);
        encoder.set_bytes(
            6,
            core::mem::size_of_val(&num_kv_heads) as u64,
            &num_kv_heads as *const _ as *const c_void,
        );
        encoder.set_bytes(
            7,
            core::mem::size_of_val(&scale) as u64,
            &scale as *const _ as *const c_void,
        );
        encoder.set_bytes(
            8,
            core::mem::size_of_val(&softcapping) as u64,
            &softcapping as *const _ as *const c_void,
        );
        encoder.set_buffer(9, Some(block_tables), block_tables_offset as NSUInteger);
        encoder.set_buffer(10, Some(context_lens), context_lens_offset as NSUInteger);
        encoder.set_bytes(
            11,
            core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
            &max_num_blocks_per_seq as *const _ as *const c_void,
        );
        if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
            encoder.set_buffer(12, Some(alibi.buffer()), alibi_offset as NSUInteger);
        }
        encoder.set_bytes(
            13,
            core::mem::size_of_val(&q_stride) as u64,
            &q_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            14,
            core::mem::size_of_val(&kv_block_stride) as u64,
            &kv_block_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            15,
            core::mem::size_of_val(&kv_head_stride) as u64,
            &kv_head_stride as *const _ as *const c_void,
        );

        if let Some(k_scale) = k_scale {
            encoder.set_buffer(16, Some(k_scale.buffer()), 0);
        }

        if let Some(v_scale) = v_scale {
            encoder.set_buffer(17, Some(v_scale.buffer()), 0);
        }

        let thread_groups_count = MTLSize {
            width: num_heads as u64,
            height: num_seqs as u64,
            depth: max_num_partitions as u64,
        };
        let thread_group_size = MTLSize {
            width: NUM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }

    // Paged attention reduce kernel
    {
        let name = match ty {
            PagedAttentionDType::F32 => {
                if quantized_cache {
                    "paged_attention_v2_reduce_float_uint8_t"
                } else {
                    "paged_attention_v2_reduce_float_float"
                }
            }
            PagedAttentionDType::BF16 => {
                if quantized_cache {
                    "paged_attention_v2_reduce_bfloat16_t_uint8_t"
                } else {
                    "paged_attention_v2_reduce_bfloat16_t_bfloat16_t"
                }
            }
            PagedAttentionDType::F16 => {
                if quantized_cache {
                    "paged_attention_v2_reduce_half_uint8_t"
                } else {
                    "paged_attention_v2_reduce_half_half"
                }
            }
        };
        let mut name = name.to_string();
        name.push_str(&format!("_hs{head_size}"));
        name.push_str(&format!("_nt{NUM_THREADS}"));
        name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
        name.push_str(&format!("_ps{}", PARTITION_SIZE));

        let pipeline = kernels.load_pipeline(device, name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

        let max_num_partitions =
            (max_context_len + PARTITION_SIZE as i32 - 1) / PARTITION_SIZE as i32;
        let reduce_shared_mem_size = 2 * max_num_partitions * std::mem::size_of::<f32>() as i32;
        encoder.set_threadgroup_memory_length(0, reduce_shared_mem_size as u64);

        encoder.set_buffer(0, Some(output), 0 as NSUInteger);
        encoder.set_buffer(1, Some(exp_sums), 0 as NSUInteger);
        encoder.set_buffer(2, Some(max_logits), 0 as NSUInteger);
        encoder.set_buffer(3, Some(tmp_out), 0 as NSUInteger);
        encoder.set_buffer(4, Some(context_lens), context_lens_offset as NSUInteger);
        encoder.set_bytes(
            5,
            core::mem::size_of_val(&max_num_partitions) as u64,
            &max_num_partitions as *const _ as *const c_void,
        );

        let thread_groups_count = MTLSize {
            width: num_heads as u64,
            height: num_seqs as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: NUM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }
    Ok(())
}

/// Launches the `chunked_prefill_paged_attention` Metal kernel.
///
/// This kernel is optimized for the prefill (prompt processing) stage of attention, where a batch of new tokens is processed.
/// The strategy is to assign one thread to each token in the query sequence. The dispatch grid is structured to parallelize work across
/// query heads, key-value heads, and chunks of tokens.
///
/// # Dispatch Logic
/// - **Threadgroup Size:** `(TOKEN_CHUNK_SIZE, 1, 1)`. Each threadgroup processes a chunk of `TOKEN_CHUNK_SIZE` tokens.
/// - **Grid Dimensions:** `(num_queries_per_kv, num_kv_heads, num_token_chunks)`.
///   - `width`: Parallelizes over the query heads that map to a single key-value head.
///   - `height`: Parallelizes over the key-value heads.
///   - `depth`: Parallelizes over the chunks of query tokens.
///
/// # Arguments
///
/// * `device` - The Metal device to execute the kernel on.
/// * `ep` - An `EncoderProvider` to get a command encoder.
/// * `kernels` - A struct for loading and caching Metal pipeline states.
/// * `ty` - The data type (`F16`, `BF16`, `F32`) of the tensors.
/// * `output` - The output buffer for the attention results. Shape: `[num_query_tokens, num_query_heads, head_size]`.
/// * `q` - The query tensor.
/// * `k_cache` - The paged key-cache.
/// * `v_cache` - The paged value-cache.
/// * `block_tables` - A tensor mapping logical sequence blocks to physical blocks in the cache. Shape: `[num_seqs, max_num_blocks_per_seq]`.
/// * `seq_lens` - A buffer containing the full context length of each sequence.
/// * `query_start_len` - A buffer indicating the start token index for each sequence in the flattened query tensor.
/// * `alibi_slopes` - Optional buffer containing ALiBi slopes for positional bias.
/// * `k_scale` - Optional buffer for per-head FP8 K scales.
/// * `v_scale` - Optional buffer for per-head FP8 V scales.
/// * `sinks` - Optional buffer for sink attention.
/// * `num_kv_heads` - The number of key-value heads (for Grouped-Query Attention).
/// * `scale` - The softmax scaling factor (typically `1.0 / sqrt(head_size)`).
/// * `block_table_stride` - The stride of the `block_tables` tensor (i.e., `max_num_blocks_per_seq`).
/// * `num_seqs` - The number of sequences in the batch.
/// * `num_query_heads` - The total number of query heads.
/// * `num_query_tokens` - The total number of tokens being processed in this prefill run.
/// * `head_size` - The dimension of each attention head.
/// * `block_size` - The number of tokens per block in the KV cache.
/// * `softcapping` - Softcapping value for the tanh activation on attention scores.
/// * `o_stride_tokens` - The stride of the output tensor's first dimension.
/// * `sliding_window` - The sliding window size for attention, if applicable.
/// * `total_num_blocks` - The total number of physical blocks in the KV cache.
/// * `kv_block_stride` - The stride between blocks in the KV cache.
/// * `kv_head_stride` - The stride between heads in the KV cache.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    // Buffers and Offsets
    output: &Buffer,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    seq_lens: &Buffer, // Equivalent to `context_lens` in the v1 kernel
    seq_lens_offset: usize,
    query_start_len: &Buffer,
    query_start_len_offset: usize,
    alibi_slopes: Option<(MetalStorage, usize)>,
    k_scale: Option<MetalStorage>,
    v_scale: Option<MetalStorage>,
    sinks: Option<(MetalStorage, usize)>,
    // Scalar Parameters
    num_kv_heads: i32,
    scale: f32,              // sm_scale
    block_table_stride: i32, // max_num_blocks_per_seq
    num_seqs: i32,
    num_query_heads: i32,
    num_query_tokens: i32,
    head_size: i32,
    block_size: i32,
    softcapping: f32,
    o_stride_tokens: i32,
    sliding_window: i32,
    total_num_blocks: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
) -> Result<(), MetalKernelError> {
    // This value must match the `token_chunk_size` used in the .metal instantiation macros
    const TOKEN_CHUNK_SIZE: u64 = 64;
    let quantized_cache = k_scale.is_some() && v_scale.is_some();

    // 1. Construct the unique kernel name from its template parameters.
    let type_name = match ty {
        PagedAttentionDType::F32 => "float",
        PagedAttentionDType::BF16 => "bfloat16_t",
        PagedAttentionDType::F16 => "half",
    };
    let name = if quantized_cache {
        format!(
            "chunked_prefill_{}_uint8_t_hs{}_bs{}_tcs{}",
            type_name, head_size, block_size, TOKEN_CHUNK_SIZE
        )
    } else {
        format!(
            "chunked_prefill_{}_hs{}_bs{}_tcs{}",
            type_name, head_size, block_size, TOKEN_CHUNK_SIZE
        )
    };

    // 2. Load the pipeline. The prefill kernel does not use function constants.
    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // NOTE: Unlike the v1 kernel, the chunked prefill kernel is designed to use
    // registers and local arrays instead of threadgroup memory, so we do not
    // call `set_threadgroup_memory_length`.

    // 3. Set all kernel arguments, matching the `[[buffer(n)]]` indices.
    encoder.set_buffer(0, Some(output), 0);
    encoder.set_buffer(1, Some(q), q_offset as NSUInteger);
    encoder.set_buffer(2, Some(k_cache), k_cache_offset as NSUInteger);
    encoder.set_buffer(3, Some(v_cache), v_cache_offset as NSUInteger);
    encoder.set_bytes(
        4,
        size_of_val(&num_kv_heads),
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(5, size_of_val(&scale), &scale as *const _ as *const c_void);
    encoder.set_buffer(6, Some(block_tables), block_tables_offset as NSUInteger);
    encoder.set_buffer(7, Some(seq_lens), seq_lens_offset as NSUInteger);
    encoder.set_bytes(
        8,
        size_of_val(&block_table_stride),
        &block_table_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        9,
        size_of_val(&num_seqs),
        &num_seqs as *const _ as *const c_void,
    );
    encoder.set_bytes(
        10,
        size_of_val(&num_query_heads),
        &num_query_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(
        11,
        size_of_val(&num_query_tokens),
        &num_query_tokens as *const _ as *const c_void,
    );
    encoder.set_bytes(
        12,
        size_of_val(&softcapping),
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_bytes(
        13,
        size_of_val(&o_stride_tokens),
        &o_stride_tokens as *const _ as *const c_void,
    );
    encoder.set_buffer(
        14,
        Some(query_start_len),
        query_start_len_offset as NSUInteger,
    );
    if let Some((slop, offset)) = alibi_slopes {
        encoder.set_buffer(15, Some(slop.buffer()), offset as NSUInteger);
    }
    if let Some(k_scale) = k_scale {
        encoder.set_buffer(16, Some(k_scale.buffer()), 0);
    }
    if let Some(v_scale) = v_scale {
        encoder.set_buffer(17, Some(v_scale.buffer()), 0);
    }
    if let Some((sk, offset)) = sinks {
        encoder.set_buffer(18, Some(sk.buffer()), offset as NSUInteger);
    }
    encoder.set_bytes(
        19,
        size_of_val(&sliding_window),
        &sliding_window as *const _ as *const c_void,
    );
    encoder.set_bytes(
        20,
        size_of_val(&total_num_blocks),
        &total_num_blocks as *const _ as *const c_void,
    );
    encoder.set_bytes(
        21,
        size_of_val(&kv_block_stride),
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        22,
        size_of_val(&kv_head_stride),
        &kv_head_stride as *const _ as *const c_void,
    );

    // 4. Calculate grid and threadgroup dimensions, matching the CUDA launch config.
    // CUDA: dim3 block(TOKEN_CHUNK_SIZE);
    let thread_group_size = MTLSize {
        width: TOKEN_CHUNK_SIZE,
        height: 1,
        depth: 1,
    };

    // CUDA: dim3 grid(num_queries_per_kv, num_kv_heads, num_token_chunks);
    let num_queries_per_kv = (num_query_heads / num_kv_heads) as u64;
    let num_token_chunks = (num_query_tokens as u64 + TOKEN_CHUNK_SIZE - 1) / TOKEN_CHUNK_SIZE;
    let thread_groups_count = MTLSize {
        width: num_queries_per_kv,
        height: num_kv_heads as u64,
        depth: num_token_chunks,
    };

    // 5. Dispatch the kernel.
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

// Helper function to get size of a value for set_bytes
fn size_of_val<T>(val: &T) -> u64 {
    core::mem::size_of_val(val) as u64
}

#[allow(clippy::too_many_arguments)]
pub fn call_causal_mask(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    out: &Buffer,
    tgt_len: i32,
    sliding_window: i32,
) -> Result<(), MetalKernelError> {
    // Determine the kernel name based on the type
    let name = match ty {
        PagedAttentionDType::F32 => "causal_mask_float",
        PagedAttentionDType::BF16 => "causal_mask_bfloat16_t",
        PagedAttentionDType::F16 => "causal_mask_half",
    };

    // Load the compute pipeline for the selected type
    let pipeline = kernels.load_pipeline(device, name.to_string())?;

    // Get the encoder and cast it to the appropriate type
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Set the parameters for the kernel
    set_params!(
        encoder,
        (
            (out, 0),       // Output buffer
            tgt_len,        // Target length (size)
            sliding_window  // Sliding window size
        )
    );

    // Set up the number of thread groups and threads per group
    let thread_groups_count = MTLSize {
        width: tgt_len as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 256, // Use a reasonable number of threads per threadgroup
        height: 1,
        depth: 1,
    };

    // Dispatch the kernel
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);

    // Commit the encoder to launch the kernel
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn call_update_scales_per_head(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    k: &Buffer,
    v: &Buffer,
    num_tokens: i64,
    num_heads: i32,
    head_dim: i32,
    k_scales: &Buffer,
    v_scales: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        PagedAttentionDType::F32 => "compute_and_update_scales_per_head_float",
        PagedAttentionDType::BF16 => "compute_and_update_scales_per_head_bfloat16_t",
        PagedAttentionDType::F16 => "compute_and_update_scales_per_head_half",
    };

    let pipeline = kernels.load_pipeline(device, name.to_string())?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (k, 0),
            (v, 0),
            num_tokens,
            num_heads,
            head_dim,
            (k_scales, 0),
            (v_scales, 0)
        )
    );

    let threads_per_threadgroup = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let thread_groups_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Fused Rotary Position Embedding with position selection
///
/// This kernel fuses index_select + RoPE into a single kernel.
/// Supports GQA (different head counts for Q and K).
///
/// # Arguments
/// * `q` - Query buffer [batch, num_q_heads, seq_len, head_dim]
/// * `k` - Key buffer [batch, num_kv_heads, seq_len, head_dim]
/// * `cos` - Full cosine table [max_seq_len, head_dim/2]
/// * `sin` - Full sine table [max_seq_len, head_dim/2]
/// * `positions` - Position indices [seq_len] (i64)
/// * `q_bh` - batch * num_q_heads
/// * `k_bh` - batch * num_kv_heads
/// * `seq_len` - sequence length
/// * `d` - head_dim
/// * `is_interleaved` - if true, use interleaved RoPE layout
#[allow(clippy::too_many_arguments)]
pub fn call_fused_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    positions: &Buffer,
    positions_offset: usize,
    q_bh: u32,
    k_bh: u32,
    seq_len: u32,
    d: u32,
    is_interleaved: bool,
) -> Result<(), MetalKernelError> {
    let type_name = match ty {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    let name = if is_interleaved {
        format!("fused_rope_i_{}", type_name)
    } else {
        format!("fused_rope_{}", type_name)
    };

    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (q, q_offset),
            (k, k_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (positions, positions_offset),
            q_bh,
            k_bh,
            seq_len,
            d
        )
    );

    // Calculate total number of pairs
    let half_d = d / 2;
    let total_pairs = ((q_bh + k_bh) * seq_len * half_d) as u64;

    // Dispatch with 256 threads per threadgroup
    let threads_per_threadgroup = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let thread_groups_count = MTLSize {
        width: (total_pairs + 255) / 256,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_fp8_matmul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    input: &Buffer,
    input_offset: usize,
    weight: &Buffer,
    weight_offset: usize,
    weight_scale: &Buffer,
    weight_scale_offset: usize,
    output: &Buffer,
    output_offset: usize,
    m: i32,
    n: i32,
    k: i32,
    scale_row_stride: i32,
    block_size_y: i32,
    block_size_x: i32,
) -> Result<(), MetalKernelError> {
    let is_gemv = m <= 8;
    let name = match ty {
        DType::F16 => {
            if is_gemv {
                "fp8_gemv_half"
            } else if m <= 16 {
                "fp8_matmul_half_16_32_32"
            } else {
                "fp8_matmul_half_32_32_32"
            }
        }
        DType::BF16 => {
            if is_gemv {
                "fp8_gemv_bfloat16"
            } else if m <= 16 {
                "fp8_matmul_bfloat16_16_32_32"
            } else {
                "fp8_matmul_bfloat16_32_32_32"
            }
        }
        _ => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
                got: ty,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (input, input_offset),
            (weight, weight_offset),
            (weight_scale, weight_scale_offset),
            (output, output_offset),
            m,
            n,
            k,
            scale_row_stride,
            block_size_y,
            block_size_x
        )
    );

    if is_gemv {
        // Grid: (N * 32, M, 1)
        let thread_group_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let thread_groups_count = MTLSize {
            width: n as u64, // (N * 32) / 32 = N
            height: m as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    } else {
        let thread_group_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let m_block = if m <= 16 { 16 } else { 32 };
        let thread_groups_count = MTLSize {
            width: (n as u64 + 31) / 32,
            height: (m as u64 + m_block - 1) / m_block,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }
    Ok(())
}
