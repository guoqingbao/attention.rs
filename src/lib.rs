#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("Enable exactly one backend feature: `cuda` or `metal`, not both.");

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!("Enable exactly one backend feature: `cuda` or `metal`.");

#[cfg(all(feature = "flashinfer", feature = "flashattn"))]
compile_error!("Features `flashinfer` and `flashattn` are mutually exclusive. Enable only one.");

pub mod moe;
pub mod paged_attention;
pub mod scale_update;
#[cfg(feature = "cuda")]
pub mod workspace;
use candle_core::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
pub use paged_attention::convert_to_fp8;
use paged_attention::{paged_attention, reshape_and_cache};
use scale_update::kv_scale_update;
pub mod fused_rope;
pub mod mask;
#[cfg(feature = "cuda")]
pub mod sampler;
pub mod sort;
pub mod topk;
#[cfg(feature = "cuda")]
pub use kernels;
#[cfg(feature = "metal")]
pub use metal_kernels;
pub mod cache;
#[cfg(feature = "cuda")]
pub mod cuda_utils;
pub mod fp8_linear;
pub mod gdn;
pub mod mamba_cache;
pub mod mla;
pub mod mxfp4_linear;
pub mod nvfp4_linear;
pub mod ops;
pub mod silu_and_mul;
pub mod swiglu;

#[cfg(feature = "flash")]
pub mod flash;

#[cfg(feature = "flashinfer")]
pub mod flashinfer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboquantMode {
    Turbo8,
    Turbo4,
    Turbo3,
}

pub struct TurboquantLayerCache {
    pub k_absmax: Option<Tensor>,
    pub k_quant: Option<Tensor>,
    pub v_absmax: Tensor,
    pub v_quant: Tensor,
}

static TURBOQUANT_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<TurboquantGlobalCache>>> =
    std::sync::OnceLock::new();

pub struct TurboquantGlobalCache {
    pub mode: TurboquantMode,
    pub layers: Vec<TurboquantLayerCache>,
    pub block_size: usize,
}

pub fn init_turboquant_cache(
    mode: TurboquantMode,
    layers: Vec<TurboquantLayerCache>,
    block_size: usize,
) {
    let cache = TURBOQUANT_CACHE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cache.lock().unwrap();
    *guard = Some(TurboquantGlobalCache {
        mode,
        layers,
        block_size,
    });
}

pub fn has_flashinfer_fp8_e4m3() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe { kernels::ffi::has_flashinfer_fp8_e4m3() }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn get_turboquant_mode() -> Option<TurboquantMode> {
    TURBOQUANT_CACHE
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|g| g.as_ref().map(|c| c.mode))
}

pub fn get_turboquant_block_size() -> usize {
    TURBOQUANT_CACHE
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|g| g.as_ref().map(|c| c.block_size))
        .unwrap_or(16)
}

pub fn with_turboquant_layer<F, R>(layer_idx: usize, f: F) -> Option<R>
where
    F: FnOnce(&TurboquantLayerCache, TurboquantMode) -> R,
{
    TURBOQUANT_CACHE
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|g| {
            g.as_ref()
                .and_then(|c| c.layers.get(layer_idx).map(|l| f(l, c.mode)))
        })
}

#[cfg(feature = "trtllm")]
pub mod trtllm_cubin_loader;

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};
pub struct FlashInferMetadata {
    pub indptr: Tensor,
    pub indptr_host: Vec<u32>,
    pub indices: Tensor,
    pub last_len: Tensor,
    pub last_len_host: Option<Vec<u32>>,
    pub kv_len_arr_host: Option<Vec<u32>>,
    pub total_num_rows: Option<u32>,
    pub batch_indices: Option<Tensor>,
    pub positions: Option<Tensor>,
    pub use_cuda_graph: bool,
    pub decode_plan_info: Option<Vec<i64>>,
    pub prefill_plan_info: Option<Vec<i64>>,
    pub mla_decode_plan_info: Option<Vec<i64>>,
    pub mla_prefill_plan_info: Option<Vec<i64>>,
}

pub struct InputMetadata {
    pub is_prefill: bool,
    pub is_mla: bool,
    pub sequence_ids: Option<Vec<usize>>,
    pub mamba_slot_mapping: Option<Tensor>,
    pub slot_mapping: Tensor,
    pub block_tables: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_k: Option<Tensor>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub max_context_len: usize,
    pub seqlens: Option<Vec<u32>>,
    pub flashinfer_metadata: Option<FlashInferMetadata>,
}

#[allow(dead_code)]
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    alibi_slopes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    kv_updated_times: AtomicI32,
    #[cfg(feature = "flash")]
    flash_splitk_workspace: std::sync::OnceLock<Tensor>,
    layer_idx: usize,
}

static PAGED_ATTENTION_LAYER_COUNTER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

static TQ_DECODE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn reset_paged_attention_layer_counter() {
    PAGED_ATTENTION_LAYER_COUNTER.store(0, Ordering::SeqCst);
    TQ_DECODE_LOGGED.store(false, Ordering::SeqCst);
}

impl PagedAttention {
    fn batch_major_qkv(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, usize, usize, usize)> {
        match (query.dims().len(), key.dims().len(), value.dims().len()) {
            (4, 4, 4) => {
                let (_, attention_heads, seq_len, head_size) = query.shape().dims4()?;
                let (_, key_value_heads, key_seq_len, key_head_size) = key.shape().dims4()?;
                let (_, value_heads, value_seq_len, value_head_size) = value.shape().dims4()?;
                if key_seq_len != seq_len
                    || value_seq_len != seq_len
                    || key_head_size != head_size
                    || value_head_size != head_size
                    || value_heads != key_value_heads
                {
                    candle_core::bail!(
                        "Q/K/V layout mismatch, got Q {:?}, K {:?}, V {:?}",
                        query.shape(),
                        key.shape(),
                        value.shape()
                    );
                }
                Ok((
                    query.clone(),
                    key.clone(),
                    value.clone(),
                    attention_heads,
                    key_value_heads,
                    head_size,
                ))
            }
            (3, 3, 3) => {
                let (seq_len, attention_heads, head_size) = query.shape().dims3()?;
                let (key_seq_len, key_value_heads, key_head_size) = key.shape().dims3()?;
                let (value_seq_len, value_heads, value_head_size) = value.shape().dims3()?;
                if key_seq_len != seq_len
                    || value_seq_len != seq_len
                    || key_head_size != head_size
                    || value_head_size != head_size
                    || value_heads != key_value_heads
                {
                    candle_core::bail!(
                        "packed Q/K/V layout mismatch, got Q {:?}, K {:?}, V {:?}",
                        query.shape(),
                        key.shape(),
                        value.shape()
                    );
                }
                Ok((
                    query.transpose(0, 1)?.unsqueeze(0)?,
                    key.transpose(0, 1)?.unsqueeze(0)?,
                    value.transpose(0, 1)?.unsqueeze(0)?,
                    attention_heads,
                    key_value_heads,
                    head_size,
                ))
            }
            _ => candle_core::bail!(
                "paged attention expects 3D packed or 4D batch-major Q/K/V, got Q {:?}, K {:?}, V {:?}",
                query.shape(),
                key.shape(),
                value.shape()
            ),
        }
    }

    #[cfg(any(feature = "flash", feature = "flashattn", feature = "flashinfer"))]
    fn packed_qkv(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, usize, usize, usize)> {
        match (query.dims().len(), key.dims().len(), value.dims().len()) {
            (4, 4, 4) => {
                let (_, attention_heads, _, head_size) = query.shape().dims4()?;
                let (_, key_value_heads, _, _) = key.shape().dims4()?;
                let query = query
                    .transpose(1, 2)?
                    .reshape(((), attention_heads, head_size))?;
                let key = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let value = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                Ok((query, key, value, attention_heads, key_value_heads, head_size))
            }
            (3, 3, 3) => {
                let (_, attention_heads, head_size) = query.shape().dims3()?;
                let (_, key_value_heads, key_head_size) = key.shape().dims3()?;
                let (_, value_heads, value_head_size) = value.shape().dims3()?;
                if key_head_size != head_size || value_head_size != head_size {
                    candle_core::bail!(
                        "packed Q/K/V head_dim mismatch, got Q {:?}, K {:?}, V {:?}",
                        query.shape(),
                        key.shape(),
                        value.shape()
                    );
                }
                if value_heads != key_value_heads {
                    candle_core::bail!(
                        "packed K/V head count mismatch, got K {:?}, V {:?}",
                        key.shape(),
                        value.shape()
                    );
                }
                Ok((
                    query.clone(),
                    key.clone(),
                    value.clone(),
                    attention_heads,
                    key_value_heads,
                    head_size,
                ))
            }
            _ => candle_core::bail!(
                "flash attention expects 3D packed or 4D batch-major Q/K/V, got Q {:?}, K {:?}, V {:?}",
                query.shape(),
                key.shape(),
                value.shape()
            ),
        }
    }

    fn maybe_update_kv_scales(&self, key: &Tensor, value: &Tensor) -> Result<()> {
        if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {
            if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION {
                kv_scale_update(key, value, k_scale, v_scale)?;
                self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: Device,
        alibi_slopes: Option<Vec<f64>>,
        fp8_kvcache: bool,
    ) -> Result<Self> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, &device)?)
        } else {
            None
        };
        let layer_idx = PAGED_ATTENTION_LAYER_COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            alibi_slopes,
            k_scale: if fp8_kvcache {
                Some(Tensor::ones(
                    (num_key_value_heads,),
                    candle_core::DType::F32,
                    &device,
                )?)
            } else {
                None
            },
            v_scale: if fp8_kvcache {
                Some(Tensor::ones(
                    (num_key_value_heads,),
                    candle_core::DType::F32,
                    &device,
                )?)
            } else {
                None
            },
            kv_updated_times: AtomicI32::new(0),
            #[cfg(feature = "flash")]
            flash_splitk_workspace: std::sync::OnceLock::new(),
            layer_idx,
        })
    }

    #[allow(unused_variables)]
    pub fn sdp_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let (query, key, value, attention_heads, key_value_heads, head_size) =
            Self::batch_major_qkv(query, key, value)?;
        fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
            if n_rep == 1 {
                Ok(x)
            } else {
                let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
                Tensor::cat(&vec![&x; n_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * n_rep,
                    seq_len,
                    head_dim,
                ))
            }
        }
        let indices = &input_metadata
            .cu_seqlens_q
            .as_ref()
            .unwrap()
            .to_vec1::<u32>()?[1..];
        let seqlens: Vec<_> = indices.iter().map(|x| x).collect();

        let mut vec_attn = Vec::new();
        let mut start = 0usize;
        //chunked attention for each sequence
        for (i, seqlen) in seqlens.iter().enumerate() {
            let seq_len = (**seqlen as usize - start) as usize;
            let chunk_size = 1024;
            let mut attn_chunks = vec![];

            let query_seq = query.narrow(2, start, seq_len)?.contiguous()?;
            let key_seq = key.narrow(2, start, seq_len)?.contiguous()?;
            let value_seq = value.narrow(2, start, seq_len)?.contiguous()?;

            let key_seq = if key_value_heads != attention_heads {
                repeat_kv(key_seq, attention_heads / key_value_heads)?
            } else {
                key_seq
            };

            let value_seq = if key_value_heads != attention_heads {
                repeat_kv(value_seq, attention_heads / key_value_heads)?
            } else {
                value_seq
            };

            let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

            for c in 0..num_chunks {
                let offset = c * chunk_size;
                let len = chunk_size.min(seq_len - offset);
                //chunk at query is correct for the following
                let q_chunk = query_seq.narrow(2, offset, len)?.contiguous()?;
                let mut att = (q_chunk.matmul(&key_seq.t()?)? * f64::from(self.scale))?;

                if let Some(sc) = softcapping {
                    att = ((att / sc)?.tanh()? * sc)?;
                }

                if let Some(mask) = &attention_mask {
                    let q_chunk_mask = mask[i].narrow(2, offset, len)?;
                    att = att.broadcast_add(&q_chunk_mask)?;
                }

                att = candle_nn::ops::softmax_last_dim(&att.to_dtype(candle_core::DType::F32)?)?
                    .to_dtype(att.dtype())?;

                let att_chunk = att.matmul(&value_seq)?;
                attn_chunks.push(att_chunk);
            }

            let att = Tensor::cat(&attn_chunks, 2)?.contiguous()?;
            vec_attn.push(att);

            start = **seqlen as usize;
        }
        Tensor::cat(&vec_attn, 2)?.contiguous()?.transpose(1, 2)
    }

    #[cfg(feature = "flashattn")]
    pub fn flash_var_len(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        if self.sliding_window.is_some() {
            flashattn_rs::flash_attn_varlen_windowed_softcap(
                query,
                key,
                value,
                input_metadata.cu_seqlens_q.as_ref().unwrap(),
                input_metadata.cu_seqlens_k.as_ref().unwrap(),
                &input_metadata.block_tables,
                input_metadata.max_seqlen_q,
                input_metadata.max_seqlen_k,
                self.scale as f32,
                Some(softcapping.unwrap_or(0.0f64) as f32),
                self.sliding_window,
                Some(0),
            )
        } else {
            flashattn_rs::flash_attn_varlen_softcap(
                query,
                key,
                value,
                input_metadata.cu_seqlens_q.as_ref().unwrap(),
                input_metadata.cu_seqlens_k.as_ref().unwrap(),
                &input_metadata.block_tables,
                input_metadata.max_seqlen_q,
                input_metadata.max_seqlen_k,
                self.scale as f32,
                Some(softcapping.unwrap_or(0.0f64) as f32),
                true,
            )
        }
    }

    #[cfg(feature = "flashattn")]
    pub fn flash_forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_cache: Option<Tensor>,
        value_cache: Option<Tensor>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let (query, key, value, _attention_heads, _key_value_heads, _head_size) =
            Self::packed_qkv(query, key, value)?;
        let slot_mapping = input_metadata.slot_mapping.flatten_all()?;
        let softcap = Some(softcapping.unwrap_or(0.0f64) as f32);
        let window_size_right = self.sliding_window.map(|_| 0);

        let is_fp8_kv = self.k_scale.is_some();

        // FA3 native FP8: skip dynamic AMAX scale updates, keep K/V scales at 1.0.
        // FA3 kernel only applies q/k descales in the softcap path; with softcap=0
        // (typical for most models), non-1.0 scales would cause incorrect softmax
        // temperature. Scale=1.0 matches vLLM's approach for uncalibrated models.
        if !is_fp8_kv {
            self.maybe_update_kv_scales(&key, &value)?;
        }

        reshape_and_cache(
            &key,
            &value,
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            &slot_mapping,
        )?;

        if input_metadata.is_prefill && input_metadata.block_tables.is_none() {
            // prefill without kvcache
            return self.flash_var_len(&query, &key, &value, input_metadata, softcapping);
        }

        let block_tables = input_metadata.block_tables.as_ref().unwrap();
        let context_lens = input_metadata.context_lens.as_ref().unwrap();

        if input_metadata.is_prefill {
            #[cfg(feature = "cuda")]
            if is_fp8_kv {
                let (q_fp8, _q_descale) = convert_to_fp8(&query, Some(1.0))?;
                return flashattn_rs::flash_attn_with_kvcache_advanced(
                    &q_fp8,
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                    context_lens,
                    block_tables,
                    input_metadata.cu_seqlens_q.as_ref(),
                    Some(input_metadata.max_seqlen_q),
                    self.scale as f32,
                    true,
                    self.sliding_window,
                    window_size_right,
                    None,
                    softcap,
                    0,
                    None,
                    None,
                    None,
                    None,
                );
            }
            return flashattn_rs::flash_attn_with_kvcache_advanced(
                &query,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                context_lens,
                block_tables,
                input_metadata.cu_seqlens_q.as_ref(),
                Some(input_metadata.max_seqlen_q),
                self.scale as f32,
                true,
                self.sliding_window,
                window_size_right,
                None,
                softcap,
                0,
                None,
                None,
                None,
                None,
            );
        }

        #[cfg(feature = "cuda")]
        if is_fp8_kv {
            let (q_fp8, _q_descale) = convert_to_fp8(&query.unsqueeze(1)?, Some(1.0))?;
            return flashattn_rs::flash_attn_with_kvcache_advanced(
                &q_fp8,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                context_lens,
                block_tables,
                None,
                None,
                self.scale as f32,
                false,
                self.sliding_window,
                window_size_right,
                None,
                softcap,
                0,
                None,
                None,
                None,
                None,
            );
        }

        flashattn_rs::flash_attn_with_kvcache_advanced(
            &query.unsqueeze(1)?,
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            context_lens,
            block_tables,
            None,
            None,
            self.scale as f32,
            false,
            self.sliding_window,
            window_size_right,
            None,
            softcap,
            0,
            None,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unreachable_code)]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        key_cache: Option<Tensor>,
        value_cache: Option<Tensor>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        // head_dim > 256: FlashAttn/FlashInfer don't support it.
        // TurboQuant: only native flash path supports turbo KV cache.
        // Both cases force use of native flash path below.

        #[cfg(feature = "flash")]
        let force_native_flash =
            self.head_dim > 256 || input_metadata.flashinfer_metadata.is_none();
        #[cfg(not(feature = "flash"))]
        let force_native_flash = false;

        #[cfg(feature = "flashattn")]
        let skip_flashattn = {
            let tq = get_turboquant_mode().is_some();
            let fp8_on_non_sm90 = if self.k_scale.is_some() {
                #[cfg(feature = "cuda")]
                {
                    let sm = query
                        .device()
                        .as_cuda_device()
                        .ok()
                        .and_then(|d| crate::cuda_utils::sm_version(d))
                        .unwrap_or(0);
                    sm != 90 // FP8 kvcache only works on sm90
                }
                #[cfg(not(feature = "cuda"))]
                {
                    true
                }
            } else {
                false
            };
            self.head_dim > 256 || tq || fp8_on_non_sm90
        };

        #[cfg(feature = "flashinfer")]
        if !force_native_flash {
            if let Some(fm) = input_metadata.flashinfer_metadata.as_ref() {
                let (query, key, value, attention_heads, key_value_heads, head_size) =
                    Self::packed_qkv(query, key, value)?;
                let group_size = attention_heads / key_value_heads;
                let flashinfer_prefill_group_supported =
                    matches!(group_size, 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 32 | 64);
                let flashinfer_decode_group_supported =
                    flashinfer_prefill_group_supported && !(group_size == 64 && head_size > 128);
                let flashinfer_group_supported = if input_metadata.is_prefill {
                    flashinfer_prefill_group_supported
                } else {
                    flashinfer_decode_group_supported
                };

                if flashinfer_group_supported {
                    // FlashInfer FA2 path: skip per-head scale updates.
                    // With scale=1.0, append_kv_cache does raw BF16->FP8 cast,
                    // and the FA2 attention kernel does raw FP8->float cast — both match.
                    // The SM90 Hopper kernel handles per-head scales natively via
                    // its additional_params, so 1.0 scales are also fine there.

                    if let (Some(kc), Some(vc)) = (key_cache.as_ref(), value_cache.as_ref()) {
                        crate::flashinfer::append_kv_cache(
                            &key,
                            &value,
                            kc,
                            vc,
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            &fm.indices,
                            &fm.indptr,
                            &fm.last_len,
                            fm.batch_indices.as_ref(),
                            fm.positions.as_ref(),
                        )?;
                    }

                    let block_size = if let Some(kc) = key_cache.as_ref() {
                        kc.dim(1)?
                    } else {
                        16
                    };

                    if input_metadata.is_prefill {
                        let plan_info = fm.prefill_plan_info.as_ref().ok_or_else(|| {
                            candle_core::Error::msg(
                                "flashinfer prefill requires prefill_plan_info (plan+run path)",
                            )
                        })?;
                        return crate::flashinfer::prefill_with_plan(
                            &query,
                            key_cache.as_ref().unwrap(),
                            value_cache.as_ref().unwrap(),
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            &fm.indices,
                            &fm.indptr,
                            &fm.last_len,
                            input_metadata.cu_seqlens_q.as_ref().unwrap(),
                            fm.total_num_rows.unwrap(),
                            block_size,
                            attention_heads,
                            key_value_heads,
                            head_size,
                            self.scale as f32,
                            Some(self.sliding_window.unwrap_or(0) as i32),
                            Some(softcapping.unwrap_or(0.0f64) as f32),
                            plan_info,
                        );
                    } else {
                        let plan_info = fm.decode_plan_info.as_ref().ok_or_else(|| {
                            candle_core::Error::msg(
                                "flashinfer decode requires decode_plan_info (plan+run path)",
                            )
                        })?;
                        return crate::flashinfer::decode_with_plan(
                            &query,
                            key_cache.as_ref().unwrap(),
                            value_cache.as_ref().unwrap(),
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            &fm.indices,
                            &fm.indptr,
                            &fm.last_len,
                            block_size,
                            attention_heads,
                            key_value_heads,
                            head_size,
                            self.scale as f32,
                            plan_info,
                            fm.use_cuda_graph,
                            Some(self.sliding_window.unwrap_or(0) as i32),
                            Some(softcapping.unwrap_or(0.0f64) as f32),
                        );
                    }
                }

                if !flashinfer_group_supported {
                    tracing::warn!(
                    group_size = group_size,
                    head_size = head_size,
                    is_prefill = input_metadata.is_prefill,
                    "flashinfer disabled for this layer: unsupported gqa/head_dim combination, falling back"
                );
                }
            }
        } // end if !force_native_flash (flashinfer)

        #[cfg(feature = "flashattn")]
        if !skip_flashattn {
            return self.flash_forward(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata,
                softcapping,
            );
        }

        #[cfg(feature = "flash")]
        {
            let (query_p, key_p, value_p, attention_heads_p, key_value_heads_p, head_size_p) =
                Self::packed_qkv(query, key, value)?;

            let slot_mapping = input_metadata.slot_mapping.flatten_all()?;

            let tq_mode = get_turboquant_mode();
            let tq_bs = get_turboquant_block_size();

            let tq_uses_std_cache = matches!(tq_mode, None | Some(TurboquantMode::Turbo8));
            // Native flash FP8 path: skip dynamic AMAX scale updates, keep K/V scales at 1.0.
            // All data stored in FP8 cache uses scale=1.0 (raw BF16→FP8 cast). If we update
            // scales during decode, previously-stored prefill data would be read with wrong
            // scale, producing garbage. This matches FlashInfer/FlashAttn behavior.
            if !input_metadata.is_prefill && tq_uses_std_cache && self.k_scale.is_none() {
                self.maybe_update_kv_scales(&key_p, &value_p)?;
            }

            if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                match tq_mode {
                    Some(TurboquantMode::Turbo8) => {
                        // Turbo8: K as FP8 in standard cache, V as FP8 in standard cache
                        // (for prefill, which uses standard FP8 attention) + V as 4-bit
                        // in TQ buffers (for decode). flash_tq_store_k8v4 writes K and
                        // 4-bit V; flash_reshape_and_cache writes FP8 V for prefill use.
                        crate::flash::flash_reshape_and_cache(
                            &key_p,
                            &value_p,
                            key_cache.as_ref().unwrap(),
                            value_cache.as_ref().unwrap(),
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            &slot_mapping,
                        )?;
                        if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                            crate::flash::flash_tq_store_k8v4(
                                &key_p,
                                &value_p,
                                key_cache.as_ref().unwrap(),
                                &tq.v_absmax,
                                &tq.v_quant,
                                &slot_mapping,
                                self.k_scale.as_ref(),
                            )
                        }) {
                            r?;
                        }
                    }
                    Some(TurboquantMode::Turbo4) => {
                        // Turbo4: both K and V stored ONLY in TQ buffers (no standard cache)
                        if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                            crate::flash::flash_tq4_store(
                                &key_p,
                                &value_p,
                                tq.k_absmax.as_ref().unwrap(),
                                tq.k_quant.as_ref().unwrap(),
                                &tq.v_absmax,
                                &tq.v_quant,
                                &slot_mapping,
                                key_value_heads_p,
                                head_size_p,
                                tq_bs,
                            )
                        }) {
                            r?;
                        }
                    }
                    Some(TurboquantMode::Turbo3) => {
                        // Turbo3: both K and V stored ONLY in TQ buffers (no standard cache)
                        if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                            crate::flash::flash_tq3_store(
                                &key_p,
                                &value_p,
                                tq.k_absmax.as_ref().unwrap(),
                                tq.k_quant.as_ref().unwrap(),
                                &tq.v_absmax,
                                &tq.v_quant,
                                &slot_mapping,
                                key_value_heads_p,
                                head_size_p,
                                tq_bs,
                            )
                        }) {
                            r?;
                        }
                    }
                    None => {
                        crate::flash::flash_reshape_and_cache(
                            &key_p,
                            &value_p,
                            key_cache.as_ref().unwrap(),
                            value_cache.as_ref().unwrap(),
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            &slot_mapping,
                        )?;
                    }
                }
            }

            if input_metadata.is_prefill && input_metadata.block_tables.is_none() {
                return self.sdp_prefill(
                    query,
                    key,
                    value,
                    attention_mask,
                    input_metadata,
                    softcapping,
                );
            }

            let block_tables = input_metadata.block_tables.as_ref().unwrap();
            let context_lens = input_metadata.context_lens.as_ref().unwrap();

            if input_metadata.is_prefill {
                match tq_mode {
                    Some(TurboquantMode::Turbo4) => {
                        let r = with_turboquant_layer(self.layer_idx, |tq, _| {
                            crate::flash::flash_tq4_prefill(
                                &query_p,
                                tq.k_absmax.as_ref().unwrap(),
                                tq.k_quant.as_ref().unwrap(),
                                &tq.v_absmax,
                                &tq.v_quant,
                                block_tables,
                                context_lens,
                                attention_heads_p,
                                key_value_heads_p,
                                head_size_p,
                                self.scale,
                                softcapping.unwrap_or(0.0) as f32,
                                self.sliding_window,
                                tq_bs,
                                input_metadata.cu_seqlens_q.as_ref(),
                                input_metadata.max_seqlen_q,
                            )
                        });
                        if let Some(r) = r {
                            return r;
                        }
                    }
                    Some(TurboquantMode::Turbo3) => {
                        let r = with_turboquant_layer(self.layer_idx, |tq, _| {
                            crate::flash::flash_tq3_prefill(
                                &query_p,
                                tq.k_absmax.as_ref().unwrap(),
                                tq.k_quant.as_ref().unwrap(),
                                &tq.v_absmax,
                                &tq.v_quant,
                                block_tables,
                                context_lens,
                                attention_heads_p,
                                key_value_heads_p,
                                head_size_p,
                                self.scale,
                                softcapping.unwrap_or(0.0) as f32,
                                self.sliding_window,
                                tq_bs,
                                input_metadata.cu_seqlens_q.as_ref(),
                                input_metadata.max_seqlen_q,
                            )
                        });
                        if let Some(r) = r {
                            return r;
                        }
                    }
                    _ => {
                        return crate::flash::flash_prefill(
                            &query_p,
                            key_cache.as_ref().unwrap(),
                            value_cache.as_ref().unwrap(),
                            block_tables,
                            context_lens,
                            attention_heads_p,
                            key_value_heads_p,
                            head_size_p,
                            self.scale,
                            softcapping.unwrap_or(0.0) as f32,
                            self.sliding_window,
                            self.k_scale.as_ref(),
                            self.v_scale.as_ref(),
                            input_metadata.cu_seqlens_q.as_ref(),
                            input_metadata.max_seqlen_q,
                        );
                    }
                }
            }

            let output = query_p.zeros_like()?;

            match tq_mode {
                Some(ref mode) => {
                    if !TQ_DECODE_LOGGED.swap(true, Ordering::SeqCst) {
                        tracing::warn!(
                            layer = self.layer_idx,
                            mode = ?mode,
                            "TurboQuant decode path active"
                        );
                    }
                }
                None => {}
            }

            match tq_mode {
                Some(TurboquantMode::Turbo8) => {
                    let ws = self.flash_splitk_workspace.get_or_init(|| {
                        let max_seqs = 64;
                        let num_splits = crate::flash::TQ_NUM_SPLITS as usize;
                        let ws_stride = head_size_p + 2;
                        Tensor::zeros(
                            (max_seqs * attention_heads_p * num_splits * ws_stride,),
                            candle_core::DType::F32,
                            query_p.device(),
                        )
                        .unwrap()
                    });
                    if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                        crate::flash::flash_tq_decode_k8v4_splitk(
                            &query_p,
                            key_cache.as_ref().unwrap(),
                            &tq.v_absmax,
                            &tq.v_quant,
                            block_tables,
                            context_lens,
                            &output,
                            input_metadata.max_context_len,
                            attention_heads_p,
                            key_value_heads_p,
                            head_size_p,
                            self.scale,
                            softcapping.unwrap_or(0.0) as f32,
                            self.k_scale.as_ref(),
                            Some(ws),
                            self.sliding_window,
                        )
                    }) {
                        return r;
                    }
                }
                Some(TurboquantMode::Turbo4) => {
                    let ws = self.flash_splitk_workspace.get_or_init(|| {
                        let max_seqs = 64;
                        let num_splits = crate::flash::TQ_NUM_SPLITS as usize;
                        let ws_stride = head_size_p + 2;
                        Tensor::zeros(
                            (max_seqs * attention_heads_p * num_splits * ws_stride,),
                            candle_core::DType::F32,
                            query_p.device(),
                        )
                        .unwrap()
                    });
                    if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                        crate::flash::flash_tq4_decode(
                            &query_p,
                            tq.k_absmax.as_ref().unwrap(),
                            tq.k_quant.as_ref().unwrap(),
                            &tq.v_absmax,
                            &tq.v_quant,
                            block_tables,
                            context_lens,
                            &output,
                            attention_heads_p,
                            key_value_heads_p,
                            head_size_p,
                            input_metadata.max_context_len,
                            self.scale,
                            softcapping.unwrap_or(0.0) as f32,
                            self.sliding_window,
                            Some(ws),
                        )
                    }) {
                        return r;
                    }
                }
                Some(TurboquantMode::Turbo3) => {
                    let ws = self.flash_splitk_workspace.get_or_init(|| {
                        let max_seqs = 64;
                        let num_splits = crate::flash::TQ_NUM_SPLITS as usize;
                        let ws_stride = head_size_p + 2;
                        Tensor::zeros(
                            (max_seqs * attention_heads_p * num_splits * ws_stride,),
                            candle_core::DType::F32,
                            query_p.device(),
                        )
                        .unwrap()
                    });
                    if let Some(r) = with_turboquant_layer(self.layer_idx, |tq, _| {
                        crate::flash::flash_tq3_decode(
                            &query_p,
                            tq.k_absmax.as_ref().unwrap(),
                            tq.k_quant.as_ref().unwrap(),
                            &tq.v_absmax,
                            &tq.v_quant,
                            block_tables,
                            context_lens,
                            &output,
                            attention_heads_p,
                            key_value_heads_p,
                            head_size_p,
                            input_metadata.max_context_len,
                            self.scale,
                            softcapping.unwrap_or(0.0) as f32,
                            self.sliding_window,
                            Some(ws),
                        )
                    }) {
                        return r;
                    }
                }
                None => {}
            }

            let ws = self.flash_splitk_workspace.get_or_init(|| {
                let max_seqs = 64;
                let num_splits = crate::flash::NUM_SPLITS as usize;
                let ws_stride = head_size_p + 2;
                Tensor::zeros(
                    (max_seqs * attention_heads_p * num_splits * ws_stride,),
                    candle_core::DType::F32,
                    query_p.device(),
                )
                .unwrap()
            });
            return crate::flash::flash_decode(
                &query_p,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                block_tables,
                context_lens,
                &output,
                input_metadata.max_context_len,
                attention_heads_p,
                key_value_heads_p,
                head_size_p,
                self.scale,
                softcapping.unwrap_or(0.0) as f32,
                self.sliding_window,
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                Some(ws),
            );
        }

        let mut att = if input_metadata.is_prefill && input_metadata.block_tables.is_none() {
            //no context cache, prefill with naive scale-dot-product attention
            Some(self.sdp_prefill(
                query,
                key,
                value,
                attention_mask,
                input_metadata,
                softcapping,
            )?)
        } else {
            None
        };

        // The following for paged attention
        let slot_mapping = input_metadata.slot_mapping.flatten_all()?;

        let (query, key, value, attention_heads, key_value_heads, head_size) =
            Self::batch_major_qkv(query, key, value)?;

        // Write KvCache for SDP + Paged Attention
        let key = key
            .transpose(1, 2)?
            .reshape(((), key_value_heads, head_size))?;
        let value = value
            .transpose(1, 2)?
            .reshape(((), key_value_heads, head_size))?;

        self.maybe_update_kv_scales(&key, &value)?;

        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                &slot_mapping,
            )?;
        }

        if let Some(att) = att {
            //prefill result
            return Ok(att);
        }

        let block_tables = input_metadata.block_tables.as_ref().unwrap();
        let context_lens = input_metadata.context_lens.as_ref().unwrap();
        let query = query
            .transpose(1, 2)?
            .reshape(((), attention_heads, head_size))?;

        //decoding with paged-attn

        //if flashattn (flashattn with prefill kvcache) feature not enabled, use our custom paged attention for chunked prefill
        let cu_seqlens_q = if input_metadata.is_prefill && input_metadata.block_tables.is_some() {
            assert!(
                input_metadata.cu_seqlens_q.as_ref().is_some(),
                "Chunked prefill in conventional paged attention requires query lens tensor!"
            );
            // println!("chunked prefill with paged attention!");
            input_metadata.cu_seqlens_q.clone()
        } else {
            None
        };

        paged_attention(
            &query,
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            block_tables,
            context_lens,
            None,
            input_metadata.max_context_len,
            self.scale,
            softcapping.unwrap_or(1.0f64) as f32,
            cu_seqlens_q,
            self.sliding_window,
        )
    }
}
