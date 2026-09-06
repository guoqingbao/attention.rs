//! GPU-resident PDA (pushdown automaton) grammar kernels.
//!
//! The transition table is the flat pushdown-rs format:
//!   (q, a, top) -> (next_q, push[])
//!
//! The fusion interface (Pre3): one kernel launch does mask + sample + advance.
//! The PDA dispatch is OPTIONAL: if `ctrl` is null, the kernel runs plain
//! sampling with zero PDA overhead.
//!
//! Per-step (fused):
//!   1. Scan transitions for (ctrl, top) -> collect allowed inputs into VOB
//!   2. Apply VOB to logits (mask illegal tokens to -inf)
//!   3. Sample (greedy / top-k / top-p)
//!   4. Advance: look up (ctrl, sampled_token, top) -> (next_ctrl, new_top)
//!
//! The drafting (MTP/DFlash):
//!   5. Project: walk K draft tokens through the PDA, emit K+1 VOB masks

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Storage, Tensor};

use crate::kernels::ffi;

/// GPU-resident PDA table (the pushdown-rs transition format).
/// Uploaded once at model load. The transitions are a flat array of:
///   (q, a, top, next_q, push_len, push[0..push_len-1])
/// Each record is 5 + push_len u32s. The total array length is
///   sum over all transitions of (5 + push_len).
pub struct PdaPushdownTable {
    /// The flat transition array (all records concatenated).
    pub transitions: Tensor,
    /// The accepting state IDs.
    pub accepting: Tensor,
    /// CSR index: starting record in `transitions` for each ctrl state.
    /// `ctrl_offsets[q]` = the byte offset of the first transition where q == ctrl.
    /// `ctrl_offsets[q] + ctrl_counts[q]` = the byte offset of the first transition where q != ctrl.
    /// This allows the GPU kernel to scan only the transitions transitions (O(1-3)
    /// instead of O(8764) for the CFI grammar.
    pub ctrl_offsets: Tensor,
    pub ctrl_counts: Tensor,
    /// Number of control states.
    pub num_states: u32,
    /// Number of input symbols (the P alphabet size).
    pub num_inputs: u32,
    /// Number of stack symbols.
    pub num_stack_syms: u32,
    /// Total number of transition records.
    pub num_transitions: u32,
    /// The start state.
    pub start_state: u32,
    /// The start stack symbol (bottom marker).
    pub start_stack: u32,
    /// Words per VOB (ceil(num_inputs / 32)).
    pub words_per_vob: u32,
    /// Max stack depth (the bounded pushdown).
    pub max_stack_depth: u32,
}

/// Sampling strategy for the PDA step.
#[derive(Clone, Copy, Debug)]
pub enum PdaSampling {
    /// Greedy argmax (temperature ignored, K=1).
    Greedy,
    /// Top-k then top-p with temperature.
    TopKTopP { temperature: f32, top_k: i32, top_p: f32 },
}

impl PdaPushdownTable {
    /// Upload the pushdown-rs transition table to GPU. Called once at model load.
    /// The CSR index (ctrl_offsets, ctrl_counts) is computed from the flat
    /// transition array: for each ctrl state q, find the byte offset and count
    /// of records where the q field equals q.
    pub fn upload(
        transitions: Vec<u32>,
        accepting: Vec<u32>,
        num_states: u32,
        num_inputs: u32,
        num_stack_syms: u32,
        num_transitions: u32,
        start_state: u32,
        start_stack: u32,
        max_stack_depth: u32,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let words_per_vob = (num_inputs + 31) / 32;
        let n_transitions = transitions.len();
        let n_accepting = accepting.len();
        // Compute the CSR index: for each ctrl state, the starting record index
        // and the count of records. The transition array is a flat list of
        // variable-length records: (q, a, top, next_q, push_len, push[]).
        let mut record_starts: Vec<usize> = Vec::with_capacity(num_transitions as usize);
        let mut record_qs: Vec<u32> = Vec::with_capacity(num_transitions as usize);
        let mut off = 0usize;
        for _ in 0..num_transitions {
            if off + 5 > n_transitions { break; }
            record_starts.push(off);
            record_qs.push(transitions[off]); // the q field
            let push_len = transitions[off + 4] as usize;
            off += 5 + push_len;
        }
        // Build the CSR: ctrl_offsets[q] = the u32 offset into the flat array
        // where q's first record begins. ctrl_counts[q] = the number of
        // records for q. The kernel uses ctrl_offsets[q] to jump directly
        // to q's records (O(1) instead of walking from record 0).
        let mut ctrl_offsets = vec![n_transitions as u32; num_states as usize]; // sentinel: end of array
        let mut ctrl_counts = vec![0u32; num_states as usize];
        for (i, &q) in record_qs.iter().enumerate() {
            if q < num_states {
                if ctrl_counts[q as usize] == 0 {
                    ctrl_offsets[q as usize] = record_starts[i] as u32; // u32 offset, not record index
                }
                ctrl_counts[q as usize] += 1;
            }
        }
        Ok(Self {
            transitions: Tensor::from_vec(transitions, (n_transitions,), device)?,
            accepting: Tensor::from_vec(accepting, (n_accepting,), device)?,
            ctrl_offsets: Tensor::from_vec(ctrl_offsets, (num_states as usize,), device)?,
            ctrl_counts: Tensor::from_vec(ctrl_counts, (num_states as usize,), device)?,
            num_states,
            num_inputs,
            num_stack_syms,
            num_transitions,
            start_state,
            start_stack,
            words_per_vob,
            max_stack_depth,
        })
    }

    /// Upload from the pushdown-rs CudaPackage (the bitvec + source primitives).
    #[cfg(feature = "cuda")]
    pub fn from_cuda_package(
        pkg: &pushdown_rs::cuda::CudaPackage,
        device: &candle_core::Device,
    ) -> Result<Self> {
        use pushdown_rs::machine::PdaMachine;
        let machine = PdaMachine::from_bitvec(&pkg.bitvec)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        // Flatten the transitions into the GPU array format:
        //   (q, a, top, next_q, push_len, push[0..push_len-1]) per record.
        let mut flat: Vec<u32> = Vec::new();
        for t in &machine.transitions {
            flat.push(t.q);
            flat.push(t.a);
            flat.push(t.top);
            flat.push(t.next_q);
            flat.push(t.push.len() as u32);
            flat.extend_from_slice(&t.push);
        }
        Self::upload(
            flat,
            machine.accepting.clone(),
            machine.num_states,
            machine.num_inputs,
            machine.num_stack_syms,
            machine.transitions.len() as u32,
            machine.start_state,
            machine.start_stack,
            8, // the bounded stack depth (the D).
            device,
        )
    }

    fn ptr_u32(t: &Tensor) -> Result<*const u32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *const u32),
            _ => candle_core::bail!("PDA table tensor must be on CUDA"),
        }
    }
    fn ptr_u32_mut(t: &Tensor) -> Result<*mut u32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *mut u32),
            _ => candle_core::bail!("PDA table tensor must be on CUDA"),
        }
    }
    fn ptr_logits_f32(t: &Tensor) -> Result<*const f32> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *const f32),
            _ => candle_core::bail!("logits must be on CUDA"),
        }
    }
    fn ptr_logits_bf16(t: &Tensor) -> Result<*const std::ffi::c_void> {
        let (s, _) = t.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const _),
            _ => candle_core::bail!("logits must be on CUDA"),
        }
    }

    /// Fused PDA decode step: compute mask + sample + advance in ONE kernel launch.
    ///
    /// This is the Pre3 fusion: the maskDA transition scan is fused into the
    /// sampling kernel, saving 2 separate launches (mask compute + advance).
    /// The PDA dispatch is optional: if `ctrl` is null, the kernel runs
    /// plain sampling with no PDA overhead.
    ///
    /// `logits`: [batch, vocab] F32/BF16
    /// `ctrl`: [batch] current control state (or null for no PDA)
    /// `stack`: [batch * D] bounded stack (or null)
    /// `sp`: [batch] stack pointer (or null)
    ///
    /// Returns (out_ctrl, out_sp, sampled_tokens).
    pub fn fused_sample(
        &self,
        logits: &Tensor,
        ctrl: &Tensor,
        stack: &Tensor,
        sp: &Tensor,
        sampling: &PdaSampling,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = logits.device();
        let (batch, vocab) = logits.dims2()?;
        let d = stack.dim(1).unwrap_or(self.max_stack_depth as usize).max(1);
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let words = ((vocab + 31) / 32) as u32;
        debug_assert_eq!(
            self.words_per_vob, words,
            "PDA table words_per_vob ({}) must equal ceil(model_vocab/32) ({})",
            self.words_per_vob, words
        );

        let (k, temp, topp) = match sampling {
            PdaSampling::Greedy => (1i32, 0.0f32, 1.0f32),
            PdaSampling::TopKTopP { temperature, top_k, top_p } => (*top_k, *temperature, *top_p),
        };

        // Allocate outputs.
        let out_ctrl = Tensor::zeros((batch,), DType::U32, &dev)?;
        let out_sp = Tensor::zeros((batch,), DType::U32, &dev)?;
        let out_tokens = Tensor::zeros((batch,), DType::U32, &dev)?;

        let p_ctrl = Self::ptr_u32(ctrl)?;
        let p_stack = Self::ptr_u32(stack)?;
        let p_sp = Self::ptr_u32(sp)?;
        let p_out_ctrl = Self::ptr_u32_mut(&out_ctrl)?;
        let p_out_sp = Self::ptr_u32_mut(&out_sp)?;
        let p_out_tokens = Self::ptr_u32_mut(&out_tokens)?;
        let p_trans = Self::ptr_u32(&self.transitions)?;
        let p_accept = Self::ptr_u32(&self.accepting)?;
        let p_ctrl_offsets = Self::ptr_u32(&self.ctrl_offsets)?;
        let p_ctrl_counts = Self::ptr_u32(&self.ctrl_counts)?;

        unsafe {
            match logits.dtype() {
                DType::F32 => {
                    let p_logits = Self::ptr_logits_f32(logits)?;
                    ffi::pda_fused_sample_f32(
                        p_logits, p_ctrl, p_stack, p_sp,
                        p_out_ctrl, p_out_sp, p_out_tokens,
                        p_trans, p_accept, p_ctrl_offsets, p_ctrl_counts,
                        self.num_states, self.num_stack_syms,
                        words, batch as i32, vocab as i32,
                        k, temp, topp, d as i32, stream,
                    );
                }
                DType::BF16 => {
                    let p_logits = Self::ptr_logits_bf16(logits)?;
                    ffi::pda_fused_sample_bf16(
                        p_logits, p_ctrl, p_stack, p_sp,
                        p_out_ctrl, p_out_sp, p_out_tokens,
                        p_trans, p_accept, p_ctrl_offsets, p_ctrl_counts,
                        self.num_states, self.num_stack_syms,
                        words, batch as i32, vocab as i32,
                        k, temp, topp, d as i32, stream,
                    );
                }
                _ => {
                    return Err(candle_core::Error::Msg(format!(
                        "PDA fused sampling supports F32/BF16, got {:?}", logits.dtype()
                    )));
                }
            }
        }
        Ok((out_ctrl, out_sp, out_tokens))
    }

    /// Fused PDA projection for drafting (MTP/DFlash): walk K draft tokens
    /// through the PDA, emitting K+1 VOB masks in ONE kernel launch.
    ///
    /// `ctrl`: [batch] current control state
    /// `stack`: [batch * D] bounded stack
    /// `sp`: [batch] stack pointer
    /// `draft`: [batch, K] draft tokens
    ///
    /// Returns [batch * (K+1) * words_per_vob] U32 tensor of projected masks.
    pub fn fused_project(
        &self,
        ctrl: &Tensor,
        stack: &Tensor,
        sp: &Tensor,
        draft: &Tensor,
    ) -> Result<Tensor> {
        let dev = ctrl.device();
        let batch = ctrl.dim(0)?;
        let k = draft.dim(1)?;
        let d = stack.dim(1).unwrap_or(self.max_stack_depth as usize).max(1);
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;
        let out = Tensor::zeros((batch * (k + 1) * self.words_per_vob as usize,), DType::U32, &dev)?;

        let p_ctrl = Self::ptr_u32(ctrl)?;
        let p_stack = Self::ptr_u32(stack)?;
        let p_sp = Self::ptr_u32(sp)?;
        let p_draft = Self::ptr_u32(draft)?;
        let p_out = Self::ptr_u32_mut(&out)?;
        let p_trans = Self::ptr_u32(&self.transitions)?;
        let p_accept = Self::ptr_u32(&self.accepting)?;
        let p_ctrl_offsets = Self::ptr_u32(&self.ctrl_offsets)?;
        let p_ctrl_counts = Self::ptr_u32(&self.ctrl_counts)?;

        unsafe {
            ffi::pda_fused_project_masks(
                p_ctrl, p_stack, p_sp, p_draft, p_out,
                p_trans, p_accept, p_ctrl_offsets, p_ctrl_counts,
                self.num_states, self.num_stack_syms,
                self.words_per_vob, batch as i32, k as i32, d as i32, stream,
            );
        }
        Ok(out)
    }

    /// Convert the fused_project VOB output to the DFlash/MTP `allow` matrix
    /// format: [batch * (K+1), vocab] F32 (1.0 = legal, 0.0 = illegal).
    ///
    /// The PDA projection produces [batch * (K+1) * words_per_vob] U32 VOB masks.
    /// This expands each VOB word into per-token f32 values.
    ///
    /// If no grammar is active, the caller passes `None` to DFlash/MTP and this
    /// conversion is never called (zero overhead).
    pub fn vob_to_allow(&self, vob: &Tensor, batch: usize, k: usize, vocab: usize) -> Result<Tensor> {
        let dev = vob.device();
        let words = self.words_per_vob as usize;
        let positions = batch * (k + 1);
        let mut allow = vec![0.0f32; positions * vocab];

        // Read the VOB from GPU to CPU (small: positions * words u32s).
        let vob_cpu = vob.to_device(&candle_core::Device::Cpu)?;
        let vob_flat = vob_cpu.flatten_all().unwrap().to_vec1::<u32>().unwrap();

        for pos in 0..positions {
            for w in 0..words {
                let word = vob_flat[pos * words + w];
                for bit in 0..32 {
                    let token_idx = w * 32 + bit;
                    if token_idx >= vocab {
                        break;
                    }
                    if (word >> bit) & 1 == 1 {
                        allow[pos * vocab + token_idx] = 1.0;
                    }
                }
            }
        }

        Tensor::from_vec(allow, (positions, vocab), dev)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A tiny hand-built PDA: 2 control states, 1 input.
    ///   ctrl 0: input 1 -> Shift to ctrl 1 (push 1)
    ///   ctrl 1: accept
    /// Verifies GPU fused_sample matches a CPU reference.
    #[test]
    fn pda_fused_sample_matches_cpu_reference() {
        let dev = Device::new_cuda(0).unwrap();
        let vocab = 4;
        let words = (vocab + 31) / 32;

        // Transition flat array: (q, a, top, next_q, push_len, push[])
        //   Record 0: q=0, a=1, top=0, next_q=1, push_len=1, push=[1]
        //   Record 1: q=1, a=0, top=1, next_q=1, push_len=0 (self-loop on accept)
        let transitions = vec![
            0, 1, 0, 1, 1, 1, // record 0: (q=0, a=1, top=0, next=1, push_len=1, push=[1])
            1, 0, 1, 1, 0,     // record 1: (q=1, a=0, top=1, next=1, push_len=0)
        ];
        let table = PdaPushdownTable::upload(
            transitions,
            vec![1], // accepting
            2,       // num_states
            1,       // num_inputs
            2,       // num_stack_syms
            2,       // num_transitions
            0,       // start_state
            0,       // start_stack
            1,       // max_stack_depth
            &dev,
        ).unwrap();

        // CPU reference: from ctrl=0, stack=[0], sp=1.
        //   Input 1 is allowed (record 0 matches). Greedy picks 1.
        //   Advance: ctrl 0->1, push 1, sp 1->2.
        let logits = Tensor::from_vec(vec![0.0f32, 5.0, 1.0, 0.0], (1, vocab), &dev).unwrap();
        let ctrl = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
        let stack = Tensor::from_vec(vec![0u32, 0], (1, 2), &dev).unwrap();
        let sp = Tensor::from_vec(vec![1u32], (1,), &dev).unwrap();

        let (out_ctrl, out_sp, out_tok) = table.fused_sample(&
            &logits, &ctrl, &stack, &sp, &PdaSampling::Greedy,
        ).unwrap();
        let oc = out_ctrl.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let os = out_sp.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let ot = out_tok.flatten_all().unwrap().to_vec1::<u32>().unwrap();

        assert_eq!(ot[0], 1, "greedy should pick token 1 (highest allowed logit)");
        assert_eq!(oc[0], 1, "shift moves ctrl 0 -> 1");
        assert_eq!(os[0], 1, "push_len=1: pop 1, push 1, sp stays at 1");
        println!("PDA fused_sample matches CPU reference: tok={} ctrl={} sp={}", ot[0], oc[0], os[0]);
    }

    /// GPU projection: fused_project over a 2-token draft must emit the K+1
    /// per-position masks.
    #[test]
    fn pda_fused_project_masks() {
        let dev = Device::new_cuda(0).unwrap();
        let words = 1u32; // vocab 4
        // 3-state chain: ctrl0 -(1--> ctrl1 --(2--> ctrl2(accept)
        let transitions = vec![
            0, 1, 0, 1, 1, 1, // record 0: (q=0, a=1, top=0, next=1, push_len=1, push=[1])
            1, 2, 1, 2, 1, 2, // record 1: (q=1, a=2, top=1, next=2, push_len=1, push=[2])
        ];
        let table = PdaPushdownTable::upload(
            transitions,
            vec![2], // accepting
            3,       // num_states
            3,       // num_inputs
            3,       // num_stack_syms
            2,       // num_transitions
            0,       // start_state
            0,       // start_stack
            4,       // max_stack_depth
            &dev,
        ).unwrap();

        let ctrl = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
        let stack = Tensor::zeros((1, 4), DType::U32, &dev).unwrap();
        let sp = Tensor::from_vec(vec![1u32], (1,), &dev).unwrap();
        let draft = Tensor::from_vec(vec![1u32, 2], (1, 2), &dev).unwrap();

        let out = table.fused_project(&ctrl, &stack, &sp, &draft).unwrap();
        let flat = out.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        // K+1 = 3 masks, 1 word each: pos0=ctrl0{1}=2, pos1=ctrl1{2}=4, pos2=ctrl2{}=0
        assert_eq!(flat, vec![2u32, 4, 0], "projected masks must match the per-position control masks");
        println!("PDA fused_project: {:?} (3 positions for a 2-token draft)", flat);
    }
}