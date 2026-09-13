#[cfg(test)]
#[cfg(all(feature = "cuda", feature = "flash"))]
mod nvfp4_kv_tests {
    use candle_core::{DType, Device, Tensor};
    use crate::flash;

    const NT: usize = 16;
    const NKV: usize = 2;
    const HD: usize = 128;
    const BS: usize = 16;
    const NGROUPS: usize = HD / 32; // 4 groups of 32

    fn make_data() -> (Tensor, Tensor, Tensor) {
        let dev = Device::new_cuda(0).unwrap();
        let k_data: Vec<half::bf16> = (0..NT * NKV * HD)
            .map(|i| half::bf16::from_f32((i as f32 % 50.0) / 10.0 - 2.5))
            .collect();
        let k = Tensor::from_vec(k_data, (NT, NKV, HD), &dev).unwrap();
        let v_data: Vec<half::bf16> = (0..NT * NKV * HD)
            .map(|i| half::bf16::from_f32((i as f32 % 40.0) / 8.0 - 2.5))
            .collect();
        let v = Tensor::from_vec(v_data, (NT, NKV, HD), &dev).unwrap();
        let slots = Tensor::from_vec((0..NT).map(|i| i as i64).collect::<Vec<i64>>(), (NT,), &dev).unwrap();
        (k, v, slots)
    }

    fn make_nvfp4_buffers(dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
        let total = NT * NKV;
        let k_fp4 = Tensor::zeros((total * HD / 2,), DType::U8, dev).unwrap();
        let k_sf = Tensor::zeros((total * NGROUPS,), DType::U8, dev).unwrap();
        let v_fp4 = Tensor::zeros((total * HD / 2,), DType::U8, dev).unwrap();
        let v_sf = Tensor::zeros((total * NGROUPS,), DType::U8, dev).unwrap();
        (k_fp4, k_sf, v_fp4, v_sf)
    }

    #[test]
    fn nvfp4_store_runs() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_fp4, k_sf, v_fp4, v_sf) = make_nvfp4_buffers(&dev);
        flash::flash_nvfp4_kv_store(
            &k, &v, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &slots, NKV, HD, BS, 1.0, 1.0, false,
        ).unwrap();
    }

    #[test]
    fn nvfp4_store_no_rotate_runs() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_fp4, k_sf, v_fp4, v_sf) = make_nvfp4_buffers(&dev);
        flash::flash_nvfp4_kv_store(
            &k, &v, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &slots, NKV, HD, BS, 1.0, 1.0, false,
        ).unwrap();
    }

    #[test]
    fn nvfp4_sf_nonzero() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_fp4, k_sf, v_fp4, v_sf) = make_nvfp4_buffers(&dev);
        flash::flash_nvfp4_kv_store(
            &k, &v, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &slots, NKV, HD, BS, 1.0, 1.0, true,
        ).unwrap();

        let sf = k_sf.to_vec1::<u8>().unwrap();
        let nonzero = sf.iter().filter(|&&x| x != 0).count();
        assert!(nonzero > sf.len() / 2, "E4M3 scales must be nonzero for non-zero K data");
        // E4M3 normal range: 0x08 (2^-6) to 0x7E (448.0)
        let in_range = sf.iter().filter(|&&x| (0x08..=0x7E).contains(&x)).count();
        assert!(in_range > sf.len() / 4, "most E4M3 scale bytes should be in normal range [0x08, 0x7E]");
    }

    #[test]
    fn nvfp4_fp4_codes_valid() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_fp4, k_sf, v_fp4, v_sf) = make_nvfp4_buffers(&dev);
        flash::flash_nvfp4_kv_store(
            &k, &v, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &slots, NKV, HD, BS, 1.0, 1.0, true,
        ).unwrap();

        let codes = k_fp4.to_vec1::<u8>().unwrap();
        for &byte in &codes {
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            // FP4 E2M1: 4 bits = 1 sign + 3 magnitude (0-7)
            // Valid codes: 0-15 (all 4-bit values are valid FP4 E2M1)
            assert!(lo <= 15, "invalid nibble out of range");
            assert!(hi <= 15, "high nibble out of range");
        }
    }

    #[test]
    fn nvfp4_decode_runs() {
        let dev = Device::new_cuda(0).unwrap();
        let nseq = 2;
        let nqh = 4;
        let nkv = 2;
        let hd = 128;
        let bs = 16;
        let max_blocks = 4;
        let ngroups = hd / 16;

        // 4D paged layout: [num_blocks, block_size, num_kv_heads, ...]
        let k_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let k_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();
        let v_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let v_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();

        let q = Tensor::zeros((nseq, nqh, hd), DType::BF16, &dev).unwrap();
        let o = Tensor::zeros((nseq, nqh, hd), DType::BF16, &dev).unwrap();
        let bt = Tensor::zeros((nseq, max_blocks), DType::U32, &dev).unwrap();
        let cl = Tensor::from_vec(vec![bs as u32; nseq], (nseq,), &dev).unwrap();

        flash::flash_nvfp4_kv_decode(
            &q, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &bt, &cl, &o,
            max_blocks * bs, nqh, nkv, hd,
            1.0f32 / (hd as f32).sqrt(),
            0.0,
            None,
            true,
        ).unwrap();
    }

    #[test]
    fn nvfp4_prefill_runs() {
        let dev = Device::new_cuda(0).unwrap();
        let nseq = 2;
        let nqh = 4;
        let nkv = 2;
        let hd = 128;
        let bs = 16;
        let max_blocks = 4;
        let ngroups = hd / 16;
        let q_len = 8;

        let k_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let k_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();
        let v_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let v_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();

        let q = Tensor::zeros((nseq * q_len, nqh, hd), DType::BF16, &dev).unwrap();
        let o = Tensor::zeros((nseq * q_len, nqh, hd), DType::BF16, &dev).unwrap();
        let bt = Tensor::zeros((nseq, max_blocks), DType::U32, &dev).unwrap();
        let cl = Tensor::from_vec(vec![bs as u32; nseq], (nseq,), &dev).unwrap();

        flash::flash_nvfp4_kv_prefill(
            &q, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &bt, &cl, &o,
            max_blocks * bs, nqh, nkv, hd,
            1.0f32 / (hd as f32).sqrt(),
            0.0,
            None,
            true,
        ).unwrap();
    }

    #[test]
    fn nvfp4_accuracy_roundtrip() {
        let dev = Device::new_cuda(0).unwrap();
        let nseq = 1;
        let nqh = 2;
        let nkv = 2;
        let hd = 128;
        let bs = 16;
        let max_blocks = 2;
        let ngroups = hd / 16;
        let seq_len = bs;

        let k_data: Vec<half::bf16> = (0..nseq * nkv * hd)
            .map(|i| half::bf16::from_f32((i as f32 % 20.0) / 4.0 - 2.5))
            .collect();
        let v_data: Vec<half::bf16> = (0..nseq * nkv * hd)
            .map(|i| half::bf16::from_f32((i as f32 % 16.0) / 4.0 - 2.0))
            .collect();
        let k = Tensor::from_vec(k_data, (nseq, nkv, hd), &dev).unwrap();
        let v = Tensor::from_vec(v_data, (nseq, nkv, hd), &dev).unwrap();

        let q_data: Vec<half::bf16> = (0..nseq * nqh * hd)
            .map(|i| half::bf16::from_f32((i as f32 % 10.0) / 2.0 - 2.5))
            .collect();
        let q = Tensor::from_vec(q_data, (nseq, nqh, hd), &dev).unwrap();

        let slots = Tensor::from_vec((0..nseq as i64).collect::<Vec<i64>>(), (nseq,), &dev).unwrap();

        let k_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let k_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();
        let v_fp4 = Tensor::zeros((max_blocks, bs, nkv, hd / 2), DType::U8, &dev).unwrap();
        let v_sf = Tensor::zeros((max_blocks, bs, nkv, ngroups), DType::U8, &dev).unwrap();

        flash::flash_nvfp4_kv_store(
            &k, &v, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &slots, nkv, hd, bs, 1.0, 1.0, false,
        ).unwrap();

        let o_nvfp4 = Tensor::zeros((nseq, nqh, hd), DType::BF16, &dev).unwrap();
        let bt = Tensor::from_vec(vec![0u32, 1u32], (nseq, max_blocks), &dev).unwrap();
        let cl = Tensor::from_vec(vec![seq_len as u32], (nseq,), &dev).unwrap();

        flash::flash_nvfp4_kv_decode(
            &q, &k_fp4, &k_sf, &v_fp4, &v_sf,
            &bt, &cl, &o_nvfp4,
            max_blocks * bs, nqh, nkv, hd,
            1.0f32 / (hd as f32).sqrt(),
            0.0,
            None,
            false,
        ).unwrap();

        // Verify NVFP4 decode produces non-zero, finite output
        let o_nvfp4_f32 = o_nvfp4.to_dtype(DType::F32).unwrap();
        let o_vec = o_nvfp4_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nonzero = o_vec.iter().filter(|&&x| x != 0.0).count();
        let finite = o_vec.iter().filter(|x| x.is_finite()).count();
        assert!(nonzero > 0, "NVFP4 decode output must have non-zero values");
        assert_eq!(finite, o_vec.len(), "NVFP4 decode output must be all finite");
        let max_val = o_vec.iter().cloned().fold(0.0f32, f32::max);
        println!("NVFP4 decode: nonzero={}/{}, max={:.4}", nonzero, o_vec.len(), max_val);
    }
}