#[cfg(test)]
#[cfg(all(feature = "cuda", feature = "flash"))]
mod fp8_rot_tests {
    use candle_core::{DType, Device, Tensor};
    use crate::flash;

    const NT: usize = 16;
    const NKV: usize = 2;
    const HD: usize = 128;
    const BS: usize = 16;

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

    fn make_fp8_caches(dev: &Device) -> (Tensor, Tensor) {
        let total = NT * NKV * HD;
        let k_cache = Tensor::zeros((NT / BS, BS, NKV, HD), DType::U8, dev).unwrap();
        let v_cache = Tensor::zeros((NT / BS, BS, NKV, HD), DType::U8, dev).unwrap();
        (k_cache, v_cache)
    }

    #[test]
    fn rot_store_runs() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_cache, v_cache) = make_fp8_caches(&dev);
        flash::flash_fp8_rot_store(
            &k, &v, &k_cache, &v_cache, &slots,
            1.0, 1.0, true,
        ).unwrap();
    }

    #[test]
    fn rot_store_no_rotate_runs() {
        let (k, v, slots) = make_data();
        let dev = k.device();
        let (k_cache, v_cache) = make_fp8_caches(&dev);
        flash::flash_fp8_rot_store(
            &k, &v, &k_cache, &v_cache, &slots,
            1.0, 1.0, false,
        ).unwrap();
    }

    #[test]
    fn rot_store_c_scales() {
        let (k, v, slots) = make_data();
        let dev = k.device();

        let (k_cache1, v_cache1) = make_fp8_caches(&dev);
        flash::flash_fp8_rot_store(
            &k, &v, &k_cache1, &v_cache1, &slots,
            1.0, 1.0, true,
        ).unwrap();

        let (k_cache2, v_cache2) = make_fp8_caches(&dev);
        flash::flash_fp8_rot_store(
            &k, &v, &k_cache2, &v_cache2, &slots,
            0.5, 1.0, true,
        ).unwrap();

        // c_k=0.5 should produce different K cache bytes than c_k=1.0
        let k1 = k_cache1.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let k2 = k_cache2.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let diffs = k1.iter().zip(k2.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs > 0, "c_k=0.5 must change K cache bytes vs c_k=1.0");

        // V cache must be identical (c_v=1.0 in both)
        let v1 = v_cache1.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let v2 = v_cache2.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(v1, v2, "c_v=1.0 in both calls, V cache must be identical");
    }
}