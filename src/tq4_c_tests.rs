#[cfg(test)]
#[cfg(all(feature = "cuda", feature = "flash"))]
mod tq4_c_tests {
    use candle_core::{DType, Device, Tensor};
    use crate::flash;

    const NT: usize = 64;
    const NKV: usize = 4;
    const HD: usize = 128;
    const BS: usize = 16;

    fn make_tensors() -> (Tensor, Tensor, Tensor) {
        let dev = Device::new_cuda(0).unwrap();
        let k_data: Vec<half::bf16> = (0..NT * NKV * HD)
            .map(|i| half::bf16::from_f32((i as f32 % 100.0) / 10.0 - 5.0))
            .collect();
        let k = Tensor::from_vec(k_data, (NT, NKV, HD), &dev).unwrap();
        let v_data: Vec<half::bf16> = (0..NT * NKV * HD)
            .map(|i| half::bf16::from_f32((i as f32 % 80.0) / 8.0 - 4.0))
            .collect();
        let v = Tensor::from_vec(v_data, (NT, NKV, HD), &dev).unwrap();
        let slots = Tensor::from_vec((0..NT).map(|i| i as i64).collect::<Vec<i64>>(), (NT,), &dev).unwrap();
        (k, v, slots)
    }

    fn make_buffers(dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
        let total = NT * NKV;
        let k_abs = Tensor::zeros((total,), DType::F32, dev).unwrap();
        let k_q = Tensor::zeros((total * HD / 2,), DType::U8, dev).unwrap();
        let v_abs = Tensor::zeros((total,), DType::F32, dev).unwrap();
        let v_q = Tensor::zeros((total * HD / 2,), DType::U8, dev).unwrap();
        (k_abs, k_q, v_abs, v_q)
    }

    #[test]
    fn c1_runs() {
        let (k, v, slots) = make_tensors();
        let dev = k.device();
        let (k_abs, k_q, v_abs, v_q) = make_buffers(&dev);
        flash::flash_tq4_store(
            &k, &v, &k_abs, &k_q, &v_abs, &v_q,
            &slots, NKV, HD, BS, 1.0, 1.0,
        ).unwrap();
    }

    #[test]
    fn asymmetric_c_runs() {
        let (k, v, slots) = make_tensors();
        let dev = k.device();
        let (k_abs, k_q, v_abs, v_q) = make_buffers(&dev);
        flash::flash_tq4_store(
            &k, &v, &k_abs, &k_q, &v_abs, &v_q,
            &slots, NKV, HD, BS, 0.156, 1.0,
        ).unwrap();
    }

    #[test]
    fn c1_stores_raw_absmax() {
        let (k, v, slots) = make_tensors();
        let dev = k.device();
        let (k_abs, k_q, v_abs, v_q) = make_buffers(&dev);
        flash::flash_tq4_store(
            &k, &v, &k_abs, &k_q, &v_abs, &v_q,
            &slots, NKV, HD, BS, 1.0, 1.0,
        ).unwrap();
        let stored_k = k_abs.to_vec1::<f32>().unwrap();
        let stored_v = v_abs.to_vec1::<f32>().unwrap();
        assert!(stored_k.iter().all(|&x| x >= 0.0 && x.is_finite()));
        assert!(stored_v.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn c_s_scales_k_absmax_only() {
        let (k, v, slots) = make_tensors();
        let dev = k.device();

        let (k_abs1, k_q1, v_abs1, v_q1) = make_buffers(&dev);
        flash::flash_tq4_store(
            &k, &v, &k_abs1, &k_q1, &v_abs1, &v_q1,
            &slots, NKV, HD, BS, 1.0, 1.0,
        ).unwrap();
        let raw_k = k_abs1.to_vec1::<f32>().unwrap();
        let raw_v = v_abs1.to_vec1::<f32>().unwrap();

        let (k_abs2, k_q2, v_abs2, v_q2) = make_buffers(&dev);
        flash::flash_tq4_store(
            &k, &v, &k_abs2, &k_q2, &v_abs2, &v_q2,
            &slots, NKV, HD, BS, 0.5, 1.0,
        ).unwrap();
        let half_k = k_abs2.to_vec1::<f32>().unwrap();
        let same_v = v_abs2.to_vec1::<f32>().unwrap();

        // K should be scaled by 0.5
        for (i, (r, h)) in raw_k.iter().zip(half_k.iter()).enumerate() {
            assert!((h - 0.5 * r).abs() < 1e-5,
                "K_absmax[{}]: 0.5*raw={:.8}, stored={:.8}", i, 0.5 * r, h);
        }
        // V should be unchanged (c_v=1.0)
        for (i, (r, s)) in raw_v.iter().zip(same_v.iter()).enumerate() {
            assert!((s - r).abs() < 1e-5,
                "V_absmax[{}]: raw={:.8}, stored={:.8}", i, r, s);
        }
    }
}