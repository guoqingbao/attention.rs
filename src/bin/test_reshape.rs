fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{Device, Tensor};
    let device = Device::Cpu;
    let key = Tensor::zeros((1, 8, 61, 128), candle_core::DType::F32, &device)?;
    let key_t = key.transpose(1, 2)?;
    println!("key_t contiguous: {}", key_t.is_contiguous());
    let key_r = key_t.reshape((61, 8, 128));
    match key_r {
        Ok(t) => {
            println!("reshape worked! contiguous: {}", t.is_contiguous());
            println!("strides: {:?}", t.stride());
        }
        Err(e) => println!("reshape failed: {}", e),
    }
    Ok(())
}
