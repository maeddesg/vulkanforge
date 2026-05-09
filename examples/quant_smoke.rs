use vulkanforge::quantize::{dequantize_q4k_to_f32, quantize_f32_to_q4k};

fn main() {
    let bytes = std::fs::read("/tmp/gemma4_qproj_head.bin").expect("read bin");
    assert_eq!(bytes.len(), 256 * 4);
    let mut input = vec![0f32; 256];
    for i in 0..256 {
        let b = [bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]];
        input[i] = f32::from_le_bytes(b);
    }
    let q = quantize_f32_to_q4k(&input);
    let dq = dequantize_q4k_to_f32(&q);
    let mut max_err = 0f32;
    let mut sum_sq = 0f64;
    for i in 0..256 {
        let e = (input[i] - dq[i]).abs();
        if e > max_err {
            max_err = e;
        }
        sum_sq += (input[i] - dq[i]) as f64 * (input[i] - dq[i]) as f64;
    }
    let rmse = (sum_sq / 256.0).sqrt();
    let std = (input.iter().map(|x| (x * x) as f64).sum::<f64>() / 256.0).sqrt();
    println!("Gemma-4 q_proj.layer0 head[256] ({} bytes -> {} bytes Q4_K, ratio {:.2}x):", input.len() * 4, q.len(), (input.len() * 4) as f32 / q.len() as f32);
    println!("  max_abs_err = {:.6}", max_err);
    println!("  rmse        = {:.6}", rmse);
    println!("  signal std  = {:.6}", std);
    println!("  rmse / std  = {:.4}  (lower = less quantization noise)", rmse / std);
}
