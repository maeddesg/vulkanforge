//! CPU-side compute kernels.
//!
//! Currently houses the optional `lm_head` offload introduced in
//! Sprint 40 (v0.3.10). Activated by `VF_CPU_LM_HEAD=1`. The
//! offload requantizes `lm_head` weights from FP8 / FP16 / FP32 to
//! Q6_K once at model load and runs the per-token GEMV on the CPU
//! using AVX-512-friendly scalar code that auto-vectorizes through
//! the inner loop. Phase A: blocking — the GPU waits for the CPU
//! lm_head to finish before sampling. Phase B (pipeline overlap)
//! is a future sprint.

pub mod lm_head;
pub mod q6k;
