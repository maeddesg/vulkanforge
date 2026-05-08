//! Sprint 44B-4 — debug / introspection helpers extracted from
//! `forward/mod.rs`. Pure code-move.
//!
//! Three groups of helpers live here:
//! - `forward_layer_debug` + `forward_layer_debug_intermediate` —
//!   single-layer drivers used by the regression-test `tests/` and the
//!   Sprint 24/26 bisect harnesses; they run a fresh seed through ONE
//!   layer up to a configurable halt point and read scratch_b back to
//!   the host.
//! - `dispatch_layer_partial` — the halt-point-aware variant of
//!   `dispatch_layer`, only ever called from `forward_layer_debug*`.
//! - `read_scratch_b` — host stage-and-readback for any of the GpuOnly
//!   per-token scratch buffers; used by the layer-debug helpers and (in
//!   diagnostic builds) by `dispatch_final` via `VF_FINAL_NORM_DUMP`.
//!
//! Two free `maybe_dump_*` env-gated logging helpers also live here.
//! They're called from the decode hot path (`wait_and_read_logits`,
//! `forward_token`, `logits`, `dispatch_final`) so they're `pub(super)`
//! — see `forward/decode.rs` for the call sites.

use ash::vk;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;

use super::arch::{
    compute_barrier, layer_weight, layer_weight_opt, layer_weight_scale_block,
    layer_weight_scale_buf, layer_weight_scale_scalar, layer_weight_shader,
};
use super::state::{DebugTarget, Forward};

impl Forward {
    /// Debug helper — run ONE layer up to a chosen halt point and
    /// return the relevant intermediate as a `Vec<f32>`. The buffer
    /// returned has *exactly* the size of the relevant intermediate
    /// (e.g. n_heads*head_dim for AfterQkvProj-Q, not hidden_dim).
    pub fn forward_layer_debug_intermediate(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
        target: DebugTarget,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Debug path uses a partial layer dispatch with non-stable
        // bindings — invalidate any cached sets from prior decode runs.
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer_partial(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, target,
            );
        })?;

        // Pick the relevant buffer + size for readback.
        let cfg = self.config.clone();
        let (src_buf, count) = match target {
            DebugTarget::AttnNorm => (self.cur().hidden_norm.handle, cfg.hidden_dim as u64),
            DebugTarget::QProj | DebugTarget::QNormRope => (
                self.cur().q_buf.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::KProj | DebugTarget::KNormRope => (
                self.cur().k_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::VProj => (
                self.cur().v_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::AttnOut => (
                self.cur().attn_out.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
        };
        // Stage into scratch_a (host-visible).
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(src_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(count * 4);
            dev.device.cmd_copy_buffer(cmd, src_buf, self.cur().scratch_a.handle,
                std::slice::from_ref(&copy));
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes = self.cur().scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes[..(count as usize) * 4]).to_vec())
    }

    fn dispatch_layer_partial(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input: vk::Buffer,
        halt: DebugTarget,
    ) {
        let cfg = self.config.clone();

        // attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, self.cur().hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        if halt == DebugTarget::AttnNorm { return; }
        compute_barrier(dev, cmd);

        // Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight", self.mul_mat_vec_subgroup_enabled);
        let sk = layer_weight_shader(model, layer, "attn_k.weight", self.mul_mat_vec_subgroup_enabled);
        let sv = layer_weight_shader(model, layer, "attn_v.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_q = layer_weight_scale_scalar(model, layer, "attn_q.weight");
        let scale_k = layer_weight_scale_scalar(model, layer, "attn_k.weight");
        let scale_v = layer_weight_scale_scalar(model, layer, "attn_v.weight");
        // Sprint 24-Inline Step 0 — route FP8 weights through the
        // harness-style per-channel path when a scale buffer exists.
        let sb_q = layer_weight_scale_buf(model, layer, "attn_q.weight");
        let sb_k = layer_weight_scale_buf(model, layer, "attn_k.weight");
        let sb_v = layer_weight_scale_buf(model, layer, "attn_v.weight");
        let q_h = self.cur().q_buf.handle;
        let k_h = self.cur().k_buf.handle;
        let v_h = self.cur().v_buf.handle;
        let in_h = self.cur().hidden_norm.handle;
        if let Some(s) = sb_q {
            self.run_gemv_fp8_dispatch(dev, cmd, wq, s, in_h, q_h,
                cfg.hidden_dim, cfg.n_heads * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_q.weight"), "gemv_q");
        } else {
            self.run_gemv(dev, registry, cmd, sq, wq, in_h, q_h,
                cfg.hidden_dim, cfg.n_heads * cfg.head_dim, scale_q, "gemv_q");
        }
        if let Some(s) = sb_k {
            self.run_gemv_fp8_dispatch(dev, cmd, wk, s, in_h, k_h,
                cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_k.weight"), "gemv_k");
        } else {
            self.run_gemv(dev, registry, cmd, sk, wk, in_h, k_h,
                cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, scale_k, "gemv_k");
        }
        if let Some(s) = sb_v {
            self.run_gemv_fp8_dispatch(dev, cmd, wv, s, in_h, v_h,
                cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_v.weight"), "gemv_v");
        } else {
            self.run_gemv(dev, registry, cmd, sv, wv, in_h, v_h,
                cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, scale_v, "gemv_v");
        }
        if matches!(halt, DebugTarget::QProj | DebugTarget::KProj | DebugTarget::VProj) {
            return;
        }
        compute_barrier(dev, cmd);

        // Sprint 24B — Q/K/V bias-add (Qwen2 attention biases). Skipped
        // for architectures without biases (Llama, Qwen3, Mistral).
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_bias = layer_weight_opt(model, layer, "attn_q.bias");
        let k_bias = layer_weight_opt(model, layer, "attn_k.bias");
        let v_bias = layer_weight_opt(model, layer, "attn_v.bias");
        if q_bias.is_some() || k_bias.is_some() || v_bias.is_some() {
            let q_buf = self.cur().q_buf.handle;
            let k_buf = self.cur().k_buf.handle;
            let v_buf = self.cur().v_buf.handle;
            if let Some(b) = q_bias { self.run_bias_add(dev, registry, cmd, q_buf, b, q_buf, q_dim, 1, "bias_q"); }
            if let Some(b) = k_bias { self.run_bias_add(dev, registry, cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k"); }
            if let Some(b) = v_bias { self.run_bias_add(dev, registry, cmd, v_buf, b, v_buf, kv_dim, 1, "bias_v"); }
            compute_barrier(dev, cmd);
        }

        // Q/K norm — Qwen-only (Phase 4D: gated on cfg.has_qk_norm).
        if cfg.has_qk_norm {
            let wqn = layer_weight(model, layer, "attn_q_norm.weight");
            let wkn = layer_weight(model, layer, "attn_k_norm.weight");
            self.run_rms_norm(dev, registry, cmd,
                             self.cur().q_buf.handle, wqn, self.cur().q_buf.handle,
                             cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
            self.run_rms_norm(dev, registry, cmd,
                             self.cur().k_buf.handle, wkn, self.cur().k_buf.handle,
                             cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k");
            compute_barrier(dev, cmd);
        }

        // RoPE
        self.run_rope_neox(dev, registry, cmd, self.cur().q_buf.handle, self.cur().q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.cur().k_buf.handle, self.cur().k_buf.handle,
                           cfg.head_dim, cfg.n_kv_heads, position, "rope_k");
        if matches!(halt, DebugTarget::QNormRope | DebugTarget::KNormRope) {
            return;
        }
        compute_barrier(dev, cmd);

        // KV write — Sprint 9d.3: dispatch the FP32→FP16 conversion
        // compute shader when the cache is FP16-allocated, otherwise
        // keep the cheap vkCmdCopyBuffer transfer.
        let row_bytes = self.kv_cache.row_bytes(layer);
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let k_src = self.cur().k_buf.handle;
        let v_src = self.cur().v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        if self.kv_cache.is_fp8() {
            let kv_elements = cfg.n_kv_heads * cfg.head_dim;
            self.run_kv_store_fp8(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off,
                "kv_store_fp8_k_d",
            );
            self.run_kv_store_fp8(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off,
                "kv_store_fp8_v_d",
            );
        } else if self.kv_cache.is_fp16() {
            let kv_elements = cfg.n_kv_heads * cfg.head_dim;
            self.run_kv_copy_fp16(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off,
                "kv_copy_fp16_k_d",
            );
            self.run_kv_copy_fp16(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off,
                "kv_copy_fp16_v_d",
            );
        } else {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(row_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
            }
        }
        // Barrier covers either upstream (transfer or compute write).
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&kv_bar), &[], &[],
            );
        }

        // Attention
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        // Stops here regardless — caller wanted attn_out.
        let _ = halt; // attention is the only further halt point in this partial dispatcher
    }

    /// Debug helper — run ONE layer, returning the post-layer
    /// activations as `Vec<f32>`. Each call submits its own command
    /// buffer; layers must be invoked in order, with `position`
    /// constant (we're tracing a single token through all layers).
    pub fn forward_layer_debug(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Phase-2C debug: stage input via scratch_a, single layer
        // dispatch, read scratch_b. We rebuild the descriptor pool
        // (small) from a single big reset.
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
            // Readback barrier so the next host map sees scratch_b.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[post],
                    &[],
                );
            }
        })?;
        // scratch_b is GpuOnly — we can't read it. Need to stage.
        // Hack: alloc a host-visible read buf on the fly and copy.
        Ok(self.read_scratch_b(dev, cmd_ctx)?)
    }

    /// Stage scratch_b (GpuOnly) into a host-visible buffer and read
    /// it. Per-call alloc — for debug only.
    fn read_scratch_b(
        &self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Use the host-visible scratch_a as the readback target —
        // it's the same size as scratch_b. We don't need scratch_a's
        // content after this call (caller writes a fresh input next
        // round).
        let bytes = (self.config.hidden_dim as u64) * 4;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            // Pre-barrier: previous SHADER_WRITE → TRANSFER_READ
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes);
            dev.device.cmd_copy_buffer(
                cmd,
                self.cur().scratch_b.handle,
                self.cur().scratch_a.handle,
                std::slice::from_ref(&copy),
            );
            // Post-barrier: TRANSFER_WRITE → HOST_READ
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes_slice = self.cur().scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes_slice[..(self.config.hidden_dim as usize) * 4],
        )
        .to_vec())
    }
}

/// Sprint 43D-1 diagnose helper — env-gated logit-distribution dump.
/// Set `VF_LOGIT_DUMP=1` (or `=2` for verbose) to print per-step
/// argmax / top5 / mean / min / max / NaN-count to stderr during
/// `wait_and_read_logits`. No-op otherwise. Used to bisect whether
/// the Gemma-4 `<pad>`-argmax-collapse is a coherent distribution
/// gone wrong, an all-NaN forward, or a particular logit slot
/// dominating from a soft-cap clipping artifact.
pub(super) fn maybe_dump_logits(tag: &str, logits: &[f32]) {
    let level: u32 = std::env::var("VF_LOGIT_DUMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if level == 0 {
        return;
    }
    let n = logits.len();
    let mut max_v = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut sum: f64 = 0.0;
    let mut nan_count: u32 = 0;
    let mut inf_count: u32 = 0;
    let mut max_idx: usize = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        sum += v as f64;
        if v > max_v {
            max_v = v;
            max_idx = i;
        }
        if v < min_v {
            min_v = v;
        }
    }
    let valid = (n as u32) - nan_count - inf_count;
    let mean = if valid > 0 { sum / (valid as f64) } else { 0.0 };

    // Top-5 by descending logit value (NaN/Inf-skipped).
    let mut sorted: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(usize, f32)> = sorted.into_iter().take(5).collect();

    eprintln!(
        "[VF_LOGIT_DUMP] {tag}: n={n} argmax_idx={max_idx} max={max_v:.4} \
         min={min_v:.4} mean={mean:.4} nan={nan_count} inf={inf_count}"
    );
    eprintln!("[VF_LOGIT_DUMP] {tag}: top5={top5:?}");
    if level >= 2 {
        // Verbose: dump first 16 + last 16 logits to spot patterns
        // (e.g. id 0..15 mostly zero suggests pad-collapse artifact).
        let head: Vec<(usize, f32)> = logits.iter().enumerate().take(16).map(|(i, &v)| (i, v)).collect();
        let tail_start = n.saturating_sub(16);
        let tail: Vec<(usize, f32)> = logits[tail_start..].iter().enumerate()
            .map(|(i, &v)| (tail_start + i, v)).collect();
        eprintln!("[VF_LOGIT_DUMP] {tag}: head16={head:?}");
        eprintln!("[VF_LOGIT_DUMP] {tag}: tail16={tail:?}");
    }
}

/// Sprint 43F Block A — dump stats of `hidden_staging` contents
/// (= hidden_norm post-final-RMSNorm, copied via the in-CB
/// VF_FINAL_NORM_DUMP block). Always-on print when called; gating
/// by env happens in the call sites.
pub(super) fn maybe_dump_hidden_staging(tag: &str, hidden: &[f32]) {
    let mut nan = 0u32; let mut inf = 0u32;
    let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
    let mut sum: f64 = 0.0;
    for &v in hidden {
        if v.is_nan() { nan += 1; continue; }
        if v.is_infinite() { inf += 1; continue; }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    let valid = (hidden.len() as u32) - nan - inf;
    let mean = if valid > 0 { sum / (valid as f64) } else { 0.0 };
    eprintln!(
        "[MAIN-CB-DUMP] {tag}: n={} nan={} inf={} min={:.4} max={:.4} mean={:.4}",
        hidden.len(), nan, inf, mn, mx, mean,
    );
    let head: Vec<f32> = hidden.iter().take(8).copied().collect();
    eprintln!("[MAIN-CB-DUMP] {tag}: first8 = {:?}", head);
}
