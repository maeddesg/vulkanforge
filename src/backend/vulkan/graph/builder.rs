//! GraphBuilder — converts `LayerPlan` (Vec<LayerStep>) to a [`VulkanGraph`].
//!
//! Sprint SG-1.2 deliverable. Status:
//!  * **Exhaustive `match` over LayerStep** (compile-time enforcement per
//!    coding-standards §4.4 — adding a variant breaks compilation here
//!    until handled).
//!  * **Real byte-range read/write metadata** for the Qwen3-8B subset
//!    of variants (~16 in `build_qwen3_layer`). This is what makes
//!    SG-1.2 functionally useful: [`super::VulkanGraph::resolve_dependencies`]
//!    can already discover RAW/WAW edges over this metadata even though
//!    Recorder-side dispatch (SG-1.3) isn't wired yet.
//!  * **Other arches (Qwen3.6 SSM, Gemma-4 MoE, Gemma-4 KV-share)** are
//!    `unimplemented!()` stubs. Adding them is SG-1.4+ work and is
//!    intentionally not in this commit's scope.
//!
//! Pipeline handles and push-constant payloads are intentionally left
//! as placeholders (`vk::Pipeline::null()`, empty `Vec<u8>`) in this
//! commit; the Recorder (SG-1.3) will look up the real values from the
//! `PipelineRegistry` and serialize the right push-const struct per
//! node-op. Putting the lookup logic in the Builder right now would
//! duplicate ~30 helpers from `forward/runs.rs` for zero functional
//! gain — the byte-range graph already drives the dependency-pass work
//! that motivates the whole rewrite (see
//! `results/vulkan_graph_analysis_part4.md` §2).

use ash::vk;

use crate::backend::vulkan::forward::layer_plan::{
    ActivationKind, LayerPlan, LayerStep,
};
use crate::backend::vulkan::gguf::ModelConfig;

use super::node::{
    Binding, BufferHandle, ByteRange, DispatchNode, MemAccess, NodeId,
};
use super::VulkanGraph;

/// Per-decode-token graph builder.
///
/// Holds:
///  * the in-flight [`VulkanGraph`],
///  * a [`BufferMap`] of `vk::Buffer` handles the Builder needs to
///    reference (slot buffers + persistent state). Caller fills this
///    once and reuses across all token-builds.
///  * `cfg` for shape parameters (hidden_dim, head_dim, ffn_dim, etc.).
pub struct GraphBuilder<'a> {
    graph: VulkanGraph,
    bufs: &'a BufferMap,
    cfg: &'a ModelConfig,
    /// Decode position (used by `Attention` for KV-cache range).
    position: u32,
    /// Per-layer step counter — the index of the next `LayerStep` within
    /// the current layer's plan. Incremented exactly once per `add_step`
    /// call so each `DispatchNode.step_index_in_layer` reflects its plan
    /// position. Reset to 0 when a new layer begins (see
    /// [`Self::begin_layer`]).
    current_step_in_layer: u32,
}

/// Resolved set of `vk::Buffer` handles the Builder threads into
/// nodes. Caller (SG-1.3 integration code) populates this from
/// `Forward::cur()` plus the optional SSM-state buffers; the Builder
/// itself never reaches into `Forward` so it stays trivially testable.
#[derive(Debug, Clone, Copy)]
pub struct BufferMap {
    pub scratch_a: BufferHandle,
    pub hidden_norm: BufferHandle,
    pub q_buf: BufferHandle,
    pub k_buf: BufferHandle,
    pub v_buf: BufferHandle,
    pub attn_out: BufferHandle,
    pub o_buf: BufferHandle,
    pub res1: BufferHandle,
    pub gate_buf: BufferHandle,
    pub up_buf: BufferHandle,
    pub ffn_hidden: BufferHandle,
    pub ffn_out: BufferHandle,
    /// Next-layer input / final output (set by Recorder to point at
    /// the right slot's `scratch_a` for the next layer in chain).
    pub layer_output: BufferHandle,
    /// KV-cache slab (shared across all layers via per-layer offsets).
    pub kv_cache: BufferHandle,

    // ── Sprint SG-1.4 — Qwen3.6 SSM / Q-Gate buffers ──────────────
    //
    // All `Forward::ssm_*_buf` are `Option<GpuBuffer>` (None on non-
    // qwen35 builds). The Recorder unpacks `.handle` when present and
    // leaves these as `vk::Buffer::null()` otherwise. The graph
    // builder doesn't care — non-qwen35 plans never use SSM variants
    // and the dep-resolver naturally skips null-vs-null overlaps via
    // distinct buffer identity (but won't generate noise from them
    // either; null handles compare equal but the per-step helpers
    // only touch SSM bufs when emitted by qwen35-only LayerStep
    // variants).
    pub ssm_qkv: BufferHandle,
    pub ssm_z: BufferHandle,
    pub ssm_beta: BufferHandle,
    pub ssm_alpha: BufferHandle,
    pub ssm_gate: BufferHandle,
    pub ssm_conv_input: BufferHandle,
    pub ssm_conv_output: BufferHandle,
    pub ssm_qrep: BufferHandle,
    pub ssm_krep: BufferHandle,
    pub ssm_gdn_out: BufferHandle,
    pub ssm_norm_out: BufferHandle,
    pub ssm_state: BufferHandle,
    pub conv_state: BufferHandle,
}

impl BufferMap {
    /// All-null placeholder. Used in unit tests + the Phase-1 stub
    /// path where we only care about graph structure / dependencies.
    pub fn null() -> Self {
        Self {
            scratch_a: vk::Buffer::null(),
            hidden_norm: vk::Buffer::null(),
            q_buf: vk::Buffer::null(),
            k_buf: vk::Buffer::null(),
            v_buf: vk::Buffer::null(),
            attn_out: vk::Buffer::null(),
            o_buf: vk::Buffer::null(),
            res1: vk::Buffer::null(),
            gate_buf: vk::Buffer::null(),
            up_buf: vk::Buffer::null(),
            ffn_hidden: vk::Buffer::null(),
            ffn_out: vk::Buffer::null(),
            layer_output: vk::Buffer::null(),
            kv_cache: vk::Buffer::null(),
            ssm_qkv: vk::Buffer::null(),
            ssm_z: vk::Buffer::null(),
            ssm_beta: vk::Buffer::null(),
            ssm_alpha: vk::Buffer::null(),
            ssm_gate: vk::Buffer::null(),
            ssm_conv_input: vk::Buffer::null(),
            ssm_conv_output: vk::Buffer::null(),
            ssm_qrep: vk::Buffer::null(),
            ssm_krep: vk::Buffer::null(),
            ssm_gdn_out: vk::Buffer::null(),
            ssm_norm_out: vk::Buffer::null(),
            ssm_state: vk::Buffer::null(),
            conv_state: vk::Buffer::null(),
        }
    }
}

impl<'a> GraphBuilder<'a> {
    /// Walk all `n_layers` plans and emit graph nodes. Resolves
    /// dependencies + sorts before returning. Pure CPU work; no
    /// Vulkan calls.
    pub fn build_decode_token(
        bufs: &'a BufferMap,
        cfg: &'a ModelConfig,
        position: u32,
        per_layer_plans: &[LayerPlan],
    ) -> VulkanGraph {
        let estimated = per_layer_plans.iter().map(|p| p.len()).sum::<usize>() + 4;
        let mut builder = Self {
            graph: VulkanGraph::with_capacity(estimated),
            bufs,
            cfg,
            position,
            current_step_in_layer: 0,
        };
        for (layer, plan) in per_layer_plans.iter().enumerate() {
            builder.current_step_in_layer = 0;
            for step in plan {
                builder.add_step(step, layer as u32);
                builder.current_step_in_layer += 1;
            }
        }
        // lm_head is an architecture-independent terminal node; left
        // to SG-1.3 to add since it depends on `output.weight` lookup.

        let mut graph = builder.graph;
        graph.resolve_dependencies();
        graph.sort();
        graph
    }

    /// Sprint SG-1.3 — build a graph for a single layer's plan only.
    /// The Recorder integration point in `forward::executor` lives at
    /// the per-layer granularity (`execute_layer`), so the Builder
    /// also offers per-layer construction. Each `DispatchNode` carries
    /// its `step_index_in_layer` so the Recorder can index back into
    /// the original `LayerPlan` and call `DecodeExec::execute_step`.
    pub fn build_per_layer(
        bufs: &'a BufferMap,
        cfg: &'a ModelConfig,
        layer: u32,
        plan: &LayerPlan,
    ) -> VulkanGraph {
        let mut builder = Self {
            graph: VulkanGraph::with_capacity(plan.len()),
            bufs,
            cfg,
            position: 0,
            current_step_in_layer: 0,
        };
        for step in plan {
            builder.add_step(step, layer);
            builder.current_step_in_layer += 1;
        }
        let mut graph = builder.graph;
        graph.resolve_dependencies();
        graph.sort();
        graph
    }

    fn add_step(&mut self, step: &LayerStep, layer: u32) {
        match step {
            // ===== Qwen3-8B / Llama standard attention path =====
            LayerStep::AttnNorm => self.add_rms_norm(
                layer, self.bufs.scratch_a, self.bufs.hidden_norm,
                self.cfg.hidden_dim, 1, "attn_norm",
            ),
            LayerStep::QProj => self.add_gemv(
                layer, self.bufs.hidden_norm, self.bufs.q_buf,
                self.cfg.hidden_dim, self.cfg.n_heads * self.cfg.head_dim,
                "q_proj",
            ),
            LayerStep::KProj => self.add_gemv(
                layer, self.bufs.hidden_norm, self.bufs.k_buf,
                self.cfg.hidden_dim, self.cfg.n_kv_heads * self.cfg.head_dim,
                "k_proj",
            ),
            LayerStep::VProj => self.add_gemv(
                layer, self.bufs.hidden_norm, self.bufs.v_buf,
                self.cfg.hidden_dim, self.cfg.n_kv_heads * self.cfg.head_dim,
                "v_proj",
            ),
            LayerStep::QBiasAdd => self.add_inplace_bias_add(
                layer, self.bufs.q_buf,
                self.cfg.n_heads * self.cfg.head_dim, "q_bias",
            ),
            LayerStep::KBiasAdd => self.add_inplace_bias_add(
                layer, self.bufs.k_buf,
                self.cfg.n_kv_heads * self.cfg.head_dim, "k_bias",
            ),
            LayerStep::VBiasAdd => self.add_inplace_bias_add(
                layer, self.bufs.v_buf,
                self.cfg.n_kv_heads * self.cfg.head_dim, "v_bias",
            ),
            LayerStep::QNormRope { .. } => self.add_inplace(
                layer, self.bufs.q_buf,
                self.cfg.n_heads * self.cfg.head_dim, "q_norm_rope",
            ),
            LayerStep::KNormRope { .. } => self.add_inplace(
                layer, self.bufs.k_buf,
                self.cfg.n_kv_heads * self.cfg.head_dim, "k_norm_rope",
            ),
            LayerStep::QRope { .. } => self.add_inplace(
                layer, self.bufs.q_buf,
                self.cfg.n_heads * self.cfg.head_dim, "q_rope",
            ),
            LayerStep::KRope { .. } => self.add_inplace(
                layer, self.bufs.k_buf,
                self.cfg.n_kv_heads * self.cfg.head_dim, "k_rope",
            ),
            LayerStep::KvWrite => self.add_kv_write(layer),
            LayerStep::Attention { kv_layer, kv_start } => self.add_attention(
                layer, *kv_layer, *kv_start,
            ),
            LayerStep::OProj => self.add_gemv(
                layer, self.bufs.attn_out, self.bufs.o_buf,
                self.cfg.n_heads * self.cfg.head_dim, self.cfg.hidden_dim,
                "o_proj",
            ),
            LayerStep::AttnResidualAdd => {
                // Llama/Qwen3 paths feed o_buf into the add; Qwen3.6
                // SSM layers route via attn_out (written by
                // SsmOutProj) instead. Read BOTH so the byte-range
                // graph picks up the right RAW edge regardless of
                // architecture — `execute_step` dispatches the
                // correct buffer either way.
                let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
                self.graph.add_dispatch(DispatchNode {
                    id: 0,
                    step_index_in_layer: self.current_step_in_layer,
                    pipeline: vk::Pipeline::null(),
                    pipeline_layout: vk::PipelineLayout::null(),
                    descriptor_set_layout: vk::DescriptorSetLayout::null(),
                    bindings: Vec::new(),
                    push_constants: Vec::new(),
                    dispatch: (self.cfg.hidden_dim.div_ceil(64), 1, 1),
                    reads: vec![
                        MemAccess::new(self.bufs.scratch_a, 0, hidden_bytes),
                        MemAccess::new(self.bufs.o_buf, 0, hidden_bytes),
                        MemAccess::new(self.bufs.attn_out, 0, hidden_bytes),
                    ],
                    writes: vec![MemAccess::new(self.bufs.res1, 0, hidden_bytes)],
                    layer,
                    label: Some(format!("L{layer}_attn_residual_add")),
                });
            }
            LayerStep::PreFfnNorm => self.add_rms_norm(
                layer, self.bufs.res1, self.bufs.hidden_norm,
                self.cfg.hidden_dim, 1, "pre_ffn_norm",
            ),
            LayerStep::GateProj => self.add_gemv(
                layer, self.bufs.hidden_norm, self.bufs.gate_buf,
                self.cfg.hidden_dim, self.cfg.ffn_dim, "gate_proj",
            ),
            LayerStep::UpProj => self.add_gemv(
                layer, self.bufs.hidden_norm, self.bufs.up_buf,
                self.cfg.hidden_dim, self.cfg.ffn_dim, "up_proj",
            ),
            LayerStep::Activation { kind } => self.add_activation(layer, *kind),
            LayerStep::DownProj => self.add_gemv(
                layer, self.bufs.ffn_hidden, self.bufs.ffn_out,
                self.cfg.ffn_dim, self.cfg.hidden_dim, "down_proj",
            ),
            LayerStep::FfnResidualAdd => self.add_binary(
                layer, self.bufs.res1, self.bufs.ffn_out, self.bufs.layer_output,
                self.cfg.hidden_dim, "ffn_residual_add",
            ),

            // ===== Qwen3.6 Linear-Attention (SG-1.4) =====
            //
            // Shapes derived from qwen35 spec:
            //   conv_channels = 2*(d_state*n_group) + d_inner
            //                 = 2*(128*16) + 6144 = 10240
            //   Q slice: [   0 ..  8192) bytes = 2048 floats (16×128)
            //   K slice: [8192 .. 16384) bytes = 2048 floats
            //   V slice: [16384 .. 40960) bytes = 6144 floats (48×128)
            //   d_inner = 6144 = ssm_z_buf, ssm_qrep_buf, ssm_krep_buf,
            //                    ssm_norm_out_buf  (= 24576 bytes each)
            //   d_state = 128, n_v_heads = 48 → ssm_beta/alpha/gate = 48 f32 = 192 B
            //
            // Each helper emits one DispatchNode whose reads/writes
            // describe the whole step body (which the imperative
            // `execute_step` actually issues). The byte-range graph
            // is what SG-2's barrier pass keys on; the dispatch is
            // unchanged.
            LayerStep::AttnQkvProj { .. } => self.add_ssm_qkv_proj(layer),
            LayerStep::AttnGateZProj { .. } => self.add_ssm_gate_z_proj(layer),
            LayerStep::SsmBetaProj { .. } => self.add_ssm_beta_proj(layer),
            LayerStep::SsmAlphaGate { .. } => self.add_ssm_alpha_gate(layer),
            LayerStep::SsmConv1d { .. } => self.add_ssm_conv1d(layer),
            LayerStep::SsmSilu { .. } => self.add_ssm_silu(layer),
            LayerStep::SsmQkL2Norm { .. } => self.add_ssm_qk_l2_norm(layer),
            LayerStep::SsmRepeatQK { .. } => self.add_ssm_repeat_qk(layer),
            LayerStep::GatedDeltaNet { .. } => self.add_gated_delta_net(layer),
            LayerStep::NormGated { .. } => self.add_norm_gated(layer),
            LayerStep::SsmOutProj { .. } => self.add_ssm_out_proj(layer),
            LayerStep::AttnQGateProj { q_dim } => self.add_attn_q_gate_proj(layer, *q_dim),
            LayerStep::AttnGatedOutput { q_dim } => self.add_attn_gated_output(layer, *q_dim),
            LayerStep::ResidualIdentitySeed => self.add_residual_identity_seed(layer),

            // ===== Gemma-4 KV-share + 4-norm + PLE (SG-1.5 — stubbed) =====
            LayerStep::VFromKRaw
            | LayerStep::VNorm
            | LayerStep::PostAttnNorm
            | LayerStep::PostFfnNorm
            | LayerStep::PleBlock
            | LayerStep::LayerScalarMul => {
                unimplemented!(
                    "SG-1.5 — Gemma-4 variant {step:?} not yet implemented \
                     in GraphBuilder. Use the imperative executor."
                );
            }

            // ===== Gemma-4-26B MoE (SG-1.6 — stubbed) =====
            LayerStep::PostDenseMlpNorm
            | LayerStep::PreMoeNorm
            | LayerStep::MoeRoute { .. }
            | LayerStep::MoeExpertFfn { .. }
            | LayerStep::PostMoeNorm
            | LayerStep::MoeBranchAdd => {
                unimplemented!(
                    "SG-1.6 — Gemma-4 MoE variant {step:?} not yet \
                     implemented in GraphBuilder. Use the imperative \
                     executor (the analysis-Teil-4 lever for Gemma-4 is \
                     here — biggest VF Decode-Gap. SG-1.6 ist die \
                     ranghöchste Folge-Arbeit nach SG-1.3-ship)."
                );
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Per-step helpers (Qwen3-8B subset).
    //
    // Each helper builds a DispatchNode with real reads/writes
    // byte-ranges + dispatch dims. Pipeline/bindings/push-constants
    // are placeholders — SG-1.3 Recorder resolves them at record time
    // by re-using existing forward/runs.rs helpers keyed on the node's
    // `label`. This keeps the Builder side trivially unit-testable
    // (no PipelineRegistry / Forward dependency).
    // ────────────────────────────────────────────────────────────────

    fn add_rms_norm(
        &mut self, layer: u32,
        input: BufferHandle, output: BufferHandle,
        cols: u32, rows: u32,
        label: &str,
    ) {
        let bytes = (cols as u64) * (rows as u64) * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, input, 0, bytes),
                Binding::new(1, output, 0, bytes),
            ],
            push_constants: Vec::new(),
            dispatch: (rows, 1, 1),
            reads: vec![MemAccess::new(input, 0, bytes), MemAccess::whole(input)],
            writes: vec![MemAccess::new(output, 0, bytes)],
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    fn add_gemv(
        &mut self, layer: u32,
        input: BufferHandle, output: BufferHandle,
        k: u32, m: u32,
        label: &str,
    ) {
        let input_bytes = (k as u64) * 4;
        let output_bytes = (m as u64) * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, input, 0, input_bytes),
                Binding::new(1, output, 0, output_bytes),
            ],
            push_constants: Vec::new(),
            // Standard one-WG-per-row for K-quant subgroup variants
            // (see forward/runs.rs::run_gemv MMV_NUM_ROWS=1).
            dispatch: (m, 1, 1),
            reads: vec![MemAccess::new(input, 0, input_bytes)],
            writes: vec![MemAccess::new(output, 0, output_bytes)],
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    fn add_inplace_bias_add(
        &mut self, layer: u32,
        target: BufferHandle, dim: u32,
        label: &str,
    ) {
        let bytes = (dim as u64) * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![Binding::new(0, target, 0, bytes)],
            push_constants: Vec::new(),
            dispatch: (dim.div_ceil(64), 1, 1),
            reads: vec![MemAccess::new(target, 0, bytes)],
            writes: vec![MemAccess::new(target, 0, bytes)],
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    fn add_inplace(
        &mut self, layer: u32,
        target: BufferHandle, dim: u32,
        label: &str,
    ) {
        // Same shape as bias_add — different shader. Q/K norm-rope
        // are RAW on the target (read, transform, write back).
        self.add_inplace_bias_add(layer, target, dim, label)
    }

    fn add_kv_write(&mut self, layer: u32) {
        // Writes a single (k, v) pair into kv_cache at offset
        // determined by `layer * per_layer_stride + position * head_stride`.
        // For SG-1.2 we model this as a WAW on the whole kv_cache (since
        // VF's actual offset math needs per-layer stride from kv_cache_config).
        // The byte-range coarseness is fine here: kv_cache is written
        // exactly once per (layer, position) so over-conservatism doesn't
        // hurt dependency discovery.
        let k_bytes = (self.cfg.n_kv_heads as u64) * (self.cfg.head_dim as u64) * 4;
        let v_bytes = k_bytes;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, self.bufs.k_buf, 0, k_bytes),
                Binding::new(1, self.bufs.v_buf, 0, v_bytes),
                Binding::new(2, self.bufs.kv_cache, 0, 0),
            ],
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.k_buf, 0, k_bytes),
                MemAccess::new(self.bufs.v_buf, 0, v_bytes),
            ],
            writes: vec![
                // Whole-buffer write is intentional in SG-1.2 — exact
                // offset/size computation lives in SG-1.3 Recorder.
                MemAccess::whole(self.bufs.kv_cache),
            ],
            layer,
            label: Some(format!("L{layer}_kv_write")),
        });
    }

    fn add_attention(&mut self, layer: u32, _kv_layer: u32, _kv_start: u32) {
        let q_bytes = (self.cfg.n_heads as u64) * (self.cfg.head_dim as u64) * 4;
        let attn_bytes = q_bytes;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, self.bufs.q_buf, 0, q_bytes),
                Binding::new(1, self.bufs.kv_cache, 0, 0),
                Binding::new(2, self.bufs.attn_out, 0, attn_bytes),
            ],
            push_constants: Vec::new(),
            dispatch: (self.cfg.n_heads, 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.q_buf, 0, q_bytes),
                MemAccess::whole(self.bufs.kv_cache),
            ],
            writes: vec![MemAccess::new(self.bufs.attn_out, 0, attn_bytes)],
            layer,
            label: Some(format!("L{layer}_attention@pos{}", self.position)),
        });
    }

    fn add_activation(&mut self, layer: u32, kind: ActivationKind) {
        // Reads gate_buf + up_buf, writes ffn_hidden (size = ffn_dim).
        let ffn_bytes = (self.cfg.ffn_dim as u64) * 4;
        let label = match kind {
            ActivationKind::SwiGlu => "swiglu",
            ActivationKind::GeluPytorchTanhGlu => "gelu_pt_glu",
        };
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, self.bufs.gate_buf, 0, ffn_bytes),
                Binding::new(1, self.bufs.up_buf, 0, ffn_bytes),
                Binding::new(2, self.bufs.ffn_hidden, 0, ffn_bytes),
            ],
            push_constants: Vec::new(),
            dispatch: (self.cfg.ffn_dim.div_ceil(64), 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.gate_buf, 0, ffn_bytes),
                MemAccess::new(self.bufs.up_buf, 0, ffn_bytes),
            ],
            writes: vec![MemAccess::new(self.bufs.ffn_hidden, 0, ffn_bytes)],
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    fn add_binary(
        &mut self, layer: u32,
        a: BufferHandle, b: BufferHandle, out: BufferHandle,
        ne: u32, label: &str,
    ) {
        let bytes = (ne as u64) * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: vec![
                Binding::new(0, a, 0, bytes),
                Binding::new(1, b, 0, bytes),
                Binding::new(2, out, 0, bytes),
            ],
            push_constants: Vec::new(),
            dispatch: (ne.div_ceil(64), 1, 1),
            reads: vec![
                MemAccess::new(a, 0, bytes),
                MemAccess::new(b, 0, bytes),
            ],
            writes: vec![MemAccess::new(out, 0, bytes)],
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    // ─────────────────────────────────────────────────────────────────
    // Sprint SG-1.4 — Qwen3.6 SSM / Q-Gate variant helpers.
    //
    // Each helper emits one DispatchNode whose `reads` / `writes`
    // describe the byte-ranges touched by the corresponding
    // `executor::attention::step_*` body (multi-dispatch bodies are
    // modelled as a single graph node so step_index_in_layer remains
    // 1:1 with the layer plan — `execute_step` dispatches the whole
    // body in one call). For multi-buffer steps with disjoint slice
    // accesses (L2Norm Q vs K), we emit precise byte-ranges so the
    // SG-2 barrier pass can later see that disjoint writes don't
    // require RAW sync.
    //
    // GatedDeltaNet also emits a TransferNode for the persistent SSM
    // state copy-back (cmd_copy_buffer issued from inside the step).
    // ─────────────────────────────────────────────────────────────────

    // Shape constants. Pulled out so per-helper code stays readable;
    // these match Qwen35Spec used by `executor::attention`.
    const D_INNER: u64 = 6144;           // V projection / SSM hidden
    const N_GROUP_KV: u64 = 16;          // Q + K groups
    const D_STATE: u64 = 128;
    const N_V_HEADS: u64 = 48;           // V heads / β / α head count
    const D_HEAD_V: u64 = 128;
    const D_CONV: u64 = 4;

    // 16 × 128 × 4 = 8192  (Q + K slice each)
    const QK_SLICE_BYTES: u64 = Self::N_GROUP_KV * Self::D_STATE * 4;
    // 4096 + 6144 = 10240 floats × 4 = 40960 bytes
    const CONV_CHANNELS_BYTES: u64 =
        (2 * Self::N_GROUP_KV * Self::D_STATE + Self::D_INNER) * 4;
    // V slice = conv_channels - 2*Q = 24576 bytes
    const V_SLICE_BYTES: u64 = Self::CONV_CHANNELS_BYTES - 2 * Self::QK_SLICE_BYTES;
    // d_inner × 4 = 24576 bytes  (ssm_z_buf, ssm_qrep_buf, etc.)
    const D_INNER_BYTES: u64 = Self::D_INNER * 4;
    // n_v_heads × 4 = 192 bytes (ssm_beta, ssm_alpha, ssm_gate)
    const N_V_HEADS_BYTES: u64 = Self::N_V_HEADS * 4;
    // GDN output: d_inner floats of output + state copy region (which
    // is variable per layer). We model the output region precisely
    // and use whole() for the state copy back.

    fn ssm_simple_node(
        &mut self,
        layer: u32,
        reads: Vec<MemAccess>,
        writes: Vec<MemAccess>,
        label: &str,
    ) {
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads,
            writes,
            layer,
            label: Some(format!("L{layer}_{label}")),
        });
    }

    fn add_ssm_qkv_proj(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.ssm_qkv, 0, Self::CONV_CHANNELS_BYTES)],
            "ssm_qkv_proj",
        );
    }

    fn add_ssm_gate_z_proj(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES)],
            "ssm_gate_z_proj",
        );
    }

    fn add_ssm_beta_proj(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES)],
            "ssm_beta_proj",
        );
    }

    fn add_ssm_alpha_gate(&mut self, layer: u32) {
        // Composite: GEMV → alpha; add dt.bias → alpha; softplus → alpha;
        // × ssm_a → gate. All within one step_attn_alpha_gate body.
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![
                MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES),
                MemAccess::new(self.bufs.ssm_gate, 0, Self::N_V_HEADS_BYTES),
            ],
            "ssm_alpha_gate",
        );
    }

    fn add_ssm_conv1d(&mut self, layer: u32) {
        // Reads: ssm_qkv (current input) + conv_state (rolling window).
        // Writes: ssm_conv_input (built window), ssm_conv_output, and
        // updates conv_state in place. Per-layer offsets into the
        // state buffer aren't material here — whole() suffices.
        self.ssm_simple_node(layer,
            vec![
                MemAccess::new(self.bufs.ssm_qkv, 0, Self::CONV_CHANNELS_BYTES),
                MemAccess::whole(self.bufs.conv_state),
            ],
            vec![
                MemAccess::new(self.bufs.ssm_conv_input, 0,
                    Self::CONV_CHANNELS_BYTES * Self::D_CONV),
                MemAccess::new(self.bufs.ssm_conv_output, 0, Self::CONV_CHANNELS_BYTES),
                MemAccess::whole(self.bufs.conv_state),
            ],
            "ssm_conv1d",
        );
    }

    fn add_ssm_silu(&mut self, layer: u32) {
        // In-place SiLU on the whole conv_output buffer.
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.ssm_conv_output, 0, Self::CONV_CHANNELS_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_conv_output, 0, Self::CONV_CHANNELS_BYTES)],
            "ssm_silu",
        );
    }

    fn add_ssm_qk_l2_norm(&mut self, layer: u32) {
        // Disjoint Q + K slices in conv_output. Two `MemAccess`
        // entries for both reads and writes — the SG-2 barrier pass
        // sees that the K slice's overlap-with-prior-writes is
        // independent from the Q slice's, allowing per-slice
        // resolution in the future.
        self.ssm_simple_node(layer,
            vec![
                MemAccess::new(self.bufs.ssm_conv_output, 0,                       Self::QK_SLICE_BYTES),
                MemAccess::new(self.bufs.ssm_conv_output, Self::QK_SLICE_BYTES,    Self::QK_SLICE_BYTES),
            ],
            vec![
                MemAccess::new(self.bufs.ssm_conv_output, 0,                       Self::QK_SLICE_BYTES),
                MemAccess::new(self.bufs.ssm_conv_output, Self::QK_SLICE_BYTES,    Self::QK_SLICE_BYTES),
            ],
            "ssm_qk_l2_norm",
        );
    }

    fn add_ssm_repeat_qk(&mut self, layer: u32) {
        // 16 → 48 head repeat (n_group_kv → n_v_heads). Reads Q+K
        // slices in conv_output, writes ssm_qrep + ssm_krep.
        self.ssm_simple_node(layer,
            vec![
                MemAccess::new(self.bufs.ssm_conv_output, 0,                       Self::QK_SLICE_BYTES),
                MemAccess::new(self.bufs.ssm_conv_output, Self::QK_SLICE_BYTES,    Self::QK_SLICE_BYTES),
            ],
            vec![
                MemAccess::new(self.bufs.ssm_qrep, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_krep, 0, Self::D_INNER_BYTES),
            ],
            "ssm_repeat_qk",
        );
    }

    fn add_gated_delta_net(&mut self, layer: u32) {
        // GDN: dispatch (reads many SSM bufs + ssm_state, writes
        // ssm_gdn_out) + cmd_copy_buffer (reads ssm_gdn_out state
        // region, writes ssm_state). We model the copy as a separate
        // TransferNode so SG-2 sees both deps.
        let gdn_dispatch_id = self.graph.add_dispatch(DispatchNode {
            id: 0,
            step_index_in_layer: self.current_step_in_layer,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.ssm_qrep, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_krep, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_conv_output,
                    2 * Self::QK_SLICE_BYTES, Self::V_SLICE_BYTES),  // V slice
                MemAccess::new(self.bufs.ssm_gate, 0, Self::N_V_HEADS_BYTES),
                MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES),
                MemAccess::whole(self.bufs.ssm_state),
            ],
            writes: vec![MemAccess::whole(self.bufs.ssm_gdn_out)],
            layer,
            label: Some(format!("L{layer}_gdn")),
        });
        // State copy-back as a Transfer node.
        let _copy_id = self.graph.add_transfer(super::node::TransferNode {
            id: 0,
            src_buffer: self.bufs.ssm_gdn_out,
            src_offset: 0,
            dst_buffer: self.bufs.ssm_state,
            dst_offset: 0,
            size: 0, // exact per-layer offsets/sizes are runtime;
                     // whole() in reads/writes below covers it for the
                     // SG-2 dep-pass.
            reads:  vec![MemAccess::whole(self.bufs.ssm_gdn_out)],
            writes: vec![MemAccess::whole(self.bufs.ssm_state)],
            layer,
            label: Some(format!("L{layer}_gdn_state_copy")),
        });
        let _ = gdn_dispatch_id;
    }

    fn add_norm_gated(&mut self, layer: u32) {
        // RMSNorm + SiLU + Mul → ssm_norm_out.
        //
        // SG-1.4-e — in-place write audit: step_norm_gated's middle
        // sub-dispatch is `run_silu(z, z, ...)` which writes ssm_z
        // in-place (executor/attention.rs:1277-1281). Without listing
        // ssm_z as a write, the graph couldn't see the WAW that lets
        // a later consumer of ssm_z (e.g. the NEXT layer's
        // NormGated, after the inter-layer drain) observe the
        // silu'd value.
        self.ssm_simple_node(layer,
            vec![
                MemAccess::whole(self.bufs.ssm_gdn_out),  // output portion
                MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES),
            ],
            vec![
                MemAccess::new(self.bufs.ssm_norm_out, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES),  // SiLU in-place
            ],
            "norm_gated",
        );
    }

    fn add_ssm_out_proj(&mut self, layer: u32) {
        // SG-1.4-c — corrected from SG-1.4: the actual
        // `step_ssm_out_proj` body writes `o_buf` (NOT `attn_out`) —
        // see `executor/attention.rs:1326`. AttnResidualAdd's
        // dispatcher then consumes `o_buf` as `addend`.
        //
        // Previous SG-1.4 stub wrote `attn_out`, breaking the RAW
        // edge to `AttnResidualAdd` (which reads `o_buf`) and causing
        // topo-order drift + partial-coherence under
        // `BarrierMode::GraphDriven` on Qwen3.6 (the "0language.com:"
        // signature in SG-1.4-b §3).
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.ssm_norm_out, 0, Self::D_INNER_BYTES)],
            vec![MemAccess::new(self.bufs.o_buf, 0, hidden_bytes)],
            "ssm_out_proj",
        );
    }

    fn add_attn_q_gate_proj(&mut self, layer: u32, q_dim: u32) {
        // Qwen3.6 Full-Attn: fused Q+Gate GEMV writes q_buf as
        // [Q (q_dim) | Gate (q_dim)] = 2*q_dim floats.
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        let combined_bytes = (q_dim as u64) * 2 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.q_buf, 0, combined_bytes)],
            "attn_q_gate_proj",
        );
    }

    fn add_attn_gated_output(&mut self, layer: u32, q_dim: u32) {
        // In-place sigmoid-gate-multiply on attn_out using gate slice
        // of q_buf at offset q_dim*4.
        let q_bytes = (q_dim as u64) * 4;
        self.ssm_simple_node(layer,
            vec![
                MemAccess::new(self.bufs.attn_out, 0, q_bytes),
                MemAccess::new(self.bufs.q_buf, q_bytes, q_bytes),  // gate slice
            ],
            vec![MemAccess::new(self.bufs.attn_out, 0, q_bytes)],
            "attn_gated_output",
        );
    }

    fn add_residual_identity_seed(&mut self, layer: u32) {
        // Seeds res1 with the layer input (identity / no-op for
        // recurrent layers that don't have a real Q/K/V/Attention
        // sub-block). Modeled as: reads scratch_a, writes res1.
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.ssm_simple_node(layer,
            vec![MemAccess::new(self.bufs.scratch_a, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.res1, 0, hidden_bytes)],
            "residual_identity_seed",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_qwen3_cfg() -> ModelConfig {
        use crate::backend::vulkan::gguf::RopeVariant;
        // Minimal struct construction. We touch only the shape fields
        // the Builder reads; everything else gets a benign default.
        ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 32,
            n_kv_heads: 8,
            hidden_dim: 4096,
            ffn_dim: 12288,
            vocab_size: 151936,
            head_dim: 128,
            rope_freq_base: 1_000_000.0,
            rope_dim: 128,
            rope_variant: RopeVariant::Neox,
            context_length: 2048,
            has_qk_norm: true,
            rms_norm_eps: 1e-6,
            gemma4: None,
            qwen35: None,
        }
    }

    fn fake_buffer(seed: u64) -> BufferHandle {
        use ash::vk::Handle;
        vk::Buffer::from_raw(seed)
    }

    fn distinct_buffer_map() -> BufferMap {
        BufferMap {
            scratch_a: fake_buffer(1),
            hidden_norm: fake_buffer(2),
            q_buf: fake_buffer(3),
            k_buf: fake_buffer(4),
            v_buf: fake_buffer(5),
            attn_out: fake_buffer(6),
            o_buf: fake_buffer(7),
            res1: fake_buffer(8),
            gate_buf: fake_buffer(9),
            up_buf: fake_buffer(10),
            ffn_hidden: fake_buffer(11),
            ffn_out: fake_buffer(12),
            layer_output: fake_buffer(13),
            kv_cache: fake_buffer(14),
            ssm_qkv: fake_buffer(15),
            ssm_z: fake_buffer(16),
            ssm_beta: fake_buffer(17),
            ssm_alpha: fake_buffer(18),
            ssm_gate: fake_buffer(19),
            ssm_conv_input: fake_buffer(20),
            ssm_conv_output: fake_buffer(21),
            ssm_qrep: fake_buffer(22),
            ssm_krep: fake_buffer(23),
            ssm_gdn_out: fake_buffer(24),
            ssm_norm_out: fake_buffer(25),
            ssm_state: fake_buffer(26),
            conv_state: fake_buffer(27),
        }
    }

    fn standard_qwen3_plan() -> LayerPlan {
        vec![
            LayerStep::AttnNorm,
            LayerStep::QProj,
            LayerStep::KProj,
            LayerStep::VProj,
            LayerStep::QNormRope { rotary_dim: 128, freq_base: 1_000_000.0, theta_scale: 1.0 },
            LayerStep::KNormRope { rotary_dim: 128, freq_base: 1_000_000.0, theta_scale: 1.0 },
            LayerStep::KvWrite,
            LayerStep::Attention { kv_layer: 0, kv_start: 0 },
            LayerStep::OProj,
            LayerStep::AttnResidualAdd,
            LayerStep::PreFfnNorm,
            LayerStep::GateProj,
            LayerStep::UpProj,
            LayerStep::Activation { kind: ActivationKind::SwiGlu },
            LayerStep::DownProj,
            LayerStep::FfnResidualAdd,
        ]
    }

    #[test]
    fn builder_emits_one_node_per_qwen3_step() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plan = standard_qwen3_plan();
        let plans = vec![plan.clone(), plan.clone()];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 16 * 2, "16 steps × 2 layers");
    }

    #[test]
    fn builder_records_byte_range_writes_for_q_proj() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::QProj]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);

        assert_eq!(graph.len(), 1);
        let writes = graph.nodes[0].writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].buffer, bufs.q_buf);
        // n_heads × head_dim × 4 bytes = 32 × 128 × 4 = 16384
        assert_eq!(writes[0].range.size, 32 * 128 * 4);
        assert_eq!(writes[0].range.offset, 0);
    }

    #[test]
    fn builder_finds_raw_dependency_q_proj_to_q_norm_rope() {
        // QProj writes q_buf; QNormRope reads+writes q_buf in place.
        // resolve_dependencies should emit at least one RAW edge.
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![
            LayerStep::QProj,
            LayerStep::QNormRope { rotary_dim: 128, freq_base: 1.0, theta_scale: 1.0 },
        ]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);

        assert_eq!(graph.len(), 2);
        // Should find at least one dependency (RAW or WAW) on q_buf.
        let has_q_dep = graph.edges.iter().any(|d| {
            d.from == 0 && d.to == 1 && d.buffer == bufs.q_buf
        });
        assert!(has_q_dep,
            "expected QProj → QNormRope dependency on q_buf, got edges = {:?}",
            graph.edges);
    }

    #[test]
    fn builder_topo_sort_respects_dependencies() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![standard_qwen3_plan()];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);

        assert_eq!(graph.execution_order.len(), graph.len());

        // Verify topological-order property: for every edge, from comes
        // before to in execution_order.
        let pos: std::collections::HashMap<NodeId, usize> = graph
            .execution_order.iter().enumerate()
            .map(|(i, &n)| (n, i))
            .collect();
        for edge in &graph.edges {
            let p_from = pos[&edge.from];
            let p_to = pos[&edge.to];
            assert!(p_from < p_to,
                "edge {edge:?} violates topological order: from at {p_from}, to at {p_to}");
        }
    }

    #[test]
    fn builder_handles_qwen35_ssm_variants() {
        // SG-1.4 — SsmConv1d (and the other 13 SSM/Q-Gate variants)
        // now build instead of panicking. The previous test
        // (`builder_panics_on_unimplemented_ssm_variant`) was the
        // SG-1.3-era deferral assertion; SG-1.4 ships the helpers so
        // the same plan now produces a graph node with byte-range
        // metadata.
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::SsmConv1d { layer: 0 }]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 1);
        let writes = graph.nodes[0].writes();
        // SsmConv1d writes ssm_conv_input + ssm_conv_output + conv_state.
        assert!(writes.iter().any(|w| w.buffer == bufs.ssm_conv_output));
        assert!(writes.iter().any(|w| w.buffer == bufs.conv_state));
    }

    #[test]
    #[should_panic(expected = "SG-1.5")]
    fn builder_still_panics_on_gemma4_variant() {
        // SG-1.5 still pending — Gemma-4 variants should panic with
        // the appropriate deferral message.
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::VNorm]];
        let _ = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
    }
}
