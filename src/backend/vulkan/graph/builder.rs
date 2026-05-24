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
    Binding, BufferHandle, ByteRange, DispatchNode, MemAccess, NodeId, SubDispatch,
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
    /// Sprint SG-1.6 — Gemma-4-26B MoE block scratch (PreMoeNorm output,
    /// MoE input). Present on all model states (`Forward::scratch_b`),
    /// only consumed by the Gemma-4 MoE Builder helpers.
    pub scratch_b: BufferHandle,
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

    // ── Sprint SG-1.7 — Gemma-4-26B MoE router scratch buffers ─────
    // Live on `Forward::moe_router_gpu` (Option, present only on
    // MoE-active Gemma-4 builds). Threaded through BufferMap so the
    // dep-pass sees the GateUp → GLU → Down dispatch chain.
    pub moe_gate_up_out: BufferHandle,
    pub moe_glu_out: BufferHandle,
    pub moe_down_out: BufferHandle,
    pub moe_router_logits: BufferHandle,
    pub moe_router_indices: BufferHandle,
    pub moe_router_weights: BufferHandle,
}

impl BufferMap {
    /// All-null placeholder. Used in unit tests + the Phase-1 stub
    /// path where we only care about graph structure / dependencies.
    pub fn null() -> Self {
        Self {
            scratch_a: vk::Buffer::null(),
            scratch_b: vk::Buffer::null(),
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
            moe_gate_up_out: vk::Buffer::null(),
            moe_glu_out: vk::Buffer::null(),
            moe_down_out: vk::Buffer::null(),
            moe_router_logits: vk::Buffer::null(),
            moe_router_indices: vk::Buffer::null(),
            moe_router_weights: vk::Buffer::null(),
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
                // SsmOutProj) instead; Gemma-4 paths use gate_buf
                // (written by PostAttnNorm). Read ALL THREE so the
                // byte-range graph picks up the right RAW edge
                // regardless of architecture — `execute_step`
                // dispatches the correct buffer either way.
                // Sprint SG-1.7-bisect: gate_buf was missing, which
                // left the PostAttnNorm → AttnResidualAdd edge
                // unmodelled on Gemma-4 → race under GraphDriven.
                let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
                self.graph.add_dispatch(DispatchNode {
                    id: 0,
                    sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
                        MemAccess::new(self.bufs.gate_buf, 0, hidden_bytes),
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
            LayerStep::FfnResidualAdd => {
                // Llama/Qwen3 paths read res1 + ffn_out; Gemma-4 paths
                // read res1 + gate_buf (the PostFfnNorm-scratch). Read
                // ALL THREE so the byte-range graph picks up the right
                // RAW edge regardless of architecture — `execute_step`
                // dispatches the correct buffer either way.
                // Sprint SG-1.7-bisect: gate_buf was missing, which
                // left the PostFfnNorm → FfnResidualAdd edge
                // unmodelled on Gemma-4 → race under GraphDriven.
                let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
                self.graph.add_dispatch(DispatchNode {
                    id: 0,
                    sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
                    pipeline: vk::Pipeline::null(),
                    pipeline_layout: vk::PipelineLayout::null(),
                    descriptor_set_layout: vk::DescriptorSetLayout::null(),
                    bindings: Vec::new(),
                    push_constants: Vec::new(),
                    dispatch: (self.cfg.hidden_dim.div_ceil(64), 1, 1),
                    reads: vec![
                        MemAccess::new(self.bufs.res1, 0, hidden_bytes),
                        MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes),
                        MemAccess::new(self.bufs.gate_buf, 0, hidden_bytes),
                    ],
                    writes: vec![MemAccess::new(self.bufs.layer_output, 0, hidden_bytes)],
                    layer,
                    label: Some(format!("L{layer}_ffn_residual_add")),
                });
            }

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

            // ===== Gemma-4 KV-share + 4-norm + PLE (SG-1.5) =====
            // Each helper emits a single FullStep DispatchNode whose
            // reads/writes describe the byte-range flow of the
            // corresponding `executor::*` step body. Under
            // `BarrierMode::Imperative` (the Gemma-4 default in
            // `execute_layer_via_graph`'s gate), the step body owns
            // synchronization via its existing `maybe_compute_barrier`
            // / `mark_written` / explicit `cmd_pipeline_barrier` calls.
            // The graph nodes here drive the topo-order walk and the
            // future-SG-2 byte-range optimization passes.
            //
            // PleBlock is internally 5 sub-dispatches. SG-3-style
            // decomposition would replace `force_internal_barrier`-
            // class workarounds — but Gemma-4 doesn't use that pattern
            // (PleBlock's internal `maybe_compute_barrier`s fire under
            // Imperative mode), so we keep it as FullStep until the
            // GraphDriven gate also opens for Gemma-4 (SG-1.7+).
            LayerStep::VFromKRaw => self.add_v_from_k_raw(layer),
            LayerStep::VNorm => self.add_v_norm(layer),
            LayerStep::PostAttnNorm => self.add_post_attn_norm(layer),
            LayerStep::PostFfnNorm => self.add_post_ffn_norm(layer),
            LayerStep::PleBlock => self.add_ple_block(layer),
            LayerStep::LayerScalarMul => self.add_layer_scalar_mul(layer),

            // ===== Gemma-4-26B MoE (SG-1.6) =====
            // All emit FullStep nodes — `BarrierMode::Imperative`
            // (`cfg.gemma4.is_none()` gate) keeps the step bodies'
            // internal `maybe_compute_barrier` + `mark_written` calls
            // live. MoeExpertFfn is internally 18-32 sub-dispatches
            // (P0-1 batched / per-slot); SG-3-style decomposition is
            // SG-1.7+ follow-up before the GraphDriven gate can open.
            LayerStep::PostDenseMlpNorm => self.add_post_dense_mlp_norm(layer),
            LayerStep::PreMoeNorm => self.add_pre_moe_norm(layer),
            LayerStep::MoeRoute { .. } => self.add_moe_route(layer),
            LayerStep::MoeExpertFfn { .. } => self.add_moe_expert_ffn(layer),
            LayerStep::PostMoeNorm => self.add_post_moe_norm(layer),
            LayerStep::MoeBranchAdd => self.add_moe_branch_add(layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
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

    /// Sprint SG-3 — decomposed-step node emitter. Same as
    /// [`Self::ssm_simple_node`] but the caller specifies the
    /// `SubDispatch` discriminator, so the Recorder routes via its
    /// per-variant `sub_*` helper instead of `execute_step`. The
    /// decomposed multi-dispatch steps (NormGated, AlphaGate,
    /// Conv1d, BetaProj, GatedDeltaNet) emit one of these per
    /// internal sub-dispatch, with precise byte-range reads/writes so
    /// the graph's dep-pass discovers the intra-step RAW/WAW edges.
    fn decomposed_node(
        &mut self,
        layer: u32,
        sub: SubDispatch,
        reads: Vec<MemAccess>,
        writes: Vec<MemAccess>,
        label: &str,
    ) {
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: sub,
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

    /// Sprint SG-3 — decomposed: GEMV(hidden → beta) + Sigmoid(beta).
    /// 2 nodes. Old single-node version used `force_internal_barrier`
    /// inside the step body; decomposition lets the graph's barrier
    /// pass own the intra-step beta_buf RAW/WAW.
    fn add_ssm_beta_proj(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.decomposed_node(layer, SubDispatch::BetaGemv,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES)],
            "ssm_beta_gemv",
        );
        self.decomposed_node(layer, SubDispatch::BetaSigmoid,
            vec![MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES)],
            "ssm_beta_sigmoid",
        );
    }

    /// Sprint SG-3 — decomposed: 4 nodes covering the 4 sub-dispatches
    /// inside `step_ssm_alpha_gate`. Previous SG-1.4-b version
    /// inserted 3 `force_internal_barrier` calls between them; the
    /// graph dep-pass now sees explicit RAW/WAW edges (alpha → alpha
    /// → alpha → gate) and emits real cmd_pipeline_barriers between
    /// adjacent nodes.
    fn add_ssm_alpha_gate(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        // 1. alpha = ssm_alpha.weight @ hidden_norm
        self.decomposed_node(layer, SubDispatch::AlphaGateGemv,
            vec![MemAccess::new(self.bufs.hidden_norm, 0, hidden_bytes)],
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            "ssm_alpha_gemv",
        );
        // 2. alpha += ssm_dt.bias
        self.decomposed_node(layer, SubDispatch::AlphaGateAdd,
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            "ssm_alpha_add_dt",
        );
        // 3. alpha = softplus(alpha)
        self.decomposed_node(layer, SubDispatch::AlphaGateSoftplus,
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            "ssm_alpha_softplus",
        );
        // 4. gate = alpha * ssm_a
        self.decomposed_node(layer, SubDispatch::AlphaGateMulA,
            vec![MemAccess::new(self.bufs.ssm_alpha, 0, Self::N_V_HEADS_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_gate, 0, Self::N_V_HEADS_BYTES)],
            "ssm_alpha_mul_a",
        );
    }

    /// Sprint SG-3 — decomposed: ConvSetup (builds conv_input from
    /// qkv + conv_state) + ConvDispatch (runs the conv shader). 2
    /// nodes. The graph dep-pass discovers conv_input RAW between
    /// them, replacing the SG-1.4-b `force_internal_barrier`.
    fn add_ssm_conv1d(&mut self, layer: u32) {
        // 1. Setup: builds conv_input + advances conv_state window.
        self.decomposed_node(layer, SubDispatch::ConvSetup,
            vec![
                MemAccess::new(self.bufs.ssm_qkv, 0, Self::CONV_CHANNELS_BYTES),
                MemAccess::whole(self.bufs.conv_state),
            ],
            vec![
                MemAccess::new(self.bufs.ssm_conv_input, 0,
                    Self::CONV_CHANNELS_BYTES * Self::D_CONV),
                MemAccess::whole(self.bufs.conv_state),
            ],
            "ssm_conv_setup",
        );
        // 2. Conv: reads conv_input, writes conv_output.
        self.decomposed_node(layer, SubDispatch::ConvDispatch,
            vec![MemAccess::new(self.bufs.ssm_conv_input, 0,
                Self::CONV_CHANNELS_BYTES * Self::D_CONV)],
            vec![MemAccess::new(self.bufs.ssm_conv_output, 0, Self::CONV_CHANNELS_BYTES)],
            "ssm_conv",
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

    /// Sprint SG-3 — decomposed: 1 Dispatch (the gated_delta_net
    /// shader) + 1 Transfer (cmd_copy_buffer of new-state slice from
    /// gdn_out back into ssm_state). The graph dep-pass discovers
    /// the gdn_out RAW between them; the Recorder's per-Transfer
    /// barrier emit handles the COMPUTE→TRANSFER stage transition
    /// the same way `step_gated_delta_net`'s `pre_copy` MemoryBarrier
    /// did before.
    fn add_gated_delta_net(&mut self, layer: u32) {
        // 1. GDN compute dispatch.
        self.decomposed_node(layer, SubDispatch::GdnCompute,
            vec![
                MemAccess::new(self.bufs.ssm_qrep, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_krep, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_conv_output,
                    2 * Self::QK_SLICE_BYTES, Self::V_SLICE_BYTES),  // V slice
                MemAccess::new(self.bufs.ssm_gate, 0, Self::N_V_HEADS_BYTES),
                MemAccess::new(self.bufs.ssm_beta, 0, Self::N_V_HEADS_BYTES),
                MemAccess::whole(self.bufs.ssm_state),
            ],
            vec![MemAccess::whole(self.bufs.ssm_gdn_out)],
            "gdn",
        );
        // 2. State copy-back as a Transfer node (Recorder issues
        // cmd_copy_buffer + the COMPUTE→TRANSFER pre_copy barrier +
        // a closing transfer_to_compute_barrier in `sub_gdn_state_copy`).
        let _copy_id = self.graph.add_transfer(super::node::TransferNode {
            id: 0,
            sub_dispatch: SubDispatch::GdnStateCopy,
            src_buffer: self.bufs.ssm_gdn_out,
            src_offset: 0,
            dst_buffer: self.bufs.ssm_state,
            dst_offset: 0,
            size: 0,
            reads:  vec![MemAccess::whole(self.bufs.ssm_gdn_out)],
            writes: vec![MemAccess::whole(self.bufs.ssm_state)],
            layer,
            label: Some(format!("L{layer}_gdn_state_copy")),
        });
    }

    /// Sprint SG-3 — decomposed: RMSNorm + SiLU + Mul, 3 nodes. The
    /// previous SG-1.4-e single-node modelled SiLU's in-place z write
    /// via a whole-step `writes: [ssm_z]` so the WAW propagated; with
    /// decomposition the SiLU node carries its own writes(z) and the
    /// Mul node's reads(z) drive an explicit RAW.
    fn add_norm_gated(&mut self, layer: u32) {
        // 1. RMSNorm: reads gdn_out (output portion) + weight, writes norm_out.
        self.decomposed_node(layer, SubDispatch::NormGatedRms,
            vec![MemAccess::new(self.bufs.ssm_gdn_out, 0, Self::D_INNER_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_norm_out, 0, Self::D_INNER_BYTES)],
            "ssm_norm_gated_rms",
        );
        // 2. SiLU: reads z, writes z (in-place).
        self.decomposed_node(layer, SubDispatch::NormGatedSilu,
            vec![MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES)],
            vec![MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES)],
            "ssm_z_silu",
        );
        // 3. Mul: reads norm_out + z, writes norm_out.
        self.decomposed_node(layer, SubDispatch::NormGatedMul,
            vec![
                MemAccess::new(self.bufs.ssm_norm_out, 0, Self::D_INNER_BYTES),
                MemAccess::new(self.bufs.ssm_z, 0, Self::D_INNER_BYTES),
            ],
            vec![MemAccess::new(self.bufs.ssm_norm_out, 0, Self::D_INNER_BYTES)],
            "ssm_norm_gated_mul",
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

    // ─────────────────────────────────────────────────────────────────
    // Sprint SG-1.5 — Gemma-4 dense (KV-share + 4-norm + PLE) helpers.
    //
    // All FullStep nodes — Gemma-4 stays under `BarrierMode::Imperative`
    // in `execute_layer_via_graph` (gated by `cfg.gemma4.is_none()`),
    // so the corresponding `step_*` bodies emit their own internal
    // barriers. The graph carries dep metadata only for topo-order
    // and future SG-2 byte-range optimization passes.
    //
    // Byte-range modelling stays coarse-but-correct: each helper lists
    // the principal buffer reads/writes that matter for layer-to-layer
    // ordering. Weights + constant scratch buffers (`vnorm_ones`,
    // `layer_scalar`, `per_layer_inputs`) are intentionally skipped —
    // they're constant within a forward and don't drive sync edges.
    // ─────────────────────────────────────────────────────────────────

    /// `step_v_from_k_raw` — Gemma-4 KV-share. Issues
    /// `cmd_copy_buffer(k_buf → v_buf)` wrapped in COMPUTE→TRANSFER
    /// and TRANSFER→COMPUTE barriers (see `executor/attention.rs:250`).
    /// Modeled as a FullStep DispatchNode (not a TransferNode) so the
    /// Recorder routes via `execute_step` which handles the barriers
    /// inline — same as the imperative path.
    fn add_v_from_k_raw(&mut self, layer: u32) {
        // n_kv_heads × head_dim × 4 bytes. Use whole(k_buf) for read
        // since the helper trusts the runtime kv-head count.
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.k_buf, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.v_buf, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_v_from_k_raw")),
        });
    }

    /// `step_v_norm` — Gemma-4 RMSNorm on v_buf with per-arch
    /// `vnorm_ones` constant. In-place: reads + writes v_buf.
    fn add_v_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.v_buf, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.v_buf, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_v_norm")),
        });
    }

    /// `step_post_attn_norm` — Gemma-4 extra RMSNorm after attention.
    /// Reads o_buf + `ffn_norm.weight`, writes `gate_buf` (reused as
    /// scratch since gate_buf isn't live until FFN gate-proj).
    fn add_post_attn_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.o_buf, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.gate_buf, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_post_attn_norm")),
        });
    }

    /// `step_post_ffn_norm` — Gemma-4 extra RMSNorm after FFN down-proj.
    /// Reads ffn_out + `ffn_post_norm.weight`, writes `gate_buf`
    /// (SwiGLU/GELU is done by this point so gate_buf is reusable).
    fn add_post_ffn_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.gate_buf, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_post_ffn_norm")),
        });
    }

    /// Sprint SG-1.5 (kept) / SG-1.7 (honest-partial) — PleBlock stays
    /// emitted as a single FullStep node. The decomposed SubDispatch
    /// variants (PleGemvGate / PleGeluGlu / PleGemvProj / PleRmsNorm /
    /// PleAddOutput) and the matching sub_* recorder helpers are
    /// shipped for future revisits, but the Builder doesn't emit them
    /// because Gemma-4 stays under `BarrierMode::Imperative` (gate
    /// `cfg.gemma4.is_none()` = false). Decomposed sub_* helpers don't
    /// emit barriers — they'd race under imperative mode.
    fn add_ple_block(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.layer_output, 0, hidden_bytes),
            ],
            writes: vec![
                MemAccess::new(self.bufs.gate_buf, 0, hidden_bytes),
                MemAccess::new(self.bufs.o_buf, 0, hidden_bytes),
                MemAccess::new(self.bufs.attn_out, 0, hidden_bytes),
                MemAccess::new(self.bufs.layer_output, 0, hidden_bytes),
            ],
            layer,
            label: Some(format!("L{layer}_ple")),
        });
    }

    /// `step_layer_scalar_mul` — Gemma-4 per-layer scalar multiply.
    /// In-place: reads + writes layer_output.
    fn add_layer_scalar_mul(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.layer_output, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.layer_output, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_scalar_mul")),
        });
    }

    // ─────────────────────────────────────────────────────────────────
    // Sprint SG-1.6 — Gemma-4-26B-A4B MoE block helpers.
    //
    // Type-B layers (19 of 30 on the 26B Q3_K_M) run a parallel
    // Dense-MLP + MoE branch then merge. Step ordering (from
    // `layer_plan::build_gemma4_layer`):
    //
    //   PostAttnNorm → AttnResidualAdd (writes res1)
    //   ── Dense branch ──
    //     PreFfnNorm → Gate/Up → SwiGLU → Down (writes ffn_out)
    //     PostDenseMlpNorm (ffn_out → scratch_a = h1)
    //   ── MoE branch ──
    //     PreMoeNorm (res1 → scratch_b = MoE input)
    //     MoeRoute (res1 → router GPU/CPU state)
    //     MoeExpertFfn (scratch_b + router → ffn_hidden)
    //     PostMoeNorm (ffn_hidden → ffn_out = h2)
    //   ── Merge ──
    //     MoeBranchAdd (ffn_out += scratch_a)
    //     FfnResidualAdd / PleBlock / LayerScalarMul tail
    //
    // All FullStep nodes — Gemma-4 stays under `BarrierMode::Imperative`
    // until SG-1.7 decomposes MoeExpertFfn (18-32 sub-dispatches with
    // internal `maybe_compute_barrier` calls). Byte-range modeling
    // is coarse-but-correct; weights + router-internal buffers are
    // intentionally omitted (constant within a forward + not in
    // BufferMap; topo-order falls out of insertion order regardless
    // since the layer plan is forward-only).
    // ─────────────────────────────────────────────────────────────────

    /// `step_post_dense_mlp_norm` — RMSNorm of the Dense-MLP output
    /// into the h1 scratch. Reads ffn_out (DownProj output), writes
    /// scratch_a (= h1, lives until MoeBranchAdd).
    fn add_post_dense_mlp_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.scratch_a, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_post_dense_mlp_norm")),
        });
    }

    /// `step_pre_moe_norm` — RMSNorm of the raw post-attention
    /// residual into the MoE-input scratch. Reads res1, writes
    /// scratch_b. Both branches read the SAME res1 (not the
    /// Dense-MLP-normed value) per HF's Gemma-4 reference.
    fn add_pre_moe_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.res1, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.scratch_b, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_pre_moe_norm")),
        });
    }

    /// Sprint SG-1.7 — MoeRoute decomposed into 2 nodes mirroring
    /// `run_moe_router_gpu` (`runs.rs:2714-2787`):
    ///   (1) MoeRouterNormGemv: input (res1) → logits_scratch
    ///   (2) MoeRouterSoftmaxTopk: logits + pes → indices + weights
    /// Router-internal scratch buffers (logits/indices/weights) are not
    /// in BufferMap — the dep-pass needs only the input edge (res1)
    /// and the consume-side edges from MoeExpertFfn. We model an
    /// invisible-dep between the two nodes via insertion order
    /// (Builder's topological default).
    /// Sprint SG-1.6 (kept) / SG-1.7 (honest-partial) — MoeRoute stays
    /// emitted as a single FullStep node. Decomposition deferred per
    /// the same gate-flip blocker as PleBlock.
    fn add_moe_route(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.res1, 0, hidden_bytes)],
            writes: Vec::new(),
            layer,
            label: Some(format!("L{layer}_moe_route")),
        });
    }

    /// Sprint SG-1.7 — MoeExpertFfn decomposed mirroring
    /// `step_moe_expert_ffn_gpu_direct_batched` (`moe.rs:659`):
    ///   (1) Clear ffn_hidden          — TransferNode (cmd_fill_buffer)
    ///   (2) Gate+Up Y-batched GEMV    — reads scratch_b, writes gate_up_out
    ///   (3..3+top_k) Per-slot GLU     — DISJOINT slice writes on glu_out
    ///   (last-1) Down Y-batched GEMV  — reads glu_out, writes down_out
    ///   (last-top_k..last) Per-slot FMA — SEQUENTIAL on ffn_hidden (WAW chain)
    ///
    /// GLU slots are modeled with **disjoint byte-range writes** so the
    /// SG-2 dep-pass can later allow parallel scheduling. FMA slots
    /// write the whole `ffn_hidden` range (in-place accumulator) so
    /// they form a sequential RAW + WAW chain — graph emits a barrier
    /// between every FMA slot (Sprint 61G multi-FMA race precedent).
    ///
    /// Router-scratch buffers (gate_up_out / glu_out / down_out /
    /// indices / weights) live on `Forward::moe_router_gpu` and aren't
    /// in BufferMap — `top_k` adjacent GLU/FMA nodes are emitted in
    /// insertion order so the topo-sort respects build-order.
    fn add_moe_expert_ffn(&mut self, layer: u32) {
        // Sprint SG-1.6 / SG-1.7 honest-partial — single FullStep
        // node (SG-1.6 behaviour). The decomposed Builder path lives
        // in `add_moe_expert_ffn_decomposed` (currently `#[allow(dead_code)]`)
        // and ships as future SG-1.7-bisect material.
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.scratch_b, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.ffn_hidden, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_moe_expert_ffn")),
        });
    }

    /// Sprint SG-1.7 — decomposed MoeExpertFfn (currently unused, kept
    /// for future bisect). 1 clear + 1 gate_up + top_k GLU + 1 down
    /// + top_k FMA = 18 nodes (for top_k=8). See `add_moe_expert_ffn`
    /// for the active FullStep path.
    #[allow(dead_code)]
    fn add_moe_expert_ffn_decomposed(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        let top_k = self.cfg.gemma4.as_ref()
            .map(|g| g.top_k_experts)
            .unwrap_or(8) as u8;

        // (1) Clear ffn_hidden — TransferNode.
        self.graph.add_transfer(super::node::TransferNode {
            id: 0,
            sub_dispatch: SubDispatch::MoeFfnClear,
            src_buffer: vk::Buffer::null(),
            src_offset: 0,
            dst_buffer: self.bufs.ffn_hidden,
            dst_offset: 0,
            size: hidden_bytes,
            reads: Vec::new(),
            writes: vec![MemAccess::new(self.bufs.ffn_hidden, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_moe_ffn_clear")),
        });
        let mi = self.cfg.gemma4.as_ref()
            .map(|g| g.moe_intermediate_size)
            .unwrap_or(2048);
        let mi_bytes = (mi as u64) * 4;
        let two_mi_bytes = 2 * mi_bytes;

        // (2) Gate+Up Y-batched GEMV. Reads scratch_b + indices,
        // writes the full gate_up_out slab (top_k × 2*mi).
        self.decomposed_node(layer, SubDispatch::MoeFfnGateUp,
            vec![
                MemAccess::new(self.bufs.scratch_b, 0, hidden_bytes),
                MemAccess::whole(self.bufs.moe_router_indices),
            ],
            vec![MemAccess::whole(self.bufs.moe_gate_up_out)],
            "moe_gate_up_batched",
        );
        // (3..3+top_k) Per-slot GLU — reads disjoint slot slice of
        // gate_up_out, writes disjoint slot slice of glu_out. Disjoint
        // ranges → dep-pass sees no WAW chain between adjacent slots
        // (in principle parallel-schedulable; SG-2 future optimization).
        for slot in 0..top_k {
            let k = slot as u64;
            let gate_off = k * two_mi_bytes;
            self.decomposed_node(layer, SubDispatch::MoeFfnGluSlot(slot),
                vec![MemAccess::new(self.bufs.moe_gate_up_out, gate_off, two_mi_bytes)],
                vec![MemAccess::new(self.bufs.moe_glu_out, k * mi_bytes, mi_bytes)],
                &format!("moe_glu_slot{slot}"),
            );
        }
        // (4) Down Y-batched GEMV. Reads full glu_out slab + indices,
        // writes the full down_out slab (top_k × hidden).
        self.decomposed_node(layer, SubDispatch::MoeFfnDown,
            vec![
                MemAccess::whole(self.bufs.moe_glu_out),
                MemAccess::whole(self.bufs.moe_router_indices),
            ],
            vec![MemAccess::whole(self.bufs.moe_down_out)],
            "moe_down_batched",
        );
        // (5..5+top_k) Per-slot FMA — sequential WAW chain on ffn_hidden.
        // Each FMA reads its own slot of down_out + the weights buffer.
        for slot in 0..top_k {
            let in_off = (slot as u64) * hidden_bytes;
            self.decomposed_node(layer, SubDispatch::MoeFfnFmaSlot(slot),
                vec![
                    MemAccess::new(self.bufs.moe_down_out, in_off, hidden_bytes),
                    MemAccess::whole(self.bufs.moe_router_weights),
                    MemAccess::new(self.bufs.ffn_hidden, 0, hidden_bytes),
                ],
                vec![MemAccess::new(self.bufs.ffn_hidden, 0, hidden_bytes)],
                &format!("moe_fma_slot{slot}"),
            );
        }
    }

    /// `step_post_moe_norm` — RMSNorm of the MoE-branch sum. Reads
    /// ffn_hidden (MoE expert weighted sum), writes ffn_out (= h2,
    /// aliasing the now-freed Dense-MLP-output slot).
    fn add_post_moe_norm(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads:  vec![MemAccess::new(self.bufs.ffn_hidden, 0, hidden_bytes)],
            writes: vec![MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_post_moe_norm")),
        });
    }

    /// `step_moe_branch_add` — `ffn_out (h2) += scratch_a (h1)`. The
    /// merge of Dense-MLP branch and MoE branch into ffn_out. In-place
    /// on ffn_out.
    fn add_moe_branch_add(&mut self, layer: u32) {
        let hidden_bytes = self.cfg.hidden_dim as u64 * 4;
        self.graph.add_dispatch(DispatchNode {
            id: 0,
            sub_dispatch: SubDispatch::FullStep(self.current_step_in_layer),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads: vec![
                MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes),
                MemAccess::new(self.bufs.scratch_a, 0, hidden_bytes),
            ],
            writes: vec![MemAccess::new(self.bufs.ffn_out, 0, hidden_bytes)],
            layer,
            label: Some(format!("L{layer}_moe_branch_add")),
        });
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
            scratch_b: fake_buffer(28),
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
            moe_gate_up_out: fake_buffer(29),
            moe_glu_out: fake_buffer(30),
            moe_down_out: fake_buffer(31),
            moe_router_logits: fake_buffer(34),
            moe_router_indices: fake_buffer(32),
            moe_router_weights: fake_buffer(33),
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
        // SG-3 — SsmConv1d now decomposes to 2 nodes (ConvSetup +
        // ConvDispatch). The previous SG-1.4-era assertion of 1
        // node-per-step is no longer correct.
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::SsmConv1d { layer: 0 }]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 2, "SsmConv1d decomposes to 2 nodes");
        // ConvSetup writes ssm_conv_input + conv_state.
        let setup_writes = graph.nodes[0].writes();
        assert!(setup_writes.iter().any(|w| w.buffer == bufs.ssm_conv_input));
        assert!(setup_writes.iter().any(|w| w.buffer == bufs.conv_state));
        // ConvDispatch writes ssm_conv_output.
        let conv_writes = graph.nodes[1].writes();
        assert!(conv_writes.iter().any(|w| w.buffer == bufs.ssm_conv_output));
        // RAW edge from ConvSetup to ConvDispatch on ssm_conv_input.
        assert!(
            graph.edges.iter().any(|e| e.from == 0 && e.to == 1
                && e.buffer == bufs.ssm_conv_input),
            "expected ConvSetup→ConvDispatch RAW on ssm_conv_input, got {:?}",
            graph.edges
        );
    }

    #[test]
    fn builder_decomposes_norm_gated_into_three_nodes() {
        // SG-3 — NormGated emits RMSNorm + SiLU + Mul (3 nodes) with
        // the intra-step RAW edges (rms→mul on norm_out, silu→mul on z).
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::NormGated { layer: 0 }]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 3, "NormGated decomposes to 3 nodes");
        // RAW edge from RMSNorm (node 0) → Mul (node 2) on norm_out.
        assert!(graph.edges.iter().any(|e| e.from == 0 && e.to == 2
                && e.buffer == bufs.ssm_norm_out));
        // RAW edge from SiLU (node 1) → Mul (node 2) on z.
        assert!(graph.edges.iter().any(|e| e.from == 1 && e.to == 2
                && e.buffer == bufs.ssm_z));
    }

    #[test]
    fn builder_decomposes_alpha_gate_into_four_nodes() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::SsmAlphaGate { layer: 0 }]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 4, "SsmAlphaGate decomposes to 4 nodes");
        // Chain: GEMV → Add → Softplus → MulA (each RAW on alpha_buf).
        for i in 0..3 {
            assert!(
                graph.edges.iter().any(|e| e.from == i && e.to == i + 1
                    && e.buffer == bufs.ssm_alpha),
                "expected node {i}→{} RAW on alpha", i + 1,
            );
        }
    }

    #[test]
    fn builder_decomposes_gdn_into_dispatch_plus_transfer() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::GatedDeltaNet { layer: 0 }]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 2, "GDN emits 1 Dispatch + 1 Transfer");
        // node 0 = Dispatch, node 1 = Transfer.
        // RAW from compute → state copy on ssm_gdn_out.
        assert!(graph.edges.iter().any(|e| e.from == 0 && e.to == 1
                && e.buffer == bufs.ssm_gdn_out));
    }

    #[test]
    fn builder_handles_gemma4_dense_variants() {
        // SG-1.5 — Gemma-4 KV-share + 4-norm + PLE Builder coverage.
        // SG-1.7 ships SubDispatch variants + sub_* recorder helpers
        // for PleBlock decomposition but the active Builder path stays
        // FullStep (Gemma-4 under BarrierMode::Imperative).
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![
            LayerStep::VFromKRaw,
            LayerStep::VNorm,
            LayerStep::PostAttnNorm,
            LayerStep::PostFfnNorm,
            LayerStep::PleBlock,
            LayerStep::LayerScalarMul,
        ]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 6, "6 Gemma-4 dense variants emit 6 FullStep nodes");
        // VFromKRaw → VNorm RAW edge on v_buf.
        assert!(graph.edges.iter().any(|e| e.from == 0 && e.to == 1
                && e.buffer == bufs.v_buf),
            "expected VFromKRaw→VNorm RAW on v_buf");
        // PostAttnNorm writes gate_buf; PostFfnNorm also writes gate_buf →
        // WAW edge between them.
        assert!(graph.edges.iter().any(|e| e.from == 2 && e.to == 3
                && e.buffer == bufs.gate_buf),
            "expected PostAttnNorm→PostFfnNorm WAW on gate_buf scratch");
    }

    #[test]
    fn builder_handles_gemma4_moe_variants() {
        // SG-1.6 — Gemma-4 MoE block Builder coverage. SG-1.7 ships
        // decomposition infrastructure (SubDispatch variants + sub_*
        // helpers) but the active Builder path emits FullStep nodes.
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![
            LayerStep::PostDenseMlpNorm,
            LayerStep::PreMoeNorm,
            LayerStep::MoeRoute { n_experts: 128, top_k: 8 },
            LayerStep::MoeExpertFfn { n_experts: 128, top_k: 8, moe_intermediate: 2048 },
            LayerStep::PostMoeNorm,
            LayerStep::MoeBranchAdd,
        ]];
        let graph = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
        assert_eq!(graph.len(), 6, "6 Gemma-4 MoE variants emit 6 FullStep nodes");
        // PreMoeNorm (1) writes scratch_b; MoeExpertFfn (3) reads it.
        assert!(graph.edges.iter().any(|e| e.from == 1 && e.to == 3
                && e.buffer == bufs.scratch_b),
            "expected PreMoeNorm→MoeExpertFfn RAW on scratch_b");
        // PostMoeNorm (4) writes ffn_out; MoeBranchAdd (5) reads ffn_out.
        assert!(graph.edges.iter().any(|e| e.from == 4 && e.to == 5
                && e.buffer == bufs.ffn_out),
            "expected PostMoeNorm→MoeBranchAdd RAW on ffn_out");
    }
}
