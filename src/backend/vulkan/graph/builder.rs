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
        };
        for (layer, plan) in per_layer_plans.iter().enumerate() {
            for step in plan {
                builder.add_step(step, layer as u32);
            }
        }
        // lm_head is an architecture-independent terminal node; left
        // to SG-1.3 to add since it depends on `output.weight` lookup.

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
            LayerStep::AttnResidualAdd => self.add_binary(
                layer, self.bufs.scratch_a, self.bufs.o_buf, self.bufs.res1,
                self.cfg.hidden_dim, "attn_residual_add",
            ),
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

            // ===== Qwen3.6 Linear-Attention (SG-1.4 — stubbed) =====
            LayerStep::AttnQkvProj { .. }
            | LayerStep::AttnGateZProj { .. }
            | LayerStep::SsmBetaProj { .. }
            | LayerStep::SsmAlphaGate { .. }
            | LayerStep::SsmConv1d { .. }
            | LayerStep::SsmSilu { .. }
            | LayerStep::SsmQkL2Norm { .. }
            | LayerStep::SsmRepeatQK { .. }
            | LayerStep::GatedDeltaNet { .. }
            | LayerStep::NormGated { .. }
            | LayerStep::SsmOutProj { .. }
            | LayerStep::AttnQGateProj { .. }
            | LayerStep::AttnGatedOutput { .. }
            | LayerStep::ResidualIdentitySeed => {
                unimplemented!(
                    "SG-1.4 — Qwen3.6 SSM / Q-Gate variant {step:?} not yet \
                     implemented in GraphBuilder. Use the imperative \
                     executor (VF_USE_GRAPH=0, default in v0.5.0-dev)."
                );
            }

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
    #[should_panic(expected = "SG-1.4 — Qwen3.6 SSM")]
    fn builder_panics_on_unimplemented_ssm_variant() {
        let cfg = dummy_qwen3_cfg();
        let bufs = distinct_buffer_map();
        let plans = vec![vec![LayerStep::SsmConv1d { layer: 0 }]];
        let _ = GraphBuilder::build_decode_token(&bufs, &cfg, 0, &plans);
    }
}
