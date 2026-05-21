//! VulkanGraph — software compute-graph layer for VF's decode pipeline.
//!
//! Sprint SG-1 ("Säule Graph") deliverable. This commit (SG-1.1) ships
//! the **data structures only**: nodes, dependencies, container API.
//! Two follow-up sprints complete the system:
//!
//!  * **SG-1.2 GraphBuilder** — convert the existing `LayerStep` IR
//!    (26+ variants, exhaustive match per coding-standards §4.4) into
//!    a [`VulkanGraph`].
//!  * **SG-1.3 GraphRecorder + integration** — record a graph into a
//!    Vulkan command buffer behind a `VF_USE_GRAPH=1` opt-in env-gate;
//!    the imperative `DecodeExec`/`BatchExec` path stays as the
//!    default + fallback.
//!
//! ## Why a graph at all
//!
//! The cross-team analysis at `results/vulkan_graph_analysis_part4.md`
//! attributed VF's 2.04× gap to llama.cpp on Qwen3.6 to two
//! orchestration issues:
//!
//!   1. **Whole-buffer-dirty barrier tracking** emits ~1,678 barriers
//!      per decode token; byte-range memory-overlap tracking (the
//!      `Dependency` model in [`edge`]) cuts that to ~50–70/tok
//!      (llama.cpp baseline).
//!   2. **Missing op fusion** — VF emits ~1,449 dispatches/tok against
//!      llama.cpp's ~440. MULTI_ADD-16, RMS_NORM_MUL, TOPK_MOE etc.
//!      compress dispatch count by 3×.
//!
//! Both fixes need a place to *plan ahead* across a layer's dispatches
//! (or further). The imperative `executor::execute_step` only sees one
//! step at a time. The graph is that planner.
//!
//! ## Scope of SG-1.1 (this commit)
//!
//! - [`node::DispatchNode`], [`node::TransferNode`], [`node::GraphNode`]
//! - [`node::Binding`], [`node::MemAccess`], [`node::ByteRange`]
//! - [`edge::Dependency`] + [`edge::DependencyKind`]
//! - [`VulkanGraph`] with `add_dispatch` / `add_transfer` /
//!   `resolve_dependencies` / `sort` / `optimize` / `record` API stubs
//! - Unit tests for the byte-range overlap helper (the hot path of
//!   future dependency resolution).
//!
//! No callers integrated. No production-path change. The default
//! decode path in `forward/decode.rs` is untouched; `VF_USE_GRAPH=1`
//! will be wired in SG-1.3.

pub mod builder;
pub mod edge;
pub mod node;

use ash::vk;

pub use edge::{Dependency, DependencyKind};
pub use node::{
    Binding, BufferHandle, ByteRange, DispatchNode, GraphNode, MemAccess, NodeId, TransferNode,
};

/// A planned, dependency-aware sequence of Vulkan compute dispatches +
/// buffer transfers. Built once per decode token (or per prefill chunk
/// in SG-2+), optimized in-place, then recorded into a primary command
/// buffer.
pub struct VulkanGraph {
    /// Owns every node. `NodeId` indexes this vec.
    pub nodes: Vec<GraphNode>,
    /// All inter-node dependencies (RAW / WAW / WAR). Populated by
    /// [`VulkanGraph::resolve_dependencies`].
    pub edges: Vec<Dependency>,
    /// Topological order. Populated by [`VulkanGraph::sort`]. Empty
    /// until `sort` runs.
    pub execution_order: Vec<NodeId>,
}

impl VulkanGraph {
    /// An empty graph. Build it up with `add_dispatch` / `add_transfer`,
    /// then call `resolve_dependencies` + `sort` + (optionally)
    /// `optimize` before `record`.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            execution_order: Vec::new(),
        }
    }

    /// Pre-allocate space for `capacity` nodes. Qwen3.6 emits ~1,449
    /// dispatches per decode token (analysis Teil 4 §4.1); preferring
    /// one allocation over many small grows is the cheap default.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            edges: Vec::with_capacity(capacity * 2),
            execution_order: Vec::with_capacity(capacity),
        }
    }

    /// Append a dispatch node. Caller fills in everything but `id`;
    /// the graph assigns the id and returns it.
    pub fn add_dispatch(&mut self, mut node: DispatchNode) -> NodeId {
        let id = self.nodes.len() as NodeId;
        node.id = id;
        self.nodes.push(GraphNode::Dispatch(node));
        id
    }

    /// Append a transfer node (cmd_copy_buffer). Same id-assignment
    /// convention as [`add_dispatch`].
    pub fn add_transfer(&mut self, mut node: TransferNode) -> NodeId {
        let id = self.nodes.len() as NodeId;
        node.id = id;
        self.nodes.push(GraphNode::Transfer(node));
        id
    }

    /// Number of nodes currently in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// O(N²) byte-range overlap sweep. For every pair `(i, j)` with
    /// `i < j`, emits a [`Dependency`] whenever any of
    ///   * RAW: `i.writes` overlaps `j.reads` on the same buffer
    ///   * WAW: `i.writes` overlaps `j.writes`
    ///   * WAR: `i.reads`  overlaps `j.writes`
    ///
    /// Implements the analysis-Teil-4-§2 lever: byte-range tracking
    /// replaces VF's whole-buffer-dirty `pending_writes: HashSet<u64>`
    /// and lets the future SG-1.3 Recorder emit only the barriers
    /// implied by actual memory dependencies. Insertion order is the
    /// build order — `from < to` always holds, so a topological order
    /// exists for the resulting DAG.
    pub fn resolve_dependencies(&mut self) {
        self.edges.clear();
        let n = self.nodes.len();
        for i in 0..n {
            // Snapshot i's accesses to avoid the borrow conflict with
            // `&mut self.edges.push` below.
            let writes_i: Vec<_> = self.nodes[i].writes().to_vec();
            let reads_i: Vec<_>  = self.nodes[i].reads().to_vec();
            for j in (i + 1)..n {
                let writes_j = self.nodes[j].writes();
                let reads_j  = self.nodes[j].reads();

                // RAW: i writes, j reads
                for w in &writes_i {
                    for r in reads_j {
                        if w.buffer == r.buffer && w.range.overlaps(r.range) {
                            self.edges.push(Dependency::raw(
                                i as NodeId, j as NodeId,
                                w.buffer, w.range, r.range,
                            ));
                            break; // one edge per (i, j, kind) is enough
                        }
                    }
                }
                // WAW: i writes, j writes
                'waw: for wi in &writes_i {
                    for wj in writes_j {
                        if wi.buffer == wj.buffer && wi.range.overlaps(wj.range) {
                            self.edges.push(Dependency::waw(
                                i as NodeId, j as NodeId,
                                wi.buffer, wi.range, wj.range,
                            ));
                            break 'waw;
                        }
                    }
                }
                // WAR: i reads, j writes (mostly matters for
                // compute→transfer ordering; cheap to compute either way)
                'war: for ri in &reads_i {
                    for wj in writes_j {
                        if ri.buffer == wj.buffer && ri.range.overlaps(wj.range) {
                            self.edges.push(Dependency::war(
                                i as NodeId, j as NodeId,
                                ri.buffer, ri.range, wj.range,
                            ));
                            break 'war;
                        }
                    }
                }
            }
        }
    }

    /// Kahn's topological sort respecting [`edges`]. Output goes into
    /// [`execution_order`]. Panics on cycle — cycles in a compute graph
    /// would indicate a Builder bug (forward-only edges by construction).
    pub fn sort(&mut self) {
        use std::collections::VecDeque;
        let n = self.nodes.len();
        let mut in_degree: Vec<u32> = vec![0; n];
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for e in &self.edges {
            adj[e.from as usize].push(e.to);
            in_degree[e.to as usize] += 1;
        }
        let mut queue: VecDeque<NodeId> = (0..n as NodeId)
            .filter(|&i| in_degree[i as usize] == 0)
            .collect();
        self.execution_order.clear();
        self.execution_order.reserve(n);
        while let Some(v) = queue.pop_front() {
            self.execution_order.push(v);
            for &w in &adj[v as usize] {
                in_degree[w as usize] -= 1;
                if in_degree[w as usize] == 0 {
                    queue.push_back(w);
                }
            }
        }
        assert_eq!(
            self.execution_order.len(), n,
            "cycle detected in graph: only {} of {} nodes sorted (Builder bug)",
            self.execution_order.len(), n
        );
    }

    /// **STUB (SG-1.2+).** Run optimization passes (fusion, dead-node
    /// elimination, barrier coalescing). Each `Pass` consumes and
    /// returns a graph; concrete passes (MULTI_ADD fusion, etc.) are
    /// added incrementally so they can ship one by one with a clean
    /// Coherence-5/5 gate per pass.
    pub fn optimize(&mut self, _passes: &[&dyn Pass]) {
        // Empty placeholder. First concrete pass lands in SG-2.
    }

    /// **STUB (SG-1.3).** Record the graph into a primary Vulkan
    /// command buffer. Iterates [`execution_order`] and dispatches /
    /// transfers each node, emitting only the barriers required by
    /// the dependency graph.
    pub fn record(
        &self,
        _dev: &ash::Device,
        _cmd: vk::CommandBuffer,
    ) {
        // Empty placeholder. Real implementation lands in SG-1.3.
    }
}

impl Default for VulkanGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for optional graph optimization passes. Sprint SG-2 ships the
/// first concrete pass (MULTI_ADD-16 fusion). Each pass is
/// independently switchable so the analysis side-by-side bench can
/// quantify per-pass contribution.
pub trait Pass {
    /// Short human label for telemetry (`VF_GPU_TIMER` integration
    /// later prints which passes ran).
    fn name(&self) -> &'static str;

    /// Mutates the graph in place. Must preserve dependency
    /// correctness; passes that change writes/reads must also update
    /// `MemAccess` lists.
    fn apply(&self, graph: &mut VulkanGraph);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    fn dummy_dispatch_node() -> DispatchNode {
        DispatchNode {
            id: 0, // overwritten by add_dispatch
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            bindings: Vec::new(),
            push_constants: Vec::new(),
            dispatch: (1, 1, 1),
            reads: Vec::new(),
            writes: Vec::new(),
            layer: 0,
            label: None,
        }
    }

    fn dummy_transfer_node() -> TransferNode {
        TransferNode {
            id: 0,
            src_buffer: vk::Buffer::null(),
            src_offset: 0,
            dst_buffer: vk::Buffer::null(),
            dst_offset: 0,
            size: 0,
            reads: Vec::new(),
            writes: Vec::new(),
            layer: 0,
            label: None,
        }
    }

    #[test]
    fn new_graph_is_empty() {
        let g = VulkanGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.execution_order.is_empty());
    }

    #[test]
    fn add_dispatch_assigns_increasing_ids() {
        let mut g = VulkanGraph::new();
        let id0 = g.add_dispatch(dummy_dispatch_node());
        let id1 = g.add_dispatch(dummy_dispatch_node());
        let id2 = g.add_transfer(dummy_transfer_node());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(g.len(), 3);
        assert_eq!(g.nodes[0].id(), 0);
        assert_eq!(g.nodes[1].id(), 1);
        assert_eq!(g.nodes[2].id(), 2);
    }

    #[test]
    fn sort_stub_returns_insertion_order() {
        let mut g = VulkanGraph::new();
        g.add_dispatch(dummy_dispatch_node());
        g.add_dispatch(dummy_dispatch_node());
        g.add_dispatch(dummy_dispatch_node());
        g.sort();
        assert_eq!(g.execution_order, vec![0u32, 1, 2]);
    }

    #[test]
    fn resolve_dependencies_stub_does_not_panic() {
        let mut g = VulkanGraph::new();
        g.add_dispatch(dummy_dispatch_node());
        g.resolve_dependencies();
        assert!(g.edges.is_empty(), "SG-1.1 stub leaves edges empty");
    }

    #[test]
    fn with_capacity_preallocates() {
        let g = VulkanGraph::with_capacity(1500);
        assert!(g.nodes.capacity() >= 1500);
    }
}
