//! VulkanGraph node types (Sprint SG-1.1 — data structures only).
//!
//! Two concrete dispatch-target kinds:
//!   * [`DispatchNode`] — a `vkCmdDispatch` of a pre-built compute pipeline.
//!   * [`TransferNode`] — a `vkCmdCopyBuffer` between two device buffers.
//!
//! Both carry **byte-range reads/writes** so the graph can derive
//! dependencies via memory-overlap analysis (the lever from
//! `results/vulkan_graph_analysis*.md` Teil 1 §2.3 and Teil 4 §2).
//! VF's existing imperative path uses whole-buffer-dirty tracking
//! (`pending_writes: HashSet<u64>`) and emits 1,678 barriers/token on
//! Qwen3.6 — byte-range tracking is the path to the ~50–70/token llama.cpp
//! emits.
//!
//! No record/optimize logic in this module; see `mod.rs` for the
//! `VulkanGraph` container and Sprint SG-1.2 (Builder) / SG-1.3
//! (Executor) for the active code paths.

use ash::vk;

/// Index into [`super::VulkanGraph::nodes`].
pub type NodeId = u32;

/// Raw Vulkan buffer handle (matches what the existing barrier-elision
/// tracker keys on, `forward::mod::pending_writes: HashSet<u64>`).
pub type BufferHandle = vk::Buffer;

/// Half-open byte range `[offset, offset + size)` inside a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub size: u64,
}

impl ByteRange {
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Whole-buffer marker (matches the imperative path's coarse
    /// granularity). Builder code may use this for buffers that VF
    /// touches end-to-end (e.g. scratch buffers reset per layer).
    pub fn whole() -> Self {
        Self { offset: 0, size: u64::MAX }
    }

    /// True iff the two ranges share at least one byte. Used by
    /// `VulkanGraph::resolve_dependencies` to insert edges when a
    /// later node reads memory a prior node wrote.
    pub fn overlaps(self, other: ByteRange) -> bool {
        if self.size == u64::MAX || other.size == u64::MAX {
            return true; // whole-buffer poisons everything
        }
        let a_end = self.offset.saturating_add(self.size);
        let b_end = other.offset.saturating_add(other.size);
        self.offset < b_end && other.offset < a_end
    }
}

/// One `(binding_index, buffer, offset, range)` slot for a descriptor
/// set update. Matches the tuple shape passed to
/// `forward::mod::alloc_or_get_set`.
#[derive(Debug, Clone, Copy)]
pub struct Binding {
    pub binding: u32,
    pub buffer: BufferHandle,
    pub offset: u64,
    /// `0` is interpreted as `VK_WHOLE_SIZE` at descriptor-write time
    /// (mirrors the existing `alloc_or_get_set` convention).
    pub range: u64,
}

impl Binding {
    pub fn new(binding: u32, buffer: BufferHandle, offset: u64, range: u64) -> Self {
        Self { binding, buffer, offset, range }
    }
}

/// One memory access by a [`DispatchNode`] or [`TransferNode`]. Used by
/// the dependency resolver to figure out which nodes are RAW/WAW/WAR
/// dependent.
#[derive(Debug, Clone, Copy)]
pub struct MemAccess {
    pub buffer: BufferHandle,
    pub range: ByteRange,
}

impl MemAccess {
    pub fn new(buffer: BufferHandle, offset: u64, size: u64) -> Self {
        Self { buffer, range: ByteRange::new(offset, size) }
    }
    pub fn whole(buffer: BufferHandle) -> Self {
        Self { buffer, range: ByteRange::whole() }
    }
}

/// A compute dispatch: bind pipeline → bind descriptor set →
/// push-constants → `vkCmdDispatch`. The Recorder turns this back into
/// the same vkCmd* sequence the imperative executor uses today.
pub struct DispatchNode {
    pub id: NodeId,
    /// Vulkan compute pipeline (from `PipelineRegistry::get(...)`).
    pub pipeline: vk::Pipeline,
    /// Matching pipeline layout. Stored explicitly so the Recorder
    /// doesn't need to look it up at record time.
    pub pipeline_layout: vk::PipelineLayout,
    /// Descriptor-set layout (for `alloc_or_get_set` cache key).
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// Per-slot binding info, in binding-index order.
    pub bindings: Vec<Binding>,
    /// Push-constant payload as raw bytes (Builder serializes the
    /// per-shader push struct; Recorder forwards the bytes verbatim).
    pub push_constants: Vec<u8>,
    /// `vkCmdDispatch(x, y, z)` group counts.
    pub dispatch: (u32, u32, u32),
    /// Memory regions this node reads (weights + input buffers).
    pub reads: Vec<MemAccess>,
    /// Memory regions this node writes.
    pub writes: Vec<MemAccess>,
    /// Source layer index (for per-layer profile aggregation + debug).
    pub layer: u32,
    /// Profile label that will be passed to `ShaderProfiler::begin`.
    /// `None` to skip profiling for this dispatch.
    pub label: Option<String>,
}

/// A `vkCmdCopyBuffer` of `size` bytes between two buffers. Sprint
/// G-2i found that the GDN state copy-back needs an explicit
/// transfer barrier; modeling transfer as its own graph node lets the
/// Recorder emit the right `cmd_pipeline_barrier` for compute→transfer
/// and transfer→compute transitions.
pub struct TransferNode {
    pub id: NodeId,
    pub src_buffer: BufferHandle,
    pub src_offset: u64,
    pub dst_buffer: BufferHandle,
    pub dst_offset: u64,
    pub size: u64,
    /// Always `[(src_buffer, src_offset..src_offset+size)]`.
    pub reads: Vec<MemAccess>,
    /// Always `[(dst_buffer, dst_offset..dst_offset+size)]`.
    pub writes: Vec<MemAccess>,
    pub layer: u32,
    pub label: Option<String>,
}

/// Sum type over every graph-node variant. Sprint SG-1.2 will add
/// `FusedDispatch` (e.g. MULTI_ADD-16); Sprint SG-1.3 may add
/// `Event` (VkEvent2-based fine-grained sync, see analysis Teil 2 §3.3 idea 5).
pub enum GraphNode {
    Dispatch(DispatchNode),
    Transfer(TransferNode),
}

impl GraphNode {
    pub fn id(&self) -> NodeId {
        match self {
            GraphNode::Dispatch(d) => d.id,
            GraphNode::Transfer(t) => t.id,
        }
    }

    pub fn reads(&self) -> &[MemAccess] {
        match self {
            GraphNode::Dispatch(d) => &d.reads,
            GraphNode::Transfer(t) => &t.reads,
        }
    }

    pub fn writes(&self) -> &[MemAccess] {
        match self {
            GraphNode::Dispatch(d) => &d.writes,
            GraphNode::Transfer(t) => &t.writes,
        }
    }

    pub fn layer(&self) -> u32 {
        match self {
            GraphNode::Dispatch(d) => d.layer,
            GraphNode::Transfer(t) => t.layer,
        }
    }

    pub fn label(&self) -> Option<&str> {
        match self {
            GraphNode::Dispatch(d) => d.label.as_deref(),
            GraphNode::Transfer(t) => t.label.as_deref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_range_overlap_basic() {
        let a = ByteRange::new(0, 100);
        let b = ByteRange::new(50, 50);
        let c = ByteRange::new(100, 50);
        assert!(a.overlaps(b));
        assert!(b.overlaps(a));
        assert!(!a.overlaps(c)); // [0,100) vs [100,150) — touching but not overlapping
        assert!(!c.overlaps(a));
    }

    #[test]
    fn byte_range_whole_buffer_poisons() {
        let whole = ByteRange::whole();
        let small = ByteRange::new(1_000_000, 1);
        assert!(whole.overlaps(small));
        assert!(small.overlaps(whole));
    }

    #[test]
    fn byte_range_disjoint() {
        let a = ByteRange::new(0, 10);
        let b = ByteRange::new(20, 10);
        assert!(!a.overlaps(b));
    }

    #[test]
    fn byte_range_self_overlap() {
        let a = ByteRange::new(100, 50);
        assert!(a.overlaps(a));
    }
}
