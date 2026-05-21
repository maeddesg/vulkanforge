//! Graph dependency edges (Sprint SG-1.1).
//!
//! Each [`Dependency`] records one RAW (read-after-write) or WAW
//! (write-after-write) constraint between two nodes:
//! `from` writes a byte range that `to` either reads or also writes.
//! The Recorder emits a `vkCmdPipelineBarrier` (or `vkCmdSetEvent2` in
//! Sprint G-13/SG-1+) before `to` only when at least one incoming
//! Dependency is unsatisfied — the precise byte-range form unblocks
//! the ~30× barrier reduction described in
//! `results/vulkan_graph_analysis_part4.md` §2.

use super::node::{BufferHandle, ByteRange, NodeId};

/// Why one node must wait for another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyKind {
    /// `from` wrote bytes that `to` reads (most common).
    ReadAfterWrite,
    /// `from` wrote bytes that `to` also writes (must serialize).
    WriteAfterWrite,
    /// `from` reads bytes that `to` writes (only matters for
    /// compute→transfer ordering — most compute-only graphs ignore
    /// this).
    WriteAfterRead,
}

/// One memory dependency edge. The conservative version of a graph
/// barrier — Sprint SG-1.3 will collapse multiple edges incoming on
/// the same node into a single `vkCmdPipelineBarrier`.
#[derive(Debug, Clone, Copy)]
pub struct Dependency {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: DependencyKind,
    pub buffer: BufferHandle,
    /// The byte-range `from` wrote (for RAW + WAW) or read (for WAR).
    pub from_range: ByteRange,
    /// The byte-range `to` accesses on this dependency.
    pub to_range: ByteRange,
}

impl Dependency {
    pub fn raw(
        from: NodeId, to: NodeId,
        buffer: BufferHandle, from_range: ByteRange, to_range: ByteRange,
    ) -> Self {
        Self { from, to, kind: DependencyKind::ReadAfterWrite, buffer, from_range, to_range }
    }

    pub fn waw(
        from: NodeId, to: NodeId,
        buffer: BufferHandle, from_range: ByteRange, to_range: ByteRange,
    ) -> Self {
        Self { from, to, kind: DependencyKind::WriteAfterWrite, buffer, from_range, to_range }
    }

    pub fn war(
        from: NodeId, to: NodeId,
        buffer: BufferHandle, from_range: ByteRange, to_range: ByteRange,
    ) -> Self {
        Self { from, to, kind: DependencyKind::WriteAfterRead, buffer, from_range, to_range }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    #[test]
    fn raw_dependency_carries_kind() {
        let buf = vk::Buffer::null();
        let dep = Dependency::raw(0, 1, buf, ByteRange::new(0, 100), ByteRange::new(0, 100));
        assert_eq!(dep.kind, DependencyKind::ReadAfterWrite);
        assert_eq!(dep.from, 0);
        assert_eq!(dep.to, 1);
    }

    #[test]
    fn waw_war_constructors() {
        let buf = vk::Buffer::null();
        let waw = Dependency::waw(0, 1, buf, ByteRange::new(0, 50), ByteRange::new(0, 50));
        let war = Dependency::war(2, 3, buf, ByteRange::new(0, 50), ByteRange::new(0, 50));
        assert_eq!(waw.kind, DependencyKind::WriteAfterWrite);
        assert_eq!(war.kind, DependencyKind::WriteAfterRead);
    }
}
