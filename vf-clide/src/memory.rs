// SPDX-License-Identifier: GPL-3.0-only
//! Memory seam (Phase 1: no-op).
//!
//! This is the single integration point where project memory will later
//! hook into the request-build path. Phase 1 ships only the seam + a
//! no-op implementation so that Phase 2 can drop in a real memory backend
//! without changing any call site.
//!
//! Deliberately minimal — the memory architecture (server-side in VF's
//! API vs client-side, scoping) is an open decision; this is just the
//! hook so nothing has to be restructured later.

use crate::types::ChatMessage;

/// Hook invoked while building a chat request. An implementation may
/// return `Some(system_context)` to be prepended as a `system` message
/// (e.g. recalled project memory). The Phase-1 no-op returns `None`.
pub trait Memory {
    fn context_for(&self, _project: Option<&str>, _history: &[ChatMessage]) -> Option<String> {
        None
    }
}

/// Phase-1 default: injects nothing. Swapped for a real backend in Phase 2.
pub struct NoopMemory;
impl Memory for NoopMemory {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_memory_injects_nothing() {
        let m = NoopMemory;
        assert!(m.context_for(Some("myproject"), &[]).is_none());
        assert!(m.context_for(None, &[ChatMessage::user("hi")]).is_none());
    }
}
