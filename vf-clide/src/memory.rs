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

use std::path::Path;

use crate::types::ChatMessage;

/// Hook invoked while building a chat request. An implementation may
/// return `Some(system_context)` to be prepended as a `system` message
/// (e.g. recalled project memory). The Phase-1 no-op returns `None`.
///
/// NOTE (Stufe B-1): auto-injection of recalled context is **B-2**. B-1
/// adds only the *manual* REPL commands (`/recall`/`/remember`/`/project`),
/// which call the server's `/memory/*` endpoints directly and **display**
/// results; they do not go through this seam. The seam stays a no-op until B-2.
pub trait Memory {
    fn context_for(&self, _project: Option<&str>, _history: &[ChatMessage]) -> Option<String> {
        None
    }
}

/// Phase-1 default: injects nothing. Swapped for a real backend in Phase 2.
pub struct NoopMemory;
impl Memory for NoopMemory {}

/// Derive a deterministic, collision-resistant `project_key` from the
/// (canonical) workspace path. **Deterministic**: the same workspace yields
/// the same key across sessions, so a project's memory persists. **Collision-
/// resistant**: a short hash of the *full absolute path* is appended, so two
/// different directories with the same basename (`/a/api` vs `/b/api`) get
/// different keys.
///
/// Form: `<basename-slug>-<8 hex of FNV-1a(abs path)>`, e.g.
/// `vulkanforge-1a2b3c4d`. Dep-free (a hand-rolled FNV-1a, no `hashbrown`/
/// `sha2`) so vf-clide stays a thin HTTP client.
pub fn derive_project_key(workspace: &Path) -> String {
    let abs = workspace.to_string_lossy();
    let base = workspace
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let slug = slugify(&base);
    let h = fnv1a64(abs.as_bytes());
    format!("{slug}-{:08x}", (h & 0xffff_ffff) as u32)
}

/// Lowercase ASCII-alphanumeric run; any other run collapses to a single
/// `-`. Empty result → `"workspace"` (e.g. a root path with no basename).
fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() { "workspace".to_string() } else { trimmed.to_string() }
}

/// FNV-1a 64-bit (offset basis + prime). Tiny, stable, no dependency.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_memory_injects_nothing() {
        let m = NoopMemory;
        assert!(m.context_for(Some("myproject"), &[]).is_none());
        assert!(m.context_for(None, &[ChatMessage::user("hi")]).is_none());
    }

    #[test]
    fn project_key_is_deterministic() {
        let p = Path::new("/home/dev/projects/vulkanforge");
        assert_eq!(derive_project_key(p), derive_project_key(p), "same path → same key");
    }

    #[test]
    fn project_key_distinguishes_same_basename_different_parents() {
        // Collision resistance: same basename, different absolute path.
        let a = derive_project_key(Path::new("/home/dev/a/api"));
        let b = derive_project_key(Path::new("/home/dev/b/api"));
        assert_ne!(a, b, "different abs paths must produce different keys");
        // ...but both keep the readable basename slug.
        assert!(a.starts_with("api-"), "got {a}");
        assert!(b.starts_with("api-"), "got {b}");
    }

    #[test]
    fn project_key_slugifies_basename() {
        let k = derive_project_key(Path::new("/tmp/My Project (v2)!"));
        // lowercased, non-alnum collapsed to single dashes, then `-<hash>`.
        assert!(k.starts_with("my-project-v2-"), "got {k}");
        // exactly one 8-hex suffix.
        let suffix = k.rsplit('-').next().unwrap();
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn project_key_handles_empty_basename() {
        // A root-ish path with no usable basename → "workspace-<hash>".
        let k = derive_project_key(Path::new("/"));
        assert!(k.starts_with("workspace-"), "got {k}");
    }
}
