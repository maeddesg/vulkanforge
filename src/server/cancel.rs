//! Cooperative cancellation primitive used by the streaming chat
//! handler.
//!
//! Architecture §4.4. The token is shared between three places:
//! 1. The SSE response future (axum drops it when the client TCP-
//!    disconnects).
//! 2. The mpsc-sender callback inside the GPU thread (sets the flag
//!    when `tx.blocking_send` fails — i.e. the receiver was dropped).
//! 3. The VF decode loop in `decode.rs` (checks the flag once per
//!    token via `GenerateConfig::cancel_token`). When the flag flips
//!    to `true`, the loop exits gracefully on the next iteration.
//!
//! The decode-loop check stops wasted GPU work within ~1 token of
//! the client disconnect.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Return the underlying `Arc<AtomicBool>` so it can be passed
    /// into [`crate::backend::vulkan::decode::GenerateConfig::cancel_token`].
    /// Cloning the `Arc` shares the flag — both ends observe the
    /// same store/load.
    pub fn as_arc(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_not_cancelled() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
    }

    #[test]
    fn cancel_flips_the_flag() {
        let t = CancelToken::new();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let t = CancelToken::new();
        t.cancel();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!a.is_cancelled());
        assert!(!b.is_cancelled());
        b.cancel();
        // Cancellation visible through the original handle.
        assert!(a.is_cancelled());
        assert!(b.is_cancelled());
    }

    #[test]
    fn as_arc_shares_the_inner_atomic_bool() {
        let t = CancelToken::new();
        let arc = t.as_arc();
        assert!(!arc.load(Ordering::Acquire));
        t.cancel();
        // GenerateConfig::cancel_token sees the same flip.
        assert!(arc.load(Ordering::Acquire));
    }

    #[test]
    fn default_is_uncancelled() {
        let t = CancelToken::default();
        assert!(!t.is_cancelled());
    }
}
