// SPDX-License-Identifier: GPL-3.0-only
//! vf-clide — a lean CLI client for VulkanForge's OpenAI-compatible API.
//!
//! **HTTP-only.** This crate has no dependency on any VulkanForge engine
//! crate; it talks to a running `vulkanforge serve` instance purely over
//! the OpenAI `/v1/chat/completions` HTTP endpoint. That keeps the seam
//! clean for a future `git subtree split` into its own repository.
//!
//! Phase 1 shipped chat (REPL + headless one-shot, streaming +
//! non-streaming), in-session history, and a no-op memory seam. Phase 2
//! Slice 1 adds an **opt-in agent loop** (`--agent`): one `read_file`
//! tool, a permission gate, and a full tool-call roundtrip. The plain
//! chat path is unchanged.

pub mod agent;
pub mod client;
pub mod memory;
pub mod repl;
pub mod status;
pub mod tools;
pub mod types;
