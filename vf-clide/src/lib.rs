// SPDX-License-Identifier: GPL-3.0-only
//! vf-clide — a lean CLI client for VulkanForge's OpenAI-compatible API.
//!
//! **HTTP-only.** This crate has no dependency on any VulkanForge engine
//! crate; it talks to a running `vulkanforge serve` instance purely over
//! the OpenAI `/v1/chat/completions` HTTP endpoint. That keeps the seam
//! clean for a future `git subtree split` into its own repository.
//!
//! Phase 1 scope: chat only (REPL + headless one-shot, streaming +
//! non-streaming), in-session history, and a no-op memory seam. No agent
//! loop, no tool-calling, no permission model — those are later phases.

pub mod client;
pub mod memory;
pub mod repl;
pub mod types;
