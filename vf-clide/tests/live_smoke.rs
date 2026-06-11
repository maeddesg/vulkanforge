// SPDX-License-Identifier: GPL-3.0-only
//! Live smoke test — requires a running `vulkanforge serve` instance.
//!
//! Skipped by default (`#[ignore]`). Run explicitly:
//!   cargo test --test live_smoke -- --ignored
//! Override target via env: VF_CLIDE_URL, VF_CLIDE_MODEL.

use vf_clide::client::Client;
use vf_clide::types::ChatMessage;

#[tokio::test]
#[ignore = "requires a running VulkanForge server"]
async fn live_stream_roundtrip() {
    let url = std::env::var("VF_CLIDE_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let model = std::env::var("VF_CLIDE_MODEL").unwrap_or_else(|_| "x".into());
    let client = Client::new(url, model);

    let mut streamed = String::new();
    let out = client
        .chat_stream(vec![ChatMessage::user("Reply with exactly one word: PONG")], |t| {
            streamed.push_str(t)
        })
        .await
        .expect("stream request should succeed against a live server");

    assert!(!out.text.is_empty(), "streamed answer must be non-empty");
    assert_eq!(out.text, streamed, "per-token callback must reconstruct the full answer");
}

/// Demonstrates the REPL's core mechanism — in-memory multi-turn history
/// — end-to-end (the REPL loop itself needs a TTY, so it can't be piped).
/// Turn 1 states a fact; turn 2 can only answer if the accumulated
/// history carried it over.
#[tokio::test]
#[ignore = "requires a running VulkanForge server"]
async fn live_multiturn_history() {
    let url = std::env::var("VF_CLIDE_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let model = std::env::var("VF_CLIDE_MODEL").unwrap_or_else(|_| "x".into());
    let client = Client::new(url, model);

    let mut history: Vec<ChatMessage> = Vec::new();
    history.push(ChatMessage::user("Remember this number: 42. Just acknowledge briefly."));
    let a1 = client.chat_once(history.clone()).await.expect("turn 1");
    // Push the assistant reply verbatim (faithful client-level multi-turn).
    // NOTE: the REPL stores `strip_think(reply)`; for thinking models the
    // exact history bytes are a greedy knife-edge (trimming whitespace can
    // flip turn 2 into a think-only, empty-visible answer). See the report.
    history.push(ChatMessage::assistant(a1.text.clone()));

    history.push(ChatMessage::user(
        "What number did I ask you to remember? Reply with just the number.",
    ));
    let a2 = client.chat_once(history.clone()).await.expect("turn 2");
    assert!(a2.text.contains("42"), "multi-turn history must carry the fact; got: {:?}", a2.text);
}

#[tokio::test]
#[ignore = "requires a running VulkanForge server"]
async fn live_non_streaming_roundtrip() {
    let url = std::env::var("VF_CLIDE_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let model = std::env::var("VF_CLIDE_MODEL").unwrap_or_else(|_| "x".into());
    let client = Client::new(url, model);

    let out = client
        .chat_once(vec![ChatMessage::user("Reply with exactly one word: PONG")])
        .await
        .expect("non-streaming request should succeed against a live server");
    assert!(!out.text.is_empty(), "answer must be non-empty");
}
