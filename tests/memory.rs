//! Stufe-A memory subsystem integration tests (real embeddings + SQLiteGraph).
//!
//! The whole subsystem is behind the `memory` Cargo feature (default OFF), so
//! this test crate compiles to nothing unless built with `--features memory`.
//! Run: `cargo test --release --features memory --test memory -- --test-threads=1`.
//!
//! These exercise the `remember → recall` loop, project isolation, the global
//! default scope, persistence across a store restart, get-or-create
//! idempotency, and the node-id ↔ recall coupling. They need the Nomic ONNX
//! model; if it can't be loaded (offline first run), each test **self-skips**
//! with a printed note rather than failing — so the lib/CI gate stays clean.
//! Run single-threaded (each test owns a SQLite file): `--test-threads=1`.
#![cfg(feature = "memory")]

use vulkanforge::server::memory::MemoryStore;

/// All tests share one parent dir so the embedder model cache (`embed-cache/`,
/// a sibling of the db file) is downloaded once, not per test.
fn db_for(name: &str) -> std::path::PathBuf {
    let base = std::env::temp_dir().join("vf_mem_tests");
    let _ = std::fs::create_dir_all(&base);
    let p = base.join(format!("{name}.db"));
    for suf in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{}", p.display(), suf));
    }
    p
}

/// Build a store, or skip the test (returns None) if the embedder is absent.
macro_rules! store_or_skip {
    ($name:expr) => {
        match MemoryStore::new(db_for($name)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP {}: memory store unavailable ({e})", $name);
                return;
            }
        }
    };
}

#[test]
fn remember_then_recall_returns_the_nugget() {
    let store = store_or_skip!("loop");
    let id = store
        .remember(Some("vf"), "Learning", "Dispatch-Reduktion hilft nicht auf gfx1201", None, None)
        .expect("remember");
    let hits = store
        .recall(Some("vf"), "bringt weniger barriers performance?", 3)
        .expect("recall");
    assert!(!hits.is_empty(), "recall returned nothing");
    assert_eq!(hits[0].id, id, "top hit must be the remembered nugget");
    assert!(hits[0].text.contains("Dispatch-Reduktion"), "text round-tripped");
    assert!(hits[0].score > 0.0 && hits[0].score <= 1.0001, "score in [0,1]: {}", hits[0].score);
}

#[test]
fn recall_is_project_isolated() {
    let store = store_or_skip!("isolation");
    let a = store
        .remember(Some("proj_a"), "Concept", "Vulkan compute shaders on AMD RDNA4 gfx1201", None, None)
        .expect("remember A");
    let b = store
        .remember(Some("proj_b"), "Concept", "A sourdough bread recipe with flour and water", None, None)
        .expect("remember B");
    // Query semantically near A; must never surface B's id.
    let hits = store.recall(Some("proj_a"), "GPU kernel dispatch", 5).expect("recall A");
    assert!(!hits.is_empty(), "expected A hit");
    assert!(hits.iter().all(|h| h.id != b), "project A recall leaked a B nugget: {hits:?}");
    assert!(hits.iter().any(|h| h.id == a), "A nugget should be recalled");
    // And B's project never returns A.
    let hb = store.recall(Some("proj_b"), "GPU kernel dispatch", 5).expect("recall B");
    assert!(hb.iter().all(|h| h.id != a), "project B recall leaked an A nugget: {hb:?}");
}

#[test]
fn global_default_scope_is_isolated_from_named() {
    let store = store_or_skip!("global");
    let g = store.remember(None, "Learning", "Global default scope nugget about tensors", None, None).expect("remember global");
    let n = store.remember(Some("named"), "Learning", "Named project nugget about tensors", None, None).expect("remember named");
    let hits = store.recall(None, "tensors", 5).expect("recall global");
    assert!(hits.iter().any(|h| h.id == g), "global recall should find the global nugget");
    assert!(hits.iter().all(|h| h.id != n), "global recall leaked a named-project nugget");
}

#[test]
fn persists_across_store_restart_without_re_embed() {
    let path = db_for("persist");
    {
        let store = match MemoryStore::new(path.clone()) {
            Ok(s) => s,
            Err(e) => { eprintln!("SKIP persist: {e}"); return; }
        };
        store.remember(Some("vf"), "Decision", "MTP geparkt weil MoE net-negativ", None, None).expect("remember");
        store.shutdown();
        // store dropped here → simulate server restart
    }
    let reopened = MemoryStore::new(path).expect("reopen same db");
    let hits = reopened.recall(Some("vf"), "why was multi-token prediction parked?", 3).expect("recall after restart");
    assert!(!hits.is_empty(), "nugget did not survive restart");
    assert!(hits[0].text.contains("MTP geparkt"), "restored nugget text: {:?}", hits[0]);
}

#[test]
fn get_or_create_is_idempotent() {
    let store = store_or_skip!("idem");
    let id1 = store.create_project("dup", None).expect("create 1");
    let id2 = store.create_project("dup", None).expect("create 2 (must not error)");
    assert_eq!(id1, id2, "re-create must return the same project node, no duplicate");
    // remember into a not-yet-created project must not error (auto-creates).
    store.remember(Some("fresh"), "Bug", "auto-created project on first remember", None, None).expect("remember into fresh project");
}

#[test]
fn node_id_couples_remember_and_recall() {
    let store = store_or_skip!("coupling");
    let id = store.remember(Some("vf"), "Benchmark", "Q8 vs FP32 decode is bit-identical", None, None).expect("remember");
    let hits = store.recall(Some("vf"), "are Q8 and FP32 decode the same?", 1).expect("recall");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id, "recall hit id must equal the remember()-returned node id (linchpin)");
}
