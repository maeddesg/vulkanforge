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

use serde_json::json;
use vulkanforge::server::memory::{CurateError, CutReason, MemoryStore, NoteType};

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
        .expect("remember")
        .id;
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
        .expect("remember A")
        .id;
    let b = store
        .remember(Some("proj_b"), "Concept", "A sourdough bread recipe with flour and water", None, None)
        .expect("remember B")
        .id;
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
    let g = store.remember(None, "Learning", "Global default scope nugget about tensors", None, None).expect("remember global").id;
    let n = store.remember(Some("named"), "Learning", "Named project nugget about tensors", None, None).expect("remember named").id;
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
    let id = store.remember(Some("vf"), "Benchmark", "Q8 vs FP32 decode is bit-identical", None, None).expect("remember").id;
    let hits = store.recall(Some("vf"), "are Q8 and FP32 decode the same?", 1).expect("recall");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id, "recall hit id must equal the remember()-returned node id (linchpin)");
}

// ---- Stufe B-3: curation (dedup + archive + delete) ----

#[test]
fn dedup_skips_near_identical_but_keeps_distinct() {
    let store = store_or_skip!("dedup");
    let first = store
        .remember(Some("d"), "Note", "Q4_K matvec on gfx1201 is memory-bandwidth bound, not ALU bound", None, None)
        .expect("first");
    assert!(!first.deduped, "first store is not a dup");
    // Near-identical (only a hyphen removed) → deduped to the SAME node.
    let dup = store
        .remember(Some("d"), "Note", "Q4_K matvec on gfx1201 is memory bandwidth bound, not ALU bound", None, None)
        .expect("dup");
    assert!(dup.deduped, "near-identical note must be deduped");
    assert_eq!(dup.id, first.id, "dedup returns the existing node id, no new node");
    // A genuinely distinct note must NOT be swallowed.
    let distinct = store
        .remember(Some("d"), "Note", "Add a dark-mode toggle to the settings page", None, None)
        .expect("distinct");
    assert!(!distinct.deduped, "distinct note must not be deduped");
    assert_ne!(distinct.id, first.id);
    // Exactly two distinct vectors in the index (dup didn't add one).
    assert_eq!(store.raw_vector_count(Some("d")).expect("count"), 2, "dedup added no vector");
}

#[test]
fn archive_removes_from_recall_but_keeps_node() {
    let store = store_or_skip!("archive");
    let id = store
        .remember(Some("a"), "Decision", "Park MTP because MoE turned out net-negative", None, None)
        .expect("remember").id;
    assert!(store.recall(Some("a"), "what about multi-token prediction?", 5).expect("recall").iter().any(|h| h.id == id),
        "note should be recallable before archive");
    store.archive(Some("a"), id).expect("archive");
    // Out of recall...
    assert!(store.recall(Some("a"), "what about multi-token prediction?", 5).expect("recall").iter().all(|h| h.id != id),
        "archived note must not surface in recall");
    // ...but still a record in the graph, status=archived.
    let node = store.get(id).expect("archived node still in graph");
    assert_eq!(node.status, "archived");
    assert!(node.text.contains("Park MTP"), "record text retained");
    // And its vector is gone (raw count, not statistics()).
    assert_eq!(store.raw_vector_count(Some("a")).expect("count"), 0);
}

#[test]
fn unarchive_restores_to_recall_round_trip() {
    // The core round-trip: remember → recall finds · archive → recall misses ·
    // unarchive → recall finds AGAIN, with the right node, content and a sane
    // score. A wrong re-insert / lost node_id link fails the LAST recall.
    let store = store_or_skip!("unarchive_rt");
    let id = store
        .remember(Some("u"), "Decision", "Park MTP because MoE turned out net-negative", None, None)
        .expect("remember").id;
    let q = "what about multi-token prediction?";
    assert!(store.recall(Some("u"), q, 5).expect("recall").iter().any(|h| h.id == id),
        "recallable before archive");
    store.archive(Some("u"), id).expect("archive");
    assert!(store.recall(Some("u"), q, 5).expect("recall").iter().all(|h| h.id != id),
        "archived note out of recall");
    assert_eq!(store.raw_vector_count(Some("u")).expect("count"), 0, "vector gone after archive");

    store.unarchive(Some("u"), id).expect("unarchive");
    // Vector is back + node is active again.
    assert_eq!(store.raw_vector_count(Some("u")).expect("count"), 1, "vector restored after unarchive");
    assert_eq!(store.get(id).expect("node").status, "active", "status back to active");
    // recall finds it again — same id, right content, sensible score.
    let hits = store.recall(Some("u"), q, 5).expect("recall after unarchive");
    let hit = hits.iter().find(|h| h.id == id).expect("unarchived note recallable again");
    assert!(hit.text.contains("Park MTP"), "right content (node_id link intact): {hit:?}");
    assert!(hit.score > 0.0 && hit.score <= 1.0001, "sane score: {}", hit.score);
}

#[test]
fn unarchive_is_idempotent_and_errors_cleanly() {
    let store = store_or_skip!("unarchive_idem");
    let id = store
        .remember(Some("u2"), "Note", "a single distinct note to bounce around", None, None)
        .expect("remember").id;
    // Un-archiving an ACTIVE note is a no-op: no duplicate vector, no error.
    store.unarchive(Some("u2"), id).expect("unarchive of active note is a no-op");
    assert_eq!(store.raw_vector_count(Some("u2")).expect("count"), 1, "no duplicate vector for an active note");
    assert_eq!(store.recall(Some("u2"), "distinct note", 5).expect("recall").iter().filter(|h| h.id == id).count(), 1,
        "exactly one recall hit for the note (no dup)");
    // Archive then double-unarchive: second unarchive is a no-op (still 1).
    store.archive(Some("u2"), id).expect("archive");
    store.unarchive(Some("u2"), id).expect("unarchive 1");
    store.unarchive(Some("u2"), id).expect("unarchive 2 (no-op)");
    assert_eq!(store.raw_vector_count(Some("u2")).expect("count"), 1, "double unarchive adds no duplicate");
    // A non-existent id errors cleanly (no panic).
    assert!(store.unarchive(Some("u2"), 999_999).is_err(), "unarchive of a missing id must error");
}

#[test]
fn curation_missing_id_is_typed_not_found() {
    // Each id-targeting curation op must report a MISSING id as the typed
    // `CurateError::NotFound` (→ the handler maps it to 404), not a flat
    // internal error (→ 500). Real faults stay `Internal`.
    use vulkanforge::server::memory::CurateError;
    let store = store_or_skip!("curate_not_found");
    // A present note so the project/index exist — the error is purely "no id".
    let _ = store.remember(Some("nf"), "Note", "a present note about gfx1201", None, None).expect("remember");
    let missing = 999_999;
    for (op, res) in [
        ("archive", store.archive(Some("nf"), missing)),
        ("unarchive", store.unarchive(Some("nf"), missing)),
        ("delete", store.delete(Some("nf"), missing)),
    ] {
        match res {
            Err(CurateError::NotFound(id)) => assert_eq!(id, missing, "{op} reports the missing id"),
            other => panic!("{op} of a missing id must be CurateError::NotFound, got {other:?}"),
        }
    }
}

#[test]
fn unarchive_persists_across_restart() {
    let path = db_for("unarchive_persist");
    let id;
    {
        let store = match MemoryStore::new(path.clone()) {
            Ok(s) => s,
            Err(e) => { eprintln!("SKIP unarchive_persist: {e}"); return; }
        };
        id = store.remember(Some("p"), "Note", "delta note archived then restored", None, None).expect("a").id;
        store.archive(Some("p"), id).expect("archive");
        store.unarchive(Some("p"), id).expect("unarchive");
        store.shutdown();
    }
    let re = MemoryStore::new(path).expect("reopen");
    assert_eq!(re.raw_vector_count(Some("p")).expect("count"), 1, "restored vector survives restart");
    assert_eq!(re.get(id).expect("node").status, "active", "status active after restart");
    assert!(re.recall(Some("p"), "delta note", 5).expect("recall").iter().any(|h| h.id == id),
        "unarchived note recallable after restart");
}

#[test]
fn delete_is_hard_and_raw_count_drops() {
    let store = store_or_skip!("delete");
    let a = store.remember(Some("x"), "Note", "first distinct note about shaders", None, None).expect("a").id;
    let b = store.remember(Some("x"), "Note", "second distinct note about tokenizers", None, None).expect("b").id;
    assert_eq!(store.raw_vector_count(Some("x")).expect("count"), 2);
    store.delete(Some("x"), a).expect("delete");
    // Correctness hangs on raw COUNT + recall, NEVER statistics().vector_count
    // (stale-on-reopen, recall_scoping_smoke.md): the raw count drops to 1...
    assert_eq!(store.raw_vector_count(Some("x")).expect("count"), 1, "hard delete drops the raw count");
    // ...recall no longer returns the deleted note...
    assert!(store.recall(Some("x"), "shaders", 5).expect("recall").iter().all(|h| h.id != a),
        "deleted note gone from recall");
    // ...and the node is gone from the graph (unlike archive).
    assert!(store.get(a).is_none(), "deleted node removed from graph");
    assert!(store.get(b).is_some(), "the other note is untouched");
}

#[test]
fn archive_and_delete_persist_across_restart() {
    let path = db_for("curate_persist");
    let (arch_id, del_id, keep_id);
    {
        let store = match MemoryStore::new(path.clone()) {
            Ok(s) => s,
            Err(e) => { eprintln!("SKIP curate_persist: {e}"); return; }
        };
        arch_id = store.remember(Some("p"), "Note", "alpha note to be archived", None, None).expect("a").id;
        del_id = store.remember(Some("p"), "Note", "beta note to be deleted", None, None).expect("b").id;
        keep_id = store.remember(Some("p"), "Note", "gamma note to keep around", None, None).expect("c").id;
        store.archive(Some("p"), arch_id).expect("archive");
        store.delete(Some("p"), del_id).expect("delete");
        store.shutdown();
    }
    let re = MemoryStore::new(path).expect("reopen");
    // Only the kept note's vector survives (raw count = 1).
    assert_eq!(re.raw_vector_count(Some("p")).expect("count"), 1, "archived+deleted vectors stay gone after restart");
    // Archived note: still a record, still out of recall.
    assert_eq!(re.get(arch_id).expect("archived record survives").status, "archived");
    assert!(re.recall(Some("p"), "alpha", 5).expect("recall").iter().all(|h| h.id != arch_id), "archived stays out of recall");
    // Deleted note: gone for good.
    assert!(re.get(del_id).is_none(), "deleted node stays gone after restart");
    // Kept note: still recallable.
    assert!(re.recall(Some("p"), "gamma note", 5).expect("recall").iter().any(|h| h.id == keep_id), "kept note still recallable");
}

/// `recall --explain` surfaces the cutoff metadata + the near-miss candidates
/// that fell just outside the top-k, and its `returned` set matches a plain
/// recall of the same `k` (the instrument is additive — same ranking).
#[test]
fn recall_explain_surfaces_near_misses_and_cutoff() {
    let store = store_or_skip!("explain");
    // 3 on-topic notes + 2 deliberately off-topic. With k=3 the on-topic notes
    // are the returned top-3 and the off-topic ones rank below the cutoff →
    // they are the near-misses.
    for text in [
        "Vulkan compute shaders run on AMD RDNA4 gfx1201",
        "The GPU dispatches Q4_K matrix-vector multiply kernels",
        "Compute pipelines bind descriptor sets for storage buffers",
        "A sourdough bread recipe needs flour water salt and time",
        "The cat sat quietly on the warm sunny windowsill",
    ] {
        store.remember(Some("explain"), "fact", text, None, None).expect("remember");
    }

    let q = "GPU compute kernel dispatch on Vulkan";
    // margin=None → pure top-k (the diagnostic without the threshold).
    let ex = store.recall_explain(Some("explain"), q, 3, 5, None, None, false).expect("recall_explain");

    // Cutoff metadata is explicit and honest: pure top-k, no threshold, 768-dim.
    assert_eq!(ex.top_k, 3, "top_k echoes the requested cutoff");
    assert_eq!(ex.threshold, None, "margin=None → pure top-k — there is no score threshold");
    assert_eq!(ex.query_dim, 768, "Nomic-Embed-Text-v1.5 is 768-dim");

    // Returned = top-3; near-misses surfaced (5 notes, k=3 → ≥1 near-miss).
    assert_eq!(ex.returned.len(), 3, "returned the top-k");
    assert!(!ex.near_miss.is_empty(), "near-misses must be surfaced (5 notes, k=3)");
    // With no threshold, every near-miss was cut by the top-k cap.
    assert!(
        ex.near_miss.iter().all(|nm| nm.cut == CutReason::TopK),
        "margin=None → all near-misses are cut by top-k, not a threshold",
    );

    // Ranking holds and separation = last-hit − first-near-miss.
    let last_hit = ex.returned.last().unwrap().score;
    let first_miss = ex.near_miss.first().unwrap().hit.score;
    assert!(last_hit >= first_miss, "last hit {last_hit} must rank ≥ first near-miss {first_miss}");
    assert_eq!(ex.separation, Some(last_hit - first_miss), "separation = last hit − first near-miss");

    // The off-topic notes fell outside the top-3.
    assert!(
        ex.returned.iter().all(|h| !h.text.contains("sourdough") && !h.text.contains("cat")),
        "off-topic notes must not be in the returned top-3: {:?}",
        ex.returned.iter().map(|h| &h.text).collect::<Vec<_>>(),
    );

    // Additive: explain.returned is identical to a plain recall of the same k.
    let plain = store.recall(Some("explain"), q, 3).expect("recall");
    let plain_ids: Vec<i64> = plain.iter().map(|h| h.id).collect();
    let ret_ids: Vec<i64> = ex.returned.iter().map(|h| h.id).collect();
    assert_eq!(plain_ids, ret_ids, "explain.returned must match plain recall(k) — same ranking");
}

/// The Schicht-Enabler payoff: a **type filter** disambiguates where the
/// similarity threshold cannot — adjacent-domain notes share a score band but
/// differ in (explicit, non-embedding) type.
#[test]
fn note_type_disambiguates_where_similarity_cannot() {
    let store = store_or_skip!("typing");
    // 3 GPU "working" facts + 2 GPU "decision" notes — same domain/score band,
    // different type. Type set via metadata (what the handler folds into data).
    let working = [
        "The Q4_K mul_mat_vec shader runs near 90 tok/s on gfx1201",
        "KV cache uses a pos-major layout indexed by absolute position",
        "The flash-attention batch kernel tiles over visible KV positions",
    ];
    let decisions = [
        "We decided to default KV-prefix-reuse on after the byte-identity proof",
        "We chose a relative-to-top recall margin over a fixed absolute cutoff",
    ];
    for t in working {
        store.remember(Some("typing"), "fact", t, None, Some(json!({"type":"working"}))).expect("remember working");
    }
    for t in decisions {
        store.remember(Some("typing"), "fact", t, None, Some(json!({"type":"decision"}))).expect("remember decision");
    }
    let q = "GPU kernel and KV cache design choices";

    // Untyped recall MIXES both types — similarity cannot separate them.
    let all = store.recall(Some("typing"), q, 5).expect("recall all");
    assert_eq!(all.len(), 5, "all 5 notes sit in the same band → top-5 returns them");
    let kinds: std::collections::HashSet<&str> = all.iter().map(|h| h.note_type.as_str()).collect();
    assert!(
        kinds.contains("working") && kinds.contains("decision"),
        "untyped recall mixes both types (similarity does not disambiguate): {:?}",
        all.iter().map(|h| (&h.note_type, &h.text)).collect::<Vec<_>>(),
    );

    // TYPE filter returns EXACTLY the 2 decisions — reliable, non-embedding cut.
    let only_dec = store
        .recall_filtered(Some("typing"), q, 5, None, Some(NoteType::Decision), false)
        .expect("recall decisions");
    assert_eq!(only_dec.len(), 2, "type=decision returns exactly the 2 decisions");
    assert!(only_dec.iter().all(|h| h.note_type == "decision"));

    // --explain with the type filter labels the 3 working facts cut: type.
    let ex = store
        .recall_explain(Some("typing"), q, 5, 5, None, Some(NoteType::Decision), false)
        .expect("explain");
    assert_eq!(ex.returned.len(), 2, "explain.returned = the 2 decisions");
    let typed_cuts = ex.near_miss.iter().filter(|nm| nm.cut == CutReason::Type).count();
    assert!(typed_cuts >= 3, "the 3 working facts must be cut: type (got {typed_cuts})");
}

/// Additive + reversible: no type → `untyped` default (no backfill), recall of
/// an untyped note is unchanged, and `retype` relabels it (pure metadata).
#[test]
fn untyped_default_and_retype_relabels() {
    let store = store_or_skip!("retype");
    let id = store
        .remember(Some("retype"), "Note", "An untyped legacy-style note about tensors", None, None)
        .expect("remember")
        .id;
    assert_eq!(store.get(id).expect("get").note_type, "untyped", "no type → untyped (no backfill)");
    // Untyped notes recall unchanged.
    assert!(store.recall(Some("retype"), "tensors", 5).expect("recall").iter().any(|h| h.id == id));

    // retype → decision (pure metadata, reversible).
    store.retype(Some("retype"), id, NoteType::Decision).expect("retype");
    assert_eq!(store.get(id).expect("get2").note_type, "decision", "retype relabels");
    assert!(store
        .recall_filtered(Some("retype"), "tensors", 5, None, Some(NoteType::Decision), false)
        .expect("rf").iter().any(|h| h.id == id), "now found under type=decision");
    assert!(store
        .recall_filtered(Some("retype"), "tensors", 5, None, Some(NoteType::Untyped), false)
        .expect("rf2").iter().all(|h| h.id != id), "no longer found under type=untyped");

    // retype a missing id → typed NotFound (→ 404 at the handler).
    assert!(matches!(store.retype(Some("retype"), 999_999, NoteType::Working), Err(CurateError::NotFound(_))));
}

/// `SUPERSEDES` suppresses the superseded note from default recall; the chain
/// resolves to the un-superseded head; `--explain` shows the cut; the note is
/// recoverable (not deleted) via include-superseded / `unsupersede`; and with
/// no edges recall is unchanged. The payoff: stale notes leave recall.
#[test]
fn supersedes_suppresses_target_chain_resolves_and_is_recoverable() {
    let store = store_or_skip!("super");
    let q = "project decision history";
    // Three DISTINCT notes (unrelated wording → no dedup). The project holds
    // only these, so recall(k=5) returns all three regardless of similarity —
    // the test is about which ids the SUPERSEDES edges suppress.
    let b = store.remember(Some("super"), "fact", "Sourdough bread needs flour water and patience", None, None).expect("b").id;
    let a = store.remember(Some("super"), "fact", "Mountain trails get steep and cold in late autumn", None, None).expect("a").id;
    let c = store.remember(Some("super"), "fact", "Espresso extraction depends heavily on the grind size", None, None).expect("c").id;
    assert!(b != a && a != c && b != c, "seed notes deduped — pick more distinct texts (b={b} a={a} c={c})");

    // Baseline: no edges yet → all three recallable (byte-identical behavior).
    let base = store.recall(Some("super"), q, 5).expect("base");
    assert_eq!(base.iter().filter(|h| [b, a, c].contains(&h.id)).count(), 3, "no edges → all recallable");

    // a SUPERSEDES b, then c SUPERSEDES a → chain head is c.
    store.supersede(Some("super"), a, b).expect("a supersedes b");
    store.supersede(Some("super"), c, a).expect("c supersedes a");

    let after = store.recall(Some("super"), q, 5).expect("after");
    assert!(after.iter().any(|h| h.id == c), "chain head c is current");
    assert!(after.iter().all(|h| h.id != a && h.id != b), "a and b are superseded → out of recall: {:?}", after.iter().map(|h| h.id).collect::<Vec<_>>());

    // --explain shows a and b as cut: superseded, each with its superseder id.
    let ex = store.recall_explain(Some("super"), q, 5, 5, None, None, false).expect("explain");
    let sup_cuts: Vec<(i64, Option<i64>)> = ex.near_miss.iter()
        .filter(|nm| nm.cut == CutReason::Superseded)
        .map(|nm| (nm.hit.id, nm.hit.superseded_by))
        .collect();
    assert!(sup_cuts.iter().any(|(id, by)| *id == b && *by == Some(a)), "b cut: superseded by a: {sup_cuts:?}");
    assert!(sup_cuts.iter().any(|(id, by)| *id == a && *by == Some(c)), "a cut: superseded by c: {sup_cuts:?}");

    // Recoverable #1: include_superseded brings them back (not deleted).
    let incl = store.recall_filtered(Some("super"), q, 5, None, None, true).expect("incl");
    assert!(incl.iter().any(|h| h.id == b) && incl.iter().any(|h| h.id == a), "include_superseded surfaces a and b");
    // The superseded notes still exist in the store.
    assert!(store.get(b).is_some() && store.get(a).is_some(), "superseded notes are not deleted");

    // Recoverable #2: unsupersede(c, a) → a returns (b stays superseded by a).
    store.unsupersede(Some("super"), c, a).expect("release c->a");
    let rel = store.recall(Some("super"), q, 5).expect("rel");
    assert!(rel.iter().any(|h| h.id == a), "a returns to recall after unsupersede");
    assert!(rel.iter().all(|h| h.id != b), "b stays superseded (a still supersedes it)");

    // supersede idempotent; missing id → typed NotFound (→ 404).
    store.supersede(Some("super"), c, a).expect("idempotent");
    store.supersede(Some("super"), c, a).expect("idempotent again");
    assert!(matches!(store.supersede(Some("super"), c, 999_999), Err(CurateError::NotFound(_))));
    assert!(matches!(store.unsupersede(Some("super"), 999_999, a), Err(CurateError::NotFound(_))));
}

/// `DERIVES_FROM` builds the Why-Graph: `/why` returns the justification tree
/// (single- + multi-hop), `--explain` shows the derivation, and — the stronger
/// invariant — **default recall is byte-identical even WITH DERIVES_FROM edges**
/// (awareness ≠ injection). `/underive` releases a link.
#[test]
fn derives_from_why_tree_explain_and_recall_byte_identical() {
    let store = store_or_skip!("why");
    // Distinct texts (avoid dedup). d=decision; w0/w1/w2=working evidence.
    let d = store.remember(Some("why"), "fact", "We decided to default KV prefix reuse to on", None, Some(json!({"type":"decision"}))).expect("d").id;
    let w1 = store.remember(Some("why"), "fact", "Logit byte-identity was proven on the eight billion model", None, Some(json!({"type":"working"}))).expect("w1").id;
    let w2 = store.remember(Some("why"), "fact", "The FP8 residual question closed on the gemma path", None, Some(json!({"type":"working"}))).expect("w2").id;
    let w0 = store.remember(Some("why"), "fact", "Sliding window attention masks positions and never evicts", None, Some(json!({"type":"working"}))).expect("w0").id;
    assert_eq!([d, w1, w2, w0].iter().collect::<std::collections::HashSet<_>>().len(), 4, "seed deduped");

    let q = "kv reuse decision rationale";
    let before: Vec<(i64, f32)> = store.recall(Some("why"), q, 10).expect("before").iter().map(|h| (h.id, h.score)).collect();

    // D derives from W1, W2; W1 derives from W0 (a multi-hop justification).
    store.derive(Some("why"), d, &[w1, w2]).expect("derive D");
    store.derive(Some("why"), w1, &[w0]).expect("derive W1");

    // CRITICAL stronger invariant: default recall byte-identical WITH the edges.
    let after: Vec<(i64, f32)> = store.recall(Some("why"), q, 10).expect("after").iter().map(|h| (h.id, h.score)).collect();
    assert_eq!(before, after, "DERIVES_FROM must not change recall (ids/scores/order)");
    assert!(store.recall(Some("why"), q, 10).expect("r").iter().all(|h| h.derives_from.is_empty()),
        "default recall hits must omit derives_from (display-only / explain-only)");

    // --explain surfaces D's derivation.
    let ex = store.recall_explain(Some("why"), q, 10, 5, None, None, false).expect("explain");
    let dh = ex.returned.iter().chain(ex.near_miss.iter().map(|nm| &nm.hit)).find(|h| h.id == d).expect("D present");
    let mut got = dh.derives_from.clone(); got.sort();
    let mut want = vec![w1, w2]; want.sort();
    assert_eq!(got, want, "explain shows D derives from W1, W2");

    // /why D → D → [W1 → [W0], W2].
    let tree = store.why(Some("why"), d, 8).expect("why");
    assert_eq!(tree.id, d);
    let child_ids: Vec<i64> = tree.derives_from.iter().map(|n| n.id).collect();
    assert!(child_ids.contains(&w1) && child_ids.contains(&w2), "D's premises: {child_ids:?}");
    let w1node = tree.derives_from.iter().find(|n| n.id == w1).expect("w1 node");
    assert_eq!(w1node.derives_from.iter().map(|n| n.id).collect::<Vec<_>>(), vec![w0], "multi-hop: W1 → W0");
    assert!(w1node.derives_from[0].derives_from.is_empty(), "W0 is a leaf");

    // /underive D from W1 → /why D shows only W2.
    store.underive(Some("why"), d, w1).expect("underive");
    let tree2 = store.why(Some("why"), d, 8).expect("why2");
    assert_eq!(tree2.derives_from.iter().map(|n| n.id).collect::<Vec<_>>(), vec![w2], "after underive D→W1, only W2");

    // why on a missing id → typed NotFound (→ 404).
    assert!(matches!(store.why(Some("why"), 999_999, 8), Err(CurateError::NotFound(_))));
}

/// The Why-Graph traversal always terminates: a user-built cycle (`A from B`,
/// `B from A`) is flagged and not expanded, and a long chain truncates at the
/// depth cap.
#[test]
fn why_cycle_guard_and_depth_cap_terminate() {
    let store = store_or_skip!("why_cyc");
    let a = store.remember(Some("why_cyc"), "fact", "Alpha concerns volcanic geology deeply", None, None).expect("a").id;
    let b = store.remember(Some("why_cyc"), "fact", "Beta involves deep sea bioluminescent fish", None, None).expect("b").id;
    assert_ne!(a, b, "seed deduped");
    store.derive(Some("why_cyc"), a, &[b]).expect("a from b");
    store.derive(Some("why_cyc"), b, &[a]).expect("b from a"); // cycle

    let tree = store.why(Some("why_cyc"), a, 8).expect("why terminates");
    let bnode = &tree.derives_from[0];
    assert_eq!(bnode.id, b);
    let anode = &bnode.derives_from[0];
    assert_eq!(anode.id, a);
    assert!(anode.cycle, "A re-entry on the path is flagged as a cycle");
    assert!(anode.derives_from.is_empty(), "cycle node is not expanded");

    // Depth cap: a 4-chain with cap=2 truncates at the boundary.
    let topics = ["accordion music theory", "medieval tax law", "arctic jellyfish biology", "volcanic ash chemistry"];
    let c: Vec<i64> = topics.iter().map(|t| store.remember(Some("why_cyc"), "fact", &format!("Chain note about {t}"), None, None).expect("c").id).collect();
    for i in 0..3 {
        store.derive(Some("why_cyc"), c[i], &[c[i + 1]]).expect("chain");
    }
    let capped = store.why(Some("why_cyc"), c[0], 2).expect("capped");
    let c1 = &capped.derives_from[0];
    assert_eq!(c1.id, c[1]);
    let c2 = &c1.derives_from[0];
    assert_eq!(c2.id, c[2]);
    assert!(c2.truncated, "depth cap truncates at the boundary");
    assert!(c2.derives_from.is_empty(), "truncated node is not expanded");
}

/// SUPERSEDES suppression must not silently shrink recall below `k` when fresh
/// relevant notes remain: 7 relevant notes, 2 of the top-5 superseded → a
/// `recall(k=5)` must backfill to **5 fresh** hits (not 3), drawing the next
/// fresh notes up into the freed slots.
#[test]
fn supersedes_recall_backfills_to_k() {
    let store = store_or_skip!("backfill");
    // 7 notes all relevant to the query, distinct enough to dodge dedup.
    let texts = [
        "Volcanic basalt forms from rapidly cooling lava flows",
        "The accordion produces sound through a bellows mechanism",
        "Medieval guilds tightly controlled craft apprenticeships",
        "Deep sea anglerfish lure prey with bioluminescence",
        "Espresso crema depends on fresh coffee bean oils",
        "Suspension bridges distribute their load through steel cables",
        "Origami tessellations fold from a single uncut sheet",
    ];
    let ids: Vec<i64> = texts.iter().map(|t| store.remember(Some("backfill"), "fact", t, None, None).expect("remember").id).collect();
    assert_eq!(ids.iter().collect::<std::collections::HashSet<_>>().len(), 7, "seed deduped");

    let q = "recall design note";
    // Find the current top-5, then supersede 2 of them with two of the others.
    let top5: Vec<i64> = store.recall(Some("backfill"), q, 5).expect("top5").iter().map(|h| h.id).collect();
    assert_eq!(top5.len(), 5, "baseline returns 5");
    // Supersede the first two top-5 notes (the superseders are notes NOT in the
    // top-5, so they don't themselves get suppressed and remain fresh).
    let superseders: Vec<i64> = ids.iter().copied().filter(|id| !top5.contains(id)).collect();
    assert!(superseders.len() >= 2, "need ≥2 notes outside top-5 to supersede with");
    store.supersede(Some("backfill"), superseders[0], top5[0]).expect("s0");
    store.supersede(Some("backfill"), superseders[1], top5[1]).expect("s1");

    // 5 fresh notes still exist (7 − 2 superseded). recall(k=5) must return 5.
    let after = store.recall(Some("backfill"), q, 5).expect("after");
    assert_eq!(after.len(), 5, "suppression must backfill to k=5 (got {}): {:?}",
        after.len(), after.iter().map(|h| h.id).collect::<Vec<_>>());
    // None of the returned are the superseded ones.
    assert!(after.iter().all(|h| h.id != top5[0] && h.id != top5[1]), "superseded notes must not appear");
    assert!(after.iter().all(|h| h.superseded_by.is_none()), "no superseded note in the result");
}

/// Supersession composes with the type filter: a superseded note is suppressed
/// even when it matches the requested type (the `superseded` gate has priority),
/// and `include_superseded` brings it back under that type.
#[test]
fn supersedes_composes_with_type_filter() {
    let store = store_or_skip!("super_ty");
    let q = "project decision";
    let old = store.remember(Some("super_ty"), "fact", "Old decision about flour and bread", None, Some(json!({"type":"decision"}))).expect("old").id;
    let new = store.remember(Some("super_ty"), "fact", "New decision about mountain trails", None, Some(json!({"type":"decision"}))).expect("new").id;
    assert_ne!(old, new, "seed deduped");
    store.supersede(Some("super_ty"), new, old).expect("supersede");

    // type=decision: only `new` (old is superseded, suppressed despite matching type).
    let dec = store.recall_filtered(Some("super_ty"), q, 5, None, Some(NoteType::Decision), false).expect("dec");
    assert!(dec.iter().any(|h| h.id == new), "current decision present");
    assert!(dec.iter().all(|h| h.id != old), "superseded decision suppressed even with matching type");

    // include_superseded + type=decision → old returns under its type.
    let incl = store.recall_filtered(Some("super_ty"), q, 5, None, Some(NoteType::Decision), true).expect("incl");
    assert!(incl.iter().any(|h| h.id == old), "include_superseded surfaces the superseded decision");
}

/// The adaptive threshold cuts the clear noise floor while dropping **zero**
/// relevant notes (the False-Negative riegel), `--explain` labels each cut, and
/// the disable path (`margin=None`) is byte-identical to plain top-k recall.
#[test]
fn recall_threshold_cuts_noise_keeps_relevant_and_disable_is_pure_topk() {
    let store = store_or_skip!("threshold");
    // 3 clearly on-topic (relevant) + 4 clearly off-topic (noise).
    let relevant = [
        "Vulkan compute shaders run on AMD RDNA4 gfx1201",
        "The GPU dispatches Q4_K matrix-vector multiply kernels",
        "Compute pipelines bind descriptor sets for storage buffers",
    ];
    let noise = [
        "A sourdough bread recipe needs flour water salt and time",
        "The cat sat quietly on the warm sunny windowsill",
        "Coffee brewing temperature affects the extraction yield",
        "A mountain trail winds through the autumn forest",
    ];
    for t in relevant.iter().chain(noise.iter()) {
        store.remember(Some("threshold"), "fact", t, None, None).expect("remember");
    }
    let q = "GPU compute kernel dispatch on Vulkan";

    // Measure the score landscape (margin=None, wide) to pick a margin that
    // sits below the lowest relevant score but above the noise floor.
    let scan = store.recall_explain(Some("threshold"), q, 7, 5, None, None, false).expect("scan");
    let all: Vec<(&str, f32)> = scan
        .returned.iter().map(|h| (h.text.as_str(), h.score))
        .chain(scan.near_miss.iter().map(|nm| (nm.hit.text.as_str(), nm.hit.score)))
        .collect();
    let is_relevant = |t: &str| relevant.iter().any(|r| *r == t);
    let min_relevant = all.iter().filter(|(t, _)| is_relevant(t)).map(|(_, s)| *s).fold(f32::MAX, f32::min);
    let max_noise = all.iter().filter(|(t, _)| !is_relevant(t)).map(|(_, s)| *s).fold(f32::MIN, f32::max);
    let top = all.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
    eprintln!("[threshold] top={top:.4} min_relevant={min_relevant:.4} max_noise={max_noise:.4}");
    // The set must be separable at all for the test to be meaningful.
    assert!(min_relevant > max_noise,
        "seed not separable (min_relevant {min_relevant} ≤ max_noise {max_noise}); pick clearer notes");

    // Conservative margin: keep everything down to the lowest relevant note
    // (err to include), i.e. margin ≥ top − min_relevant, with a little slack.
    let margin = (top - min_relevant) + 0.01;

    // recall WITH the margin: 0 relevant dropped (the riegel), noise cut.
    let hits = store.recall_with_margin(Some("threshold"), q, 7, Some(margin)).expect("recall margin");
    for r in relevant {
        assert!(hits.iter().any(|h| h.text == r), "FALSE NEGATIVE: dropped a relevant note: {r}");
    }
    assert!(hits.iter().all(|h| is_relevant(&h.text)),
        "noise survived the threshold: {:?}", hits.iter().map(|h| &h.text).collect::<Vec<_>>());

    // --explain with the margin: threshold is the real value, and at least one
    // near-miss is labelled `threshold` (the cut noise).
    let ex = store.recall_explain(Some("threshold"), q, 7, 5, Some(margin), None, false).expect("explain margin");
    assert!(ex.threshold.is_some(), "explain must report the active threshold value");
    assert!(ex.near_miss.iter().any(|nm| nm.cut == CutReason::Threshold),
        "noise below the threshold must be labelled cut: threshold");

    // Disable path: margin=None is byte-identical to plain top-k recall.
    let disabled = store.recall_with_margin(Some("threshold"), q, 7, None).expect("disabled");
    let plain = store.recall(Some("threshold"), q, 7).expect("plain");
    let ids = |v: &[vulkanforge::server::memory::Hit]| v.iter().map(|h| h.id).collect::<Vec<_>>();
    assert_eq!(ids(&disabled), ids(&plain), "margin=None must equal plain top-k recall");
    assert!(disabled.len() > hits.len(), "the threshold must actually cut something vs disabled");
}

/// Conservative-margin guardrail: a relevant note scoring only slightly below
/// the top must NOT be cut (err to include). A generous margin keeps a thin
/// gap; a near-zero margin would drop it — we assert the inclusive behavior.
#[test]
fn recall_threshold_errs_to_include_borderline_relevant() {
    let store = store_or_skip!("borderline");
    // Two closely-related notes (small score gap) — both relevant to the query.
    store.remember(Some("borderline"), "fact", "Vulkan compute pipelines dispatch Q4_K kernels on RDNA4", None, None).expect("a");
    store.remember(Some("borderline"), "fact", "The GPU runs matrix-vector multiply shaders for inference", None, None).expect("b");
    let q = "GPU compute kernel dispatch";

    // A generous margin (0.2 similarity) is wide enough that the close second
    // note stays in — the threshold only trims the clear gradient, not a thin
    // gap between two relevant notes.
    let hits = store.recall_with_margin(Some("borderline"), q, 5, Some(0.2)).expect("recall");
    assert_eq!(hits.len(), 2, "a generous margin must keep both close-relevant notes (err to include)");
}
