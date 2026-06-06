# Sprint 8 — v0.6.0 RELEASED: 3-Flag-Default-Flip (Prefill 198→1073/604 t/s out of the box)

**Datum:** 2026-06-06 · **Push-Go von mg erteilt** · gemma4-26B-A4B Q3_K_M · `VULKANFORGE_KV_FP8=1`.

---

## VERDIKT (die 4 Pflichtfragen)

1. **Geflippt + gepusht:** Die 3 Flags `VF_PREFILL_FULL_BATCHED` + `VF_MOE_GLU_BATCHED` +
   `VF_MOE_GATHER_COMBINE` als **gekoppeltes Set auf default-ON** (exakt die Sprint-5-validierte Konfiguration;
   Semantik `=0`/`=false` = Opt-out, Muster wie `VF_MOE_GROUPED` v0.5.3). **Opt-out erhalten** — alle Legacy-
   Pfade (per-Token-Workaround, GLU-Loop, Scatter-Loop) bleiben als Fallback im Tree. Gepusht: **HEAD inkl.
   `eeef090`** (tiled-FA-Plattform, gated-OFF, nullrisiko — kein Grund für Git-Chirurgie; Default-Empfehlung
   des Briefs). Stack: `cadd632` (v0.5.9) → `afa536d` (Sprint 5) → `eeef090` (Sprint 7) → Flip+Release-Commit.
2. **Re-Validierung des geflippten Defaults (ALLE GRÜN, OHNE Flags gesetzt):**
   - Recall: Paris ✓ · 391 ✓ · Jupiter ✓.
   - 5-Smoke (ids 1/3/6/9/14): alle kohärent ✓ (Outputs identisch zur Sprint-5-Validierung).
   - `cargo test --release --lib -- --test-threads=1`: **273/273** ✓.
   - Multichunk-2048: kohärent ✓.
   - **Perf-Confirm: 1073 t/s @p512 / 604 @p2048** (Median 3, clean, VRAM-Gates) — ≥ Soll 1025/602 ✓;
     definitiv nicht der alte Pfad.
   - **Opt-out-Check** (alle 3 Flags=0): kohärent (Paris + doc512) bei 201 t/s = der alte v0.5.9-Default-Pfad ✓
     (inkl. des dokumentierten FP-Reorder-Output-Wortlaut-Unterschieds — beide Pfade korrekt).
3. **Release:** Version **v0.6.0** (Cargo.toml + Lock), Tag `v0.6.0` gepusht, **GitHub-Release „Latest"**:
   https://github.com/maeddesg/vulkanforge/releases/tag/v0.6.0 · Docs revidiert: **CHANGELOG** (v0.6.0-Block
   mit ehrlichen Zahlen + Output-Hinweis + tiled-FA-Negativ), **README** (3 Flags default-ON + Opt-out; tiled-FA
   als „experimentell, NICHT empfohlen"), **NEU `docs/prefill_arc_2026_06.md`** (der Arc 198→1073/604 mit
   Mechanismen + das Attention-Occupancy-Negativ als dokumentierte Sackgasse inkl. Lektion und
   Folge-Richtungen, damit es nicht naiv wiederholt wird). Lizenz UNVERÄNDERT.
4. **Release-Notes-Kernaussage (ehrlich):** Prefill gemma4-26B **+229 %/+130 %** vs v0.5.9-Default bzw.
   **5,4×/3,4×** vs v0.5.8 (198→1073 @p512, 178→604 @p2048) = ~37 %/~21 % von llama-server (war 6–7 %).
   **Output-Hinweis:** GLU+Gather byte-identisch; der FULL_BATCHED-Anteil ÄNDERT den Default-Greedy-Output vs
   ≤v0.5.9 (FP-Reorder + Routing-Flips, cos ~0,93–0,99, kein Cliff, beide Pfade validiert korrekt).
   **tiled-FA-Flags:** experimentell, default-OFF, „NICHT für Performance" (−31…−36 %, occupancy-bound,
   Sprint-7-Negativ offen dokumentiert). Dense-Modelle (head_dim 128) unberührt (Qwen3-8B-Smoke grün).

## Console-Summary
```
[Phase 1: 3 Flags gekoppelt default-ON (=0-Opt-out, v0.5.3-Muster); Legacy-Pfade als Fallback erhalten]
[Phase 2 ALLE GRÜN: Paris/391/Jupiter ✓ | 5-Smoke ✓ | 273/273 ✓ | Multichunk ✓ | Perf-Confirm 1073/604 t/s
 (Soll 1025/602) ✓ | Opt-out: 3×=0 → alter Pfad kohärent @201 t/s ✓]
[Phase 3: v0.6.0; CHANGELOG ehrlich (Zahlen + Output-Hinweis + tiled-FA-Negativ); README-Flag-Tabelle;
 NEU docs/prefill_arc_2026_06.md (Arc + Occupancy-Sackgasse als Lessons-Doc); Lizenz unangetastet]
[Phase 4: Commit + Tag v0.6.0 + Push HEAD (inkl. eeef090, gated-off) + GitHub-Release Latest]
→ gemma4-26B-Prefill out-of-the-box: 198→1073 t/s @p512 (5,4×) / 178→604 @p2048 (3,4×) = 37%/21% llama-serve
→ Offen bleibt: run_pp_bench.rs (06-05), SIGSEGV-Teardown-Backlog, optionaler FA-Occupancy-Tuning-Sprint
```
