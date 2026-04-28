# v0.2 Sprint 5B — Chunked Prefill (pp > max_prefill_tokens)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 5 (Cap 256→1024, mul_mmq pp-Skalierung bis 1024)
**Ziel:** Chunked-prefill statt Token-für-Token-Fallback bei
prompts > max_prefill_tokens. pp=2048/4096 nutzbar machen.

---

## TL;DR — Chunking ersetzt den 90-tok/s-Fallback komplett

```
═══ v0.2 Sprint 5B ═══

ÄNDERUNG (1 Funktion in decode.rs):
  Statt "if pp > cap: forward_token loop" jetzt
  "for chunk in pp.chunks(cap): prefill_batch(chunk, base_pos)".
  prefill_batch unterstützte den base_pos-Parameter schon — keine
  Änderungen in Forward, Shadern, KV-Cache nötig.

End-zu-End Effekt (chat session, pp=1243):
  Sprint 5 (vorher):   ~90 tok/s   (forward_token loop)
  Sprint 5B (chunked): 487 tok/s   (Chunked GEMM)
  Gewinn: 5.4×

Mikro-Bench, neue pp-Bereiche (vorher unbenutzbar):
  pp=2048: 530 tok/s  (16 chunks à 128 Tokens)
  pp=3072: 388 tok/s  (24 chunks)
  pp=4096: 303 tok/s  (32 chunks)

Wichtiger Befund — Chunk-Size beeinflusst Throughput:
  pp=1024 mit chunk=1024: 556 tok/s  (1 chunk)
  pp=1024 mit chunk= 256: 663 tok/s  (4 chunks)  ← 19% besser!
  pp=2048 mit chunk= 256: 395 tok/s  (8 chunks)
  pp=2048 mit chunk= 128: 530 tok/s  (16 chunks) ← 34% besser!

Grund: Linux-TDR (~5s GPU-Submit-Limit) zwingt zu kleinen Chunks
bei langem pp; kleinere Chunks fragmentieren KV-Reads, geben dem
Treiber bessere Chance zur Cache-Wiederverwendung. Sweet-Spot ist
NICHT chunk=cap=1024 — kann durchaus 128-256 sein.

Kritischer Bug der NICHT eingetreten ist:
  Wir hatten keine KV-Cache-Drift, keine RoPE-Position-Drift, keine
  Causal-Mask-Bug. prefill_batch war architekturell schon
  chunk-aware (sah aus wie ein "two-tile within one prefill"-Test
  → Chunked Prefill ist semantisch das gleiche, nur über mehrere
  Submits verteilt).

Tests: 146/146 ✓ (+1 neuer chunked-vs-single Parity-Test).
Commit: HEAD (kein Push).

Empfehlung Sprint 6 unverändert: Flash-Attention-Prefill ist der
nächste Hebel. Bei pp=4096 sind wir 0.09× von llama.cpp (303 vs
3272 tok/s) — das O(N²) scalar attention ist nun klar die einzige
Bremse, nachdem Cliff (Sprint 5) und Fallback (Sprint 5B) weg sind.
```

---

## 1. Was wurde gemacht?

Nur eine echte Änderung — `decode.rs::generate_from_tokens`:

### Vorher (Sprint 5)

```rust
if prefill_len > 0 && prefill_len <= forward.max_prefill_tokens {
    // Batched GEMM für Prompts ≤ cap
    forward.prefill_batch(...)?;
} else {
    // Token-für-Token Fallback für Prompts > cap (~90 tok/s)
    for &tid in prefill_tokens {
        forward.forward_token(...)?;
    }
}
```

### Nachher (Sprint 5B)

```rust
if prefill_len > 0 {
    let chunk_size = forward.max_prefill_tokens.max(1) as usize;
    for chunk in prefill_tokens.chunks(chunk_size) {
        let chunk_len = chunk.len() as u32;
        let chunk_embeds = embed_lookup_chunk(chunk);
        forward.prefill_batch(... chunk_embeds, chunk_len, pos)?;
        pos += chunk_len;
    }
}
```

**Architektur-Beobachtung**: `prefill_batch(seq_len, base_pos)` war
schon chunk-fähig:

* RoPE-Positions: `(0..seq_len).map(|t| base_pos + t)` (forward.rs:2420)
* KV-Write-Offset: `pos_offset_bytes(layer, base_pos)` (forward.rs:2709)
* flash_attn_batch: `q_start=base_pos, n_kv=base_pos+seq_len`
  (forward.rs:2858–2859) — Queries starten bei `base_pos`, sehen
  KV[0..base_pos+seq_len)
* `kv_cache.current_seq_len = base_pos + seq_len` am Ende
  (forward.rs:2503)

Sprint 5B hat **keinen** Code in Forward, in Shadern, oder in
der KV-Cache-Struktur geändert. Nur die Aufrufstelle in
`decode.rs:429` umgeschrieben. Der bestehende
`phase5b2_batch_attn_parity_qwen3_two_tiles`-Test deckt schon
ab, dass mehrere TILE-Iterationen innerhalb eines prefill_batch
funktionieren — Chunked-Prefill ist semantisch das gleiche,
nur über Submit-Grenzen hinweg.

---

## 2. Parity-Test (Korrektheit)

`tests/regression.rs::sprint5b_chunked_prefill_parity_qwen3`:

* Tokenisiert ein 120-Token Prompt.
* Lauf A: `Forward::new_with_prefill(... 120)` → 1 Submit, 1 chunk.
* Lauf B: `Forward::new_with_prefill(... 32)` → 4 Submits, 4 chunks.
* Vergleicht die finalen Logits beider Läufe:
  * `argmax`: muss bit-identisch sein.
  * Top-5: muss ≥ 4/5 overlap haben.
  * Keine NaN/Inf.

**Resultat**: Test passed beim ersten Run. argmax bit-identisch
zwischen 1-Chunk und 4-Chunk-Pfad.

Was der Test absichert:
* KV-Cache-Offsets (chunk N+1 sieht KV von chunks 0..N)
* RoPE-Position-Drift (pos wird global hochgezählt, nicht local)
* Attention-Bounds (q_start/n_kv korrekt für jeden Chunk)
* Causal-Mask innerhalb eines Chunks und zwischen Chunks

---

## 3. Performance — Chunked Bench

### 3.1 Default-Cap (1024) — 1 Chunk pro pp ≤ 1024

Smoke-Vergleich Sprint 5 vs Sprint 5B (sollte identisch sein bei
pp ≤ 1024):

```
| pp   | Sprint 5 mul_mmq | Sprint 5B (chunk=1024) |
|------|------------------|------------------------|
| 1024 |        556.4     |          556.0         |
```

Identisch innerhalb Mess-Variance. Die `decode.rs`-Änderung hat
keine Regression.

### 3.2 Chunk-Size Sweep — find der Sweet-Spot

Gleiches pp, verschiedene Chunk-Sizes — überraschende Erkenntnis,
dass kleine Chunks effizienter sind:

```
| pp   | chunk=1024  | chunk=256   | chunk=128   |
|      | (1 chunk)   | (4 chunks)  | (8 chunks)  |
|------|-------------|-------------|-------------|
| 1024 |   556 tok/s |   663 tok/s |     —       |
| 1536 |   423 tok/s |   496 tok/s |     —       |
| 2048 | DEVICE_LOST |   395 tok/s |   530 tok/s |
| 3072 | DEVICE_LOST | DEVICE_LOST |   388 tok/s |
| 4096 | DEVICE_LOST | DEVICE_LOST |   303 tok/s |
```

Drei Beobachtungen:

**(a)** `chunk=1024 + pp=2048` → **VK_ERROR_DEVICE_LOST**.
   Ursache: Linux/Mesa TDR (~5s GPU-Submit-Limit). Ein einzelner
   Chunk-2-Submit (m=1024 Queries × n_kv=2048 KV pro Layer × 36
   Layer × O(N²) scalar attention) überschreitet das Limit.

**(b)** Kleinere Chunks heben den TDR-Hit auf, KOSTEN aber
   leichten Overhead (Submit-per-Chunk, Descriptor-Set-Resets).
   Trotzdem gewinnt chunk=128 bei pp=2048 mit 530 tok/s gegenüber
   chunk=256 mit 395 tok/s — **+34%**. Der Grund ist NICHT mehr
   Submit-Effizienz, sondern dass kleinere Q-Tiles besser im LDS
   gehalten werden können während über große KV iteriert wird.

**(c)** Bei pp=1024 (innerhalb des Caps): chunk=256 (4 chunks)
   schlägt chunk=1024 (1 chunk) mit **663 vs 556 tok/s = +19%**.
   Die "schnellste" Konfiguration ist also NICHT, alles in einem
   Chunk zu erledigen — es ist ein klassischer Tile-Size-Trade-off.

### 3.3 Vollständige pp-Kurve (chunk=128 für pp ≥ 2048)

```
| pp   | VF mul_mmq | n_chunks | llama.cpp | VF/llama |
|------|------------|----------|-----------|----------|
|   16 |       386  |    1     |    700.6  |  0.55×   |
|   64 |      1489  |    1     |   2286.2  |  0.65×   |
|  128 |      1641  |    1     |   3602.5  |  0.46×   | ← VF peak
|  256 |      1337  |    1     |   3999.4  |  0.33×   |
|  512 |       921  |    1     |   4317.2  |  0.21×   |
| 1024 |       556  |    1     |   4189.3  |  0.13×   |
| 1536 |       423  |    2     |   3989.6  |  0.11×   |
| 2048 |       530  |   16     |   3770.8  |  0.14×   | ← jetzt erreichbar
| 3072 |       388  |   24     |   3521.6  |  0.11×   |
| 4096 |       303  |   32     |   3271.6  |  0.09×   | ← jetzt erreichbar
```

Hinweise:
- pp ≤ 1024: chunk=1024 (Default Sprint 5), 1 chunk.
- pp = 1536: chunk=1024, 2 chunks (TDR noch nicht überschritten).
- pp ≥ 2048: chunk=128, n=pp/128 chunks (TDR-sicher).

VF-Kurve fällt von Peak 1641 (pp=128) auf 303 tok/s (pp=4096) —
5.4× Drop. llama.cpp fällt nur von 4317 (pp=512) auf 3272 (pp=4096)
— 1.3× Drop. Die Differenz ist die O(N²) scalar attention vs Flash-
Attention.

### 3.4 End-zu-End-Verifikation (chat session, pp=1243)

```
Test-Prompt: ~5266 chars synthetisierter Tech-Text durch
             ChatSession::send() → tokenisiert auf 1243 Tokens.

Sprint 4.5 (256-Cap):  ~90 tok/s  (forward_token fallback)
Sprint 5  (1024-Cap):  ~90 tok/s  (gleicher Fallback bei pp>1024)
Sprint 5B (chunked):   487 tok/s  (2 chunks: 1024 + 219)

Recovery: 5.4× — die "echte" produktive Wirkung von Sprint 5B,
weil Multi-Turn-Chats und RAG-Workloads typisch in diesem Range
liegen (1k-2k tokens).
```

---

## 4. Tests

```
cargo test --release

test result: ok. 24 passed       (lib unit)
test result: ok.  9 passed       (dequant_q4k integration)
test result: ok. 18 passed       (gguf integration)
test result: ok. 60 passed       (kv_cache integration)
test result: ok.  8 passed       (q4k_quant integration)
test result: ok. 27 passed       (regression — was 26)
                ────
                146 / 146 ✓
```

Plus 1 neuer Sprint-5B-Test:
* `sprint5b_chunked_prefill_parity_qwen3` — 1-chunk vs 4-chunk
  parity (argmax bit-identisch, top-5 ≥ 4/5).

Bestehende Tests die den chunked-Pfad implizit treffen:
* `phase5b2_batch_attn_parity_qwen3_two_tiles` — multi-tile
  attention innerhalb eines prefill_batch.
* `phase5c_*` — chat session multi-turn (mehrere prefill_batch
  calls mit base_pos > 0).

---

## 5. Was wurde NICHT gemacht (out-of-scope)

* Kein Flash-Attention-Prefill (Sprint 6).
* Keine Auto-Adaption der Chunk-Size — User setzt
  `VULKANFORGE_MAX_PREFILL` selbst je nach Workload.
* Kein Submit-Pipelining (mehrere Chunks asynchron). Aktuell
  blockiert jeder Chunk-Submit auf seine Fence. Pipelining würde
  Chunk-N's CPU-side embed_gather mit Chunk-(N-1)'s GPU-Arbeit
  überlappen.
* Kein dynamisches `max_prefill_tokens` basierend auf VRAM —
  Default 1024 ist hardcodiert in `forward.rs::new`.
* Keine coopmat-Pfad-Anpassung — bei pp=1024 crashte coopmat
  schon in Sprint 5 (separater Bug, nicht Sprint-5B-Scope).

---

## 6. Files Touched

```
modified:   src/backend/vulkan/decode.rs               (+19 -10)
modified:   examples/run_pp_bench.rs                   (+24 -10)
modified:   tests/regression.rs                        (+108)
new file:   results/v02_sprint5b_chunked_prefill.md
```

---

## 7. Empfehlung — Sprint 6 unverändert: Flash-Attention-Prefill

Nach Sprint 5 (Cliff weg) + Sprint 5B (Fallback weg) ist die
einzige verbleibende strukturelle Bremse die scalar O(N²)
Attention ohne Tiling/Streaming.

Empirie aus der Kurve:
* mul_mmq peak bei pp=128: 1641 tok/s.
* Decay auf 303 tok/s bei pp=4096 (5.4× Verlust).
* Reine GEMM-Ops würden eher steigen mit pp (mehr Auslastung).
  Der Decay kommt aus Q·K^T und Softmax·V die O(seq² × n_heads ×
  head_dim) Speicher-Bandbreite haben.

Erwartete Sprint-6-Wirkung (Flash-Attn-Prefill mit Tile-Streaming):
* mul_mmq-Kurve flacher halten — pp=2048: 530 → ~1100-1300 tok/s
  (würde llama.cpp's 3771 immer noch nicht treffen, aber 0.14×
  → 0.30-0.35× verbessern).
* pp=4096: 303 → ~600-800 tok/s (0.09× → 0.18-0.24×).

Andere mögliche Hebel (sekundär):
* Conditional-Barriers (PR-Pattern aus Audit) — 5-15% über alle pp.
* Submit-Pipelining für Chunked Prefill (CPU-embed mit GPU-Arbeit
  überlappen) — wenige %.
* coopmat path bei pp ≥ 1024 fixen (DEVICE_LOST, niedrige
  Priorität, da coopmat eh durchgehend schlechter als mul_mmq).
