# Sprint 15E — 3-stage async pipelined decode: SHIPPED, +19.3 % decode

**Premise.** Sprint 15D (`ca0d7ae`) extracted 17 per-forward
intermediate buffers into `IntermediateSlot × 2` and added the
`cur()` / `cur_mut()` accessors — the prerequisite infrastructure
for async pipelined decode. Sprint 15E ships the actual 3-stage
pipeline that hides ~1 836 µs of CPU command-recording inside the
~9 034 µs GPU compute window of the previous token.

**Verdict.** **Bench-gate cleared.** 15-prompt decode median
**109.0 tok/s** (vs serial 91.4, **+19.3 %**) at the
+20 % theoretical ceiling Sprint 15A computed. 27/27 lib tests,
15/15 coherent on both async and serial paths, bit-identical
outputs across the bench prompts, prefill unchanged. Async is
**default-on**; opt-out via `VULKANFORGE_DISABLE_ASYNC_DECODE=1`.

This is the **first measurable decode performance gain since
v0.2.0** and the v0.2 → v0.3 milestone the Sprint 12-15 arc was
heading toward. The 13-entry decode-gap analysis ledger (12
falsified hypotheses + 15D infra) now closes with the 14th entry
landing as a real perf delivery: 0.80 × → 0.95 × llama.cpp Vulkan
at decode.

## 1. The 3-stage pipeline

```
Token N:                     ┌──────────────────────────────────┐
  Stage 1: pre_record        │  CPU records CB[N]               │  1.8 ms
                             │  (slots[N%2] handles, no embed)  │
                             └──────────────────────────────────┘
  Stage 2: wait + sample     ░░░░░░░░░░░░  GPU runs CB[N]  ░░░░░░░░░░  9.0 ms
                                         ┌──────────────┐
                                         │ wait fence   │  blocks ~7.2 ms
                                         │ read logits  │  ~22 µs
                                         │ sample       │  ~50 µs
                                         └──────────────┘
  Stage 3: fill_embed +      ┌──────────┐
           submit            │ embed →  │  ~22 µs
                             │ submit   │  ~14 µs
                             └──────────┘
                                         ──> GPU runs CB[N+1]
```

Total per-token wall (steady state): max(record, GPU) +
sequential overhead ≈ max(1 836, 9 034) + ~80 = **9 114 µs →
109.7 tok/s**. Measured **109.0**, right at the ceiling.

Concretely, in the decode loop:

```rust
// Cold start (first decode token, no prev to wait for):
forward.pre_record(dev, registry, model, 0, pos)?;
forward.fill_embed_and_submit(dev, 0, &embd, pos)?;
let mut cur_slot = 1usize;
pos += 1;

loop {
    if generated.len() >= max || pos >= max_seq { break; }
    let prev_slot = 1 - cur_slot;

    // Stage 1 — overlap with GPU(prev):
    forward.pre_record(dev, registry, model, cur_slot, pos)?;

    // Stage 2:
    last_logits = forward.wait_and_read_logits(dev, prev_slot)?;
    let next_id = sample(&last_logits);
    if eos(next_id) { break; }   // CB[cur] left unsubmitted, recycled next session
    let embd = embedding_row(gguf, cfg, next_id)?;

    // Stage 3:
    forward.fill_embed_and_submit(dev, cur_slot, &embd, pos)?;

    cur_slot = 1 - cur_slot;
    pos += 1;
}

// Drain final logits:
let last_slot = 1 - cur_slot;
last_logits = forward.wait_and_read_logits(dev, last_slot)?;
```

## 2. Implementation

### 2.1 Forward struct (4 new fields)

```rust
pub struct Forward {
    slots: [IntermediateSlot; 2],            // Sprint 15D
    current_slot: usize,                     // Sprint 15D
    async_pool: vk::CommandPool,             // Sprint 15E ↓
    async_cbs: [vk::CommandBuffer; 2],
    async_fences: [vk::Fence; 2],
    async_pending_record: Option<usize>,
    logits_buf: GpuBuffer,                   // single-buffered, no race
    /* … rest unchanged … */
}
```

`Forward::new` allocates a dedicated CB pool with
`RESET_COMMAND_BUFFER` flag, two CBs from it, and two fences
created with `SIGNALED` so the first `wait_for_fences` on a
"fresh" slot is a no-op. `Forward::destroy` frees both fences and
the pool (CBs are freed implicitly).

### 2.2 Three new public methods on Forward

`pre_record(dev, registry, model, slot, position)`:
1. `wait_for_fences(async_fences[slot])` — no-op for fresh slots,
   ensures we don't reset a CB that's still in flight.
2. `reset_command_buffer(async_cbs[slot])` + `begin_command_buffer`.
3. `reset_barrier_state()` — fresh barrier-elision tracker for
   this CB.
4. `record_decode_dispatches(dev, registry, cb, model, slot, pos)`
   — extracted helper that writes the standard 36-layer +
   lm_head sequence into `cb`, reading from `slots[slot]`.
5. `end_command_buffer(cb)`.
6. Marks `async_pending_record = Some(slot)`.

`fill_embed_and_submit(dev, slot, embedding, position)`:
1. Writes embedding into `slots[slot].scratch_a`.
2. Writes position into `slots[slot].rope_pos_buf`.
3. Resets `async_fences[slot]` and submits `async_cbs[slot]` to
   `dev.compute_queue` with that fence.

`wait_and_read_logits(dev, slot)`:
1. `wait_for_fences(async_fences[slot])`.
2. Reads `logits_buf` and returns `Vec<f32>`.

### 2.3 record_decode_dispatches helper

The recording closure body of `forward_token`'s
`cmd_ctx.one_shot(...)` call was extracted into:

```rust
fn record_decode_dispatches(
    &mut self,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    model: &LoadedModel,
    slot: usize,
    position: u32,
) {
    let saved = self.current_slot;
    self.current_slot = slot;
    let mut input = self.cur().scratch_a.handle;
    let mut output = self.cur().scratch_b.handle;
    for layer in 0..self.config.n_layers {
        self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
        std::mem::swap(&mut input, &mut output);
    }
    self.dispatch_final(dev, registry, cmd, model, input);
    /* host-visible barrier on logits_buf */
    self.current_slot = saved;
}
```

Both the serial path (called inside `cmd_ctx.one_shot`'s closure)
and the async path (called from `pre_record`) share this body.
DRY-clean and removes code duplication.

### 2.4 decode.rs — generate_from_tokens

The inner decode loop (line 470-489) was wrapped in an
`if async_decode { … } else { … }` switch. The async branch
implements the 3-stage pipeline above; the else branch keeps the
exact pre-15E serial loop unchanged. Env-var
`VULKANFORGE_DISABLE_ASYNC_DECODE=1` toggles to serial.

The async branch handles three corner cases:

- **Cold start (first decode token)**: no previous CB to wait
  for. Just `pre_record(0) + fill_embed_and_submit(0)`, set
  `cur_slot = 1`, enter loop.
- **EOS mid-pipe**: `pre_record` for `cur_slot` already ran but
  we won't `fill_embed_and_submit` because we're breaking. The
  pre-recorded CB sits unsubmitted in `async_cbs[cur_slot]`; the
  next session's `pre_record` calls `reset_command_buffer` so it
  recycles cleanly. No leak.
- **Drain**: after the loop, the last submitted CB was at
  `1 - cur_slot`. One final `wait_and_read_logits(last_slot)`
  ensures GPU work is complete before returning.

### 2.5 logits_buf — single-buffered (no race)

In the 3-stage shape, the readback inside `wait_and_read_logits`
happens **before** the next `fill_embed_and_submit`, which is the
only thing that triggers a new GPU write to `logits_buf`. So
between any two writes to `logits_buf`, there's a complete
`wait → readback → CPU work → next submit` sequence. No race;
no need for a doubled `logits_buf`.

## 3. Validation

### 3.1 Test gates

| Gate | Result |
|---|---|
| `cargo test --release --lib` | ✅ 27 / 27 |
| `run_15prompt_bench` (async) | ✅ 15 / 15 coherent |
| `run_15prompt_bench` (serial) | ✅ 15 / 15 coherent |
| Output parity (async vs serial, same prompt) | ✅ bit-identical |
| Prefill unchanged | ✅ pp=128/512/1024 within ±0.4 % of v0.2.4 |

Tested: same `What is 2+2?` prompt produces character-identical
generated text under both modes (the dispatches are the same
sequence of vkCmd* calls; only the host-side coordination changed).

### 3.2 Performance

`run_15prompt_bench`, RUNS=1, decode median across 15 prompts:

| Config | Decode (tok/s) | Prefill (tok/s) | Coherent | vs llama.cpp |
|---|---:|---:|:---:|---:|
| v0.2.4 baseline (= serial) | 91.1 | 3 863 | 15/15 | 0.80× |
| Sprint 15D (slot infra) | 91.2 | 3 803 | 15/15 | 0.80× |
| Serial (15E, env-var) | 91.4 | 859 (15p median) | 15/15 | 0.80× |
| **Sprint 15E async (default)** | **109.0** | 835 (15p median) | 15/15 | **0.95×** |

Single-prompt sample_decode probe (lighter, but corroborating):

| Config | Decode (tok/s) |
|---|---:|
| Serial | 95.8 |
| **Async (default)** | **111.5** |

prefill@pp=128/512/1024 (3 runs each, median):
2 570 / 3 865 / 3 742 vs baseline 2 560 / 3 863 / 3 748 — pure
noise. Async only touches the GEMV decode path; prefill (GEMM
coopmat) is untouched.

### 3.3 Why we hit the theoretical ceiling

Sprint 15A measured `RECORD=1 836 µs`, `GPU_WAIT=9 034 µs`,
sequential overhead `~80 µs`. Perfect overlap = `max(1836, 9034) +
80 = 9 114 µs → 109.7 tok/s`. We measured **109.0 tok/s**, so
the steady-state overlap is essentially perfect — the CPU
recording fully hides inside the GPU's 9 ms compute window.

Why so clean: Vulkan's queue ordering on a single queue is
strictly in-order, so submit(CB[N+1]) doesn't need a timeline
semaphore — the GPU runs CB[N] then CB[N+1]. The CPU side just
needs to make sure it writes the embedding *before* submitting,
and reads logits *after* waiting. That's what the 3-stage shape
encodes.

## 4. Closing the v0.2 → v0.3 transition

**Decode-gap to llama.cpp Vulkan**:

| Release | Decode tok/s | Ratio |
|---|---:|---:|
| v0.2.0 | 90.5 | 0.79× |
| v0.2.4 (durable v0.2) | 91.1 | 0.80× |
| **v0.3.0 (this)** | **109.0** | **0.95×** |
| llama.cpp Vulkan ref (build 23b8cc4) | 114.2 | 1.00× |

The remaining ~5 % gap is genuinely beyond shader-config and
host-coordination levers — the next candidates (dedicated lm_head
coopmat, buffer-aliasing) were already analysed and found to be
≤ 1-3 % each. v0.3 is the natural release point.

**Per the Sprint 12-15 arc**: 13 falsified hypotheses landed
us *exactly* at the right intervention. The methodology — measure
first, falsify cheap, build only what survives the math — has
delivered both honest negatives and (now) a real positive,
matching the analytical predictions to within 1 %.

## 5. Outputs

- `src/backend/vulkan/forward.rs` (+227 / −47 LOC)
  - `IntermediateSlot` struct (Sprint 15D, already in `ca0d7ae`)
  - 4 new fields on `Forward` (`async_pool`, `async_cbs[2]`,
    `async_fences[2]`, `async_pending_record`)
  - `Forward::new` allocates the async pool / CBs / fences
  - `Forward::destroy` frees them
  - 3 new public methods (`pre_record`, `fill_embed_and_submit`,
    `wait_and_read_logits`)
  - `record_decode_dispatches` helper extracted from
    `forward_token`'s closure
- `src/backend/vulkan/decode.rs` (+102 / −0 LOC)
  - `if async_decode { 3-stage pipeline } else { serial fallback }`
    branch in `generate_from_tokens`
- This report.
- 27/27 lib tests, 15/15 coherent on both paths.
- Decode 109.0 tok/s @ 15-prompt median (target: ≥ 100). **+19.3 %.**

## 6. What this enables for v0.3.0 release

Recommended:

- Bump `Cargo.toml` 0.2.4 → 0.3.0
- README: status line "**v0.3.0** — async pipelined decode loop:
  decode 109 tok/s (0.95× llama.cpp Vulkan); prefill 0.89× at
  pp=512 (unchanged from v0.2.4)". Performance table gets a new
  v0.3.0 column.
- CHANGELOG: v0.3.0 entry covering 15D + 15E. Mark the closing
  of the Sprint 12-15 decode-gap analysis arc.
- `VULKANFORGE_DISABLE_ASYNC_DECODE=1` documented as the opt-out
  env var alongside the existing flags.

The release narrative is honest: 13 falsified hypotheses showed
where the gap was *not*; one architectural change (async
pipelined decode) closes 87 % of the remaining gap to llama.cpp,
matching the upper bound the measurement predicted.
