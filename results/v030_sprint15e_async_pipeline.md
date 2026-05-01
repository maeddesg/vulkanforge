# Sprint 15E — Async Decode Pipeline: not implemented this session

**Premise.** Sprint 15D shipped the double-buffered intermediate
infrastructure (`ca0d7ae`). Sprint 15E was scoped to deliver the
3-stage async pipeline that 15D enables: `pre_record(CB[N+1])`
running in parallel with `GPU(CB[N])`, with the embedding-write
deferred until after sampling. Theoretical ceiling +20 % decode
(91 → 110 tok/s).

**Status.** **Not implemented this session.** The design from
`results/v030_sprint15de_async_decode.md` §3 is correct and
mechanically clear, but I burned the session's budget iterating on
analysis variants before committing to code, and a half-baked
async path on the decode hot loop is exactly the failure mode the
brief warns against. 15D's win stands; 15E remains a single
focused next-session sprint.

## 1. What's needed (unchanged from `d6f8596` §3)

```rust
impl Forward {
    /// Pre-record CB into slot[slot] for token at `position`.
    /// References slots[slot] buffer handles only — the embedding
    /// content can be written after this call but before submit.
    /// Must be called when slots[slot]'s prior CB has completed (
    /// fence has been waited on or fence is fresh).
    pub fn pre_record(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        model: &LoadedModel,
        slot: usize,
        position: u32,
    ) -> Result<(), Box<dyn std::error::Error>> { … }

    /// Stage 2/3: write embedding into the pre-recorded slot's
    /// scratch_a, write the position into rope_pos_buf, submit CB.
    pub fn fill_embed_and_submit(
        &mut self,
        dev: &VulkanDevice,
        slot: usize,
        embedding: &[f32],
        position: u32,
    ) -> Result<(), Box<dyn std::error::Error>> { … }

    /// Stage 4: block on slot's fence and read logits.
    pub fn wait_and_read_logits(
        &mut self,
        dev: &VulkanDevice,
        slot: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> { … }
}
```

Plus on `Forward`:

```rust
/// Sprint 15E — async-mode CBs and fences. Allocated alongside the
/// existing CommandContext so that serial mode (forward_token) can
/// keep using cmd_ctx.one_shot unchanged.
async_cmd_pool: vk::CommandPool,
async_cbs: [vk::CommandBuffer; 2],
async_fences: [vk::Fence; 2],
async_logits_bufs: [GpuBuffer; 2],   // doubled for safe disjoint readback
```

The `async_logits_bufs` doubling is conservative — Sprint 15A
analysis suggested it might not be needed (readback happens before
the next CB submission in the 3-stage shape) but doubling is cheap
(~1.2 MB) and removes a class of subtle race possibilities.

## 2. Decode-loop rewrite (`decode.rs:470-489`)

```rust
let async_decode = std::env::var("VULKANFORGE_DISABLE_ASYNC_DECODE")
    .map(|v| v != "1").unwrap_or(true);

if async_decode {
    // First token: warm-up the pipe (no prev to wait for).
    let mut next_token = first_token;
    forward.pre_record(dev, registry, model, 0, 0)?;
    forward.fill_embed_and_submit(dev, 0, &embd, 0)?;
    let mut current_slot = 1usize;

    loop {
        let prev_slot = 1 - current_slot;
        // Stage 1: pre-record while previous GPU work is in flight.
        forward.pre_record(dev, registry, model, current_slot, pos + 1)?;
        // Stage 2: wait for prev, sample.
        let logits = forward.wait_and_read_logits(dev, prev_slot)?;
        next_token = sample_next_token(&logits, …);
        if tokenizer.is_eos(next_token) { break; }
        on_token(next_token, &tokenizer.decode_token(next_token));
        generated.push(next_token);
        if generated.len() >= max || pos + 1 >= max_seq { break; }
        // Stage 3: fill + submit.
        let embd = embedding_row(gguf, cfg, next_token)?;
        forward.fill_embed_and_submit(dev, current_slot, &embd, pos + 1)?;
        pos += 1;
        current_slot = 1 - current_slot;
    }
    // Drain final.
    let last_slot = 1 - current_slot;
    last_logits = forward.wait_and_read_logits(dev, last_slot)?;
} else {
    // Serial path (existing code, unchanged).
}
```

## 3. Implementation budget (unchanged from `d6f8596`)

- 1 hour: split out the recording closure body of `forward_token`
  into a private helper `record_forward(slot, position)` that
  takes a `cmd: vk::CommandBuffer` and writes the standard
  per-token dispatch sequence into it.
- 1 hour: build `async_cmd_pool / async_cbs / async_fences /
  async_logits_bufs` allocation in `Forward::new`; add `Drop`
  cleanup. Wire `pre_record` / `fill_embed_and_submit` /
  `wait_and_read_logits` using direct ash calls (begin/end/submit
  /wait), not via `cmd_ctx.one_shot`.
- 1-2 hours: rewrite `decode.rs` loop. Special-case first token
  (no prev), special-case EOS (don't submit a pre-recorded CB
  that we'll discard), special-case max-token break.
- 0.5 hour: bit-identical-output validation
  (`VULKANFORGE_DISABLE_ASYNC_DECODE=1` → run prompt → record output;
  re-run async → diff). At `temperature=0` the outputs must be
  identical except for the run-to-run reduction-order noise that
  Sprint 14B's Path A introduced (already < 1e-6 tolerance).
- 0.5 hour: bench the gate. Add `eprintln!` of the wait_fence
  elapsed time after pre_record to confirm CPU/GPU overlap is
  happening (target: wait ≈ 7.2 ms instead of 9 ms).

**Total: 3-5 hours of focused work**, delivered as a single
self-contained session. Runs the standard test gate
(27/27 lib + 15/15 coherent + bit-identical vs serial + decode ≥
100 tok/s) at the end before commit.

## 4. Why I stopped here

This session's productive output is `ca0d7ae` (Sprint 15D — the
struct refactor that mechanically rewrote 96+ buffer references in
~30 minutes once I committed to `sed`). That's a real, validated,
shippable infrastructure win. Adding a half-implemented 15E on
top of it would put the **decode hot path** in an inconsistent
state — exactly the failure mode the brief warned against ("BEI
UNKLARHEITEN SOFORT STOP. Das ist KERN-Infrastruktur!"). I'd
rather hand off a clean 15D + a fully-specified 15E than a
partial 15E that needs reverting.

The honest pattern here matches the larger Sprint 15 arc: 15A
(template-CB hypothesis falsified at source-reading), 15B (lm_head
falsified at quant-type lookup), 15C (1-day prototype infeasible),
15D (mechanical refactor, succeeded fast once committed), 15E
(this — design correct, implementation deferred). Each step has
been progressively useful and progressively closer to a clean
implementation brief; 15E now has both the infrastructure and the
correct design in place.

## 5. State of the v0.3 transition

| Sprint | Status | Note |
|---|---|---|
| 12D-12M, 13A-E, 14A-C | ✅ shipped or honest-negative | v0.2.4 |
| 15A | ✅ analysis-only stop | template-CB hypothesis falsified |
| 15B | ✅ honest-negative at pre-check | lm_head Q6_K at 94 % HBM |
| 15C | ✅ feasibility-stop | 1-day prototype infeasible |
| 15D | ✅ **shipped (`ca0d7ae`)** | 17-buffer × 2-slot infra |
| 15E (this) | ⏸ design ready, implementation pending | 3-5h, single session |

**Decode-gap analysis ledger: 13 entries. 15E remains the only
candidate with realistic > 5 % decode lift potential**, and now
has both infrastructure (`ca0d7ae`) and design (this report's §1
+ §2) ready for execution.

## 6. Outputs

- This report (`results/v030_sprint15e_async_pipeline.md`).
- **No new code.** v0.2.4 binary plus the `ca0d7ae` 15D refactor
  is the current state. 27 / 27 lib tests, 15 / 15 coherent.

## 7. Recommendation for next session

**Path A** (implement): use this report's §1 + §2 as the
mechanical brief. 3-5 hours of focused work. If the bench-gate
holds, commit + tag v0.3.0-async. If not, ship as opt-in via
`VULKANFORGE_ENABLE_ASYNC_DECODE=1` and call it v0.2.5 instead.

**Path B** (pause v0.3): v0.2.4 stands as durable. Pivot to
multi-modal or longer-context features. Revisit 15E when the
decode plateau becomes user-visible enough to warrant the
architectural change.

**Path C** (release 15D as v0.2.5 standalone): the struct
refactor is a useful API/infrastructure cleanup even without the
async loop on top. Tag v0.2.5, ship a small CHANGELOG entry, then
defer 15E to v0.3 properly.

I have no strong preference among the three. **My strong opinion
is that the next session pick one and commit to it**, rather than
continue the half-attempted 15E pattern that this session
collapsed into.
