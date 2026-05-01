# v0.3.1 Sprint 16C — sampling polish

**Result:** Sampling pipeline was already shipped (`sample_next_token`
in `src/backend/vulkan/decode.rs:116` does repetition-penalty →
temperature → softmax → top-K → top-P → weighted draw, with all CLI
flags wired through). The actual gaps were UX:

1. `--seed` defaulted to `0`, so `--temperature 0.7` without `--seed`
   replayed the same pseudo-random sequence every run.
2. The chat banner didn't show the active sampling config — users had
   no on-screen confirmation of `temp` / `top-k` / `top-p`.
3. README still documented the v0.2.x `cargo run --example` workflow
   instead of the v0.3.1 `vulkanforge chat / bench / info` CLI.

## Pre-check (per memory)

Per `feedback_sprint_hypothesis_check.md`: verify before coding. The
brief asked whether sampling was wired or was decoration. Pre-check
output:

```
$ grep -n 'fn sample_next_token' src/backend/vulkan/decode.rs
116: pub fn sample_next_token(...)

$ grep -n 'temperature.*top_k.*top_p' src/main.rs
107..119: clap fields exist
160..190: each maps to a Sampling field with VF_* env fallbacks
233..238: let sampling = Sampling { temperature: args.temperature, ... }
```

Sampling end-to-end. No need for the brief's `SamplingConfig` struct,
new sample function, or `rand` crate dependency.

## What changed

### `src/main.rs`

1. **Time-derived seed when none specified.** The chat-args block now
   does:

   ```rust
   seed_was_explicit: seed.is_some() || std::env::var("VF_SEED").is_ok(),
   seed: seed.unwrap_or_else(|| {
       std::env::var("VF_SEED").ok()
           .and_then(|s| s.parse().ok())
           .unwrap_or_else(|| {
               std::time::SystemTime::now()
                   .duration_since(std::time::UNIX_EPOCH)
                   .map(|d| d.as_nanos() as u64)
                   .unwrap_or(0)
           })
   }),
   ```

   So `--temperature 0.7` produces a fresh sequence each run; `--seed N`
   / `VF_SEED=N` still pin determinism.

2. **Banner shows sampling.** New `Sampling` + `seed_explicit` params
   to `print_banner` — greedy mode prints
   `Sampling:     greedy (temperature=0)`; sampling mode prints
   `Sampling:     temp=0.70 top_k=40 top_p=0.90 rep_pen=1.00 seed=auto`
   (or `seed=42` when explicit).

3. **`ChatArgs::seed_was_explicit: bool`** tracks whether the seed
   came from `--seed` / `VF_SEED` or was clock-derived, so the banner
   can label it accurately.

### `README.md`

- Replaced the old `cargo run --example sample_decode` block with a
  `vulkanforge chat / bench / info` table.
- Added a CLI-flag reference table (default + effect for each of the 9
  chat flags).
- Three sampling examples: greedy, creative+random, creative+seeded.
- Replaced the bare "Sampling (per-run)" env-var paragraph with a
  proper table aligned to the CLI flag table.

## Verification

| Check                                           | Result |
|-------------------------------------------------|--------|
| `cargo build --release`                         | clean |
| `cargo test --release --lib`                    | **27 / 27** |
| `vulkanforge chat … --temperature 0.7 --seed 42` (×2) | **identical** outputs |
| `vulkanforge chat … --temperature 1.0 --top-p 0.9` (×2 auto-seed) | **divergent** outputs (≥ 80 tokens) |
| Greedy banner                                   | `Sampling: greedy (temperature=0)` |
| Auto-seed banner                                | `... seed=auto` |
| Explicit-seed banner                            | `... seed=42` |
| `run_15prompt_bench` (greedy via example)       | **15 / 15 coherent**, decode 108.9 tok/s |

The `bench` subcommand still hardcodes `Sampling { temperature: 0.0, … }`
(`src/main.rs:705,745`) — bench results stay reproducible regardless of
shell env state.

## Why two short-output sampling runs can look identical

When testing with `--max-tokens 16` and a high-confidence prompt
(`"say one short word"`), the first 10–15 tokens are extremely peaked:
even at `temperature=0.7`, the post-softmax distribution often puts
> 0.9 probability on a single token, and the weighted draw lands on it
every time. With a less canonical prompt and 80 tokens, the two
auto-seed runs visibly diverge — seen in this report's verification
column.

## What's NOT in this sprint

- No new sampling features. The brief proposed re-implementing the
  full pipeline; not needed.
- No change to default temperature. The brief wavered between
  `0.0` (deterministic, current) and `0.7` (creative). Kept at `0.0`
  because:
  - Existing release-note invariant: "outputs stay byte-deterministic".
  - The 15-prompt coherence regression assumes greedy.
  - Users get explicit control via `--temperature N` with one flag.
- No `rand` crate dependency. The internal xorshift64* in
  `decode.rs:299` is fine for sampling.

## Files

- `src/main.rs` — `+34 / -3`
- `README.md` — `+45 / -22`
- `results/v031_sprint16c_sampling.md` — this file

Total LOC delta: ~80, two files. The brief's proposed refactor would
have been ~150 LOC of duplicate sampling code in main.rs plus a `rand`
dependency, replacing existing tested code that already does the right
thing.
