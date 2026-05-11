//! Mapping from OpenAI Chat Completion request fields to VF's
//! internal sampling representation.
//!
//! Spec references: §7 of `docs/v0.4/api_server_architecture.md`.
//! Decision §1 (Merge-Sprint): `frequency_penalty` maps to
//! `repetition_penalty` via `1.0 + max(0, fp) * 0.5`; negative
//! values clamp to 1.0 (no encourage-repetition path).
//!
//! Sprint 1 lands the pure mapping; the wire-up to
//! [`crate::backend::vulkan::decode::Sampling`] happens in Sprint 2
//! when the handler runs against a real `ServerSession`. Keeping the
//! mapping in its own module means the math is unit-testable without
//! pulling in the GPU stack.

use super::types::request::ChatCompletionRequest;

/// Resolved sampling parameters. Mirrors VF's existing
/// `crate::backend::vulkan::decode::Sampling` shape but lives in its
/// own struct so the server layer doesn't need a GPU-backed import.
/// The conversion to VF's `Sampling` happens in Sprint 2's handler.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// VF extension. `None` means "default" (= 0, disabled).
    pub top_k: Option<u32>,
    /// Multiplicative penalty applied to previously emitted tokens.
    /// `1.0` is the identity (no penalty); `>1.0` discourages repeats.
    pub repetition_penalty: f32,
    /// `None` defers to the handler's "remaining context"
    /// computation. The handler must cap this against
    /// `max_seq_len - prompt_tokens` (§7.5).
    pub max_tokens: Option<u32>,
    /// Up to 4 stop strings (OpenAI-spec limit).
    pub stop_sequences: Vec<String>,
    /// PRNG seed. `None` will be resolved to wall-clock in the
    /// handler (§7.1).
    pub seed: Option<u64>,
}

impl SamplingParams {
    /// Pure mapping from an OpenAI-shaped request. No GPU-dependent
    /// state required. The handler later clamps `max_tokens` against
    /// the loaded model's `max_seq_len` and resolves `seed = None`
    /// to a clock-derived value.
    pub fn from_request(req: &ChatCompletionRequest) -> Self {
        let temperature = req.temperature.unwrap_or(1.0).clamp(0.0, 2.0);

        // OpenAI documents top_p in (0, 1]; we accept the full
        // closed interval [0, 1] and let the downstream sampler
        // treat 0 as "filter everything except argmax" if it ever
        // matters. Clamp `> 1` defensively.
        let top_p = req.top_p.unwrap_or(1.0).clamp(0.0, 1.0);

        // VF extension wins when both repetition_penalty and
        // frequency_penalty are present (§7.1).
        let repetition_penalty = if let Some(r) = req.repetition_penalty {
            r.max(1.0)
        } else {
            map_frequency_penalty(req.frequency_penalty)
        };

        // §7.1: `top_k = 0` means disabled. We model that as `None`
        // here so the handler can pick a sensible default per model.
        let top_k = req.top_k.and_then(|k| if k == 0 { None } else { Some(k) });

        let stop_sequences = req
            .stop
            .clone()
            .map(|s| {
                let mut v = s.into_vec();
                // OpenAI-spec hard limit. Excess silently dropped
                // for client-tolerance; the handler logs a debug
                // line so the operator can spot abuse.
                v.truncate(4);
                v
            })
            .unwrap_or_default();

        // §7.1: presence_penalty is accepted-but-ignored.
        // We deliberately don't log here (parsing is on the request
        // path; the handler does any logging).

        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_tokens: req.max_tokens,
            stop_sequences,
            seed: req.seed,
        }
    }

    /// Whether this is a greedy configuration. The handler uses this
    /// to bypass the full multinomial sampler at `temperature == 0`.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// OpenAI `frequency_penalty` → VF `repetition_penalty`.
/// Negative values clamp to 0 (no encourage-repetition path).
/// Merge-Sprint decision #1.
fn map_frequency_penalty(freq: Option<f32>) -> f32 {
    let f = freq.unwrap_or(0.0);
    1.0 + f.max(0.0) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_from_json(j: &str) -> ChatCompletionRequest {
        serde_json::from_str(j).unwrap()
    }

    fn minimal_req() -> ChatCompletionRequest {
        req_from_json(r#"{"model":"x","messages":[{"role":"user","content":"x"}]}"#)
    }

    // --------------------------------------------------------------
    // Defaults (§7.1)
    // --------------------------------------------------------------

    #[test]
    fn missing_fields_use_openai_defaults() {
        let s = SamplingParams::from_request(&minimal_req());
        assert_eq!(s.temperature, 1.0); // OpenAI default
        assert_eq!(s.top_p, 1.0);
        assert_eq!(s.repetition_penalty, 1.0);
        assert!(s.top_k.is_none());
        assert!(s.max_tokens.is_none());
        assert!(s.stop_sequences.is_empty());
        assert!(s.seed.is_none());
        assert!(!s.is_greedy());
    }

    #[test]
    fn explicit_zero_temperature_is_greedy() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"temperature":0.0}"#,
        ));
        assert_eq!(s.temperature, 0.0);
        assert!(s.is_greedy());
    }

    #[test]
    fn temperature_clamps_to_two_above_range() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"temperature":5.0}"#,
        ));
        assert_eq!(s.temperature, 2.0);
    }

    #[test]
    fn temperature_clamps_to_zero_below_range() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"temperature":-1.0}"#,
        ));
        assert_eq!(s.temperature, 0.0);
    }

    #[test]
    fn top_p_clamps_to_unit_interval() {
        let high = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"top_p":2.5}"#,
        ));
        let low = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"top_p":-0.5}"#,
        ));
        assert_eq!(high.top_p, 1.0);
        assert_eq!(low.top_p, 0.0);
    }

    // --------------------------------------------------------------
    // frequency_penalty mapping (Decision #1, §7.2)
    // --------------------------------------------------------------

    #[test]
    fn frequency_penalty_zero_yields_unit_repetition_penalty() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"frequency_penalty":0.0}"#,
        ));
        assert_eq!(s.repetition_penalty, 1.0);
    }

    #[test]
    fn frequency_penalty_one_yields_one_and_a_half() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"frequency_penalty":1.0}"#,
        ));
        assert_eq!(s.repetition_penalty, 1.5);
    }

    #[test]
    fn frequency_penalty_two_yields_two() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"frequency_penalty":2.0}"#,
        ));
        assert_eq!(s.repetition_penalty, 2.0);
    }

    #[test]
    fn negative_frequency_penalty_clamps_to_unit_no_encourage_path() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"frequency_penalty":-1.0}"#,
        ));
        assert_eq!(s.repetition_penalty, 1.0);
    }

    #[test]
    fn vf_repetition_penalty_overrides_openai_frequency_penalty() {
        // VF extension wins per §7.1.
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],
                "frequency_penalty":2.0,"repetition_penalty":1.2}"#,
        ));
        assert_eq!(s.repetition_penalty, 1.2);
    }

    #[test]
    fn vf_repetition_penalty_below_one_is_floored_to_one() {
        // No encourage path even via the VF extension.
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"repetition_penalty":0.5}"#,
        ));
        assert_eq!(s.repetition_penalty, 1.0);
    }

    // --------------------------------------------------------------
    // top_k handling (VF-ext, §7.1)
    // --------------------------------------------------------------

    #[test]
    fn top_k_zero_means_disabled() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"top_k":0}"#,
        ));
        assert!(s.top_k.is_none());
    }

    #[test]
    fn top_k_positive_passes_through() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"top_k":40}"#,
        ));
        assert_eq!(s.top_k, Some(40));
    }

    // --------------------------------------------------------------
    // stop sequences (§7.3)
    // --------------------------------------------------------------

    #[test]
    fn stop_as_single_string_flattens_to_one_entry() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"stop":"END"}"#,
        ));
        assert_eq!(s.stop_sequences, vec!["END"]);
    }

    #[test]
    fn stop_as_array_passes_through() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"stop":["a","b","c"]}"#,
        ));
        assert_eq!(s.stop_sequences, vec!["a", "b", "c"]);
    }

    #[test]
    fn stop_null_yields_empty_vec() {
        let s = SamplingParams::from_request(&minimal_req());
        assert!(s.stop_sequences.is_empty());
    }

    #[test]
    fn stop_array_truncates_to_four() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],
                "stop":["a","b","c","d","e","f"]}"#,
        ));
        assert_eq!(s.stop_sequences, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn stop_sequence_into_vec_round_trip() {
        // Coverage for the helper independent of full-request path.
        use super::super::types::request::StopSequence;
        let m = StopSequence::Multiple(vec!["x".into(), "y".into()]).into_vec();
        assert_eq!(m, vec!["x", "y"]);
    }

    // --------------------------------------------------------------
    // Pass-throughs
    // --------------------------------------------------------------

    #[test]
    fn seed_passes_through_unchanged() {
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"seed":12345}"#,
        ));
        assert_eq!(s.seed, Some(12345));
    }

    #[test]
    fn max_tokens_passes_through_unchanged() {
        // Cap-against-context happens in the handler, not here.
        let s = SamplingParams::from_request(&req_from_json(
            r#"{"model":"x","messages":[{"role":"user","content":"x"}],"max_tokens":256}"#,
        ));
        assert_eq!(s.max_tokens, Some(256));
    }
}
