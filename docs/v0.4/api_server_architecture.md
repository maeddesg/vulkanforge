# VulkanForge v0.4 — OpenAI-kompatibler API-Server Architektur

**Version:** 1.0.0-final
**Datum:** 2026-05-11
**Status:** FREIGEGEBEN — bereit für Implementation gemäß Anhang B Sprint-Plan.

**Entschiedene Klärungsfragen (urspr. STOP-Gate nach §3):**

1. **frequency_penalty Mapping:** `1.0 + max(0.0, f) * 0.5` — negative Werte → 1.0 (kein encourage-Pfad). ✅
2. **Context-Overflow HTTP-Code:** 400 mit `context_length_exceeded` (OpenAI-Kompat > semantische HTTP-Korrektheit). ✅
3. **Multi-Turn KV:** Stateless — KV-Reset pro Request, kein Prefix-Cache in v0.4. ✅
4. **Default-Host:** `127.0.0.1` — `--host 0.0.0.0` explizit für Docker/Remote. ✅
5. **Port-Default:** `8080`. ✅
6. **Model-ID-Default:** lowercased Basename ohne Extension. ✅

---

## §1 Übersicht + Ziele

### 1.1 Scope (v0.4 In)

Ein neues CLI-Subkommando `vulkanforge serve` startet einen
HTTP-Server der OpenAI-kompatible Endpoints exponiert. Damit
funktionieren VF-Modelle drop-in als Backend für Standard-UIs:

- **Open WebUI** (`ollama`/`openai` Backend-Mode)
- **SillyTavern** (Custom OpenAI-API Endpoint)
- **Continue.dev** / **Cursor** (custom-base-URL)
- **text-generation-webui** (OpenAI extension)
- **AnythingLLM**, **Jan**, **LobeChat** und andere clients die
  einen "OpenAI base_url" Knopf haben

Pflicht-Endpoints in v0.4:

| Path | Methode | Funktion |
|---|---|---|
| `POST /v1/chat/completions` | POST | Streaming + Non-Streaming Chat |
| `GET /v1/models` | GET | Geladenes Modell enumerieren |
| `GET /health` | GET | Liveness-Probe |

### 1.2 Scope (v0.4 Out)

Folgende werden in v0.4 **NICHT** implementiert; sie sind v0.5+:

- `POST /v1/completions` (legacy non-chat — nur falls Use-Case
  auftaucht, sonst skip)
- `POST /v1/embeddings` (VF hat aktuell keinen Embedding-Output-Pfad)
- `POST /v1/audio/*` (kein ASR/TTS)
- `POST /v1/responses` (OpenAIs neue Responses-API — instabil,
  noch nicht weit unterstützt)
- **Multi-Model**: ein Server-Prozess hostet **genau ein** Modell.
  Mehr-Modelle = mehr Prozesse (Port unterschiedlich).
- **Tool/Function-Calling**: braucht template-getriebene Strukturen
  und parser-side JSON-Validierung; eigene Feature-Welle.
- **Vision/Multimodal-Inputs**: VF hat aktuell keinen Vision-Pfad.
- **Auth**: kein Bearer-Token Check. v0.4 ist Local-Loopback by
  default (`127.0.0.1`); öffentliche Bindings sind Opt-In via
  `--host 0.0.0.0` mit Owner-Verantwortung.
- **OpenAI structured-output / JSON-Schema constraints**: braucht
  Logits-Bias / Grammar-Sampling, nicht in VF.

### 1.3 Vergleich mit llama.cpp Server

| Aspekt | llama.cpp | VulkanForge v0.4 |
|---|---|---|
| Concurrent Slots | Ja (bis zu N) | Nein (sequentiell, 429) |
| `/v1/completions` | Ja | Nein |
| `/v1/embeddings` | Ja | Nein |
| Tool Calling | Ja (via Jinja) | Nein |
| `response_format` (JSON-Mode) | Ja (Grammar) | Nein |
| Built-in Web UI | Ja | Nein (→ Open WebUI) |
| Think-Filter | Nein | **Ja (VF-Alleinstellung)** |
| Quality Monitor | Nein | **Ja (VF-Alleinstellung)** |
| On-the-fly Quantize-on-Load | Nein | **Ja (VF-Alleinstellung)** |
| Model-Introspection (ggf. v0.5) | Nein | **Ja (VF-Alleinstellung)** |

**Strategie:** Minimaler, korrekter OpenAI-Server. Differenzierung
über VF-Alleinstellungen (Think-Filter, Quality-Monitor,
quantize-on-load), nicht über Feature-Parität mit llama.cpp.

### 1.4 Performance-Ziele

| Metrik | Ziel (Q4_K_M 8B, gfx1201) | Begründung |
|---|---|---|
| TTFT (Time-To-First-Token), pp=64 | < 200 ms | UI-Latenz-Schwelle für "Streaming startet" |
| Decode Streaming | ≥ 100 tok/s | Baseline aus VFs current decode |
| SSE-Chunk-Latency (inter-token) | < 12 ms | 1/100 tok/s ≈ 10 ms; Server-Overhead ≤ 2 ms |
| Memory-Overhead Server | < 50 MB RSS | axum + tokio runtime + state |
| Concurrent Requests | 1 (serialisiert via Queue) | VFs Forward ist single-GPU-stream |

### 1.5 Compat-Matrix (Soll-Erwartung)

Was MUSS funktionieren (Acceptance-Test):

```
Client                  → erwartet                          Status v0.4
────────────────────────────────────────────────────────────────────────
curl POST chat/completions stream=false  → JSON Body         ✅ Pflicht
curl POST chat/completions stream=true   → SSE Stream        ✅ Pflicht
Open WebUI (openai backend mode)         → Chat funktioniert ✅ Pflicht
SillyTavern (Chat Completions endpoint)  → Stream + System   ✅ Pflicht
Continue.dev (config.json custom model)  → Code-Completion   ✅ Pflicht
OpenAI Python SDK (openai.chat.create)   → Stream + Sync     ✅ Pflicht
LangChain ChatOpenAI(base_url=…)         → Calls succeed     ✅ Pflicht
```

### 1.6 Nicht-Ziele (klargestellt)

- **Bit-Identisch zu OpenAI**: VF gibt eigene Token-IDs, eigene
  Logprobs, eigene Latency. Kompatibel = "Wire-Format passt", nicht
  "Output passt zu GPT-4".
- **Hochlast-Production**: v0.4 ist ein **Single-User Local-Backend**.
  Kein Rate-Limiting, kein Auth, kein Multi-Tenancy.

---

## §2 Endpunkte

### 2.1 `POST /v1/chat/completions`

**Request Body (Content-Type: application/json):**

```json
{
  "model": "string (required)",
  "messages": [
    {"role": "system|user|assistant", "content": "string"}
  ],
  "stream": false,
  "max_tokens": 200,
  "temperature": 0.0,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": null,
  "seed": null,
  "user": null,

  // VF-Extensions (non-OpenAI):
  "top_k": 0,
  "repetition_penalty": 1.0,
  "min_p": 0.0
}
```

**Field-Support-Matrix:**

| Feld | Typ | Support | Mapping → VF |
|---|---|---|---|
| `model` | string | ✅ | Akzeptiert, aber **IGNORIERT** (Single-Model Server). Response enthält den tatsächlich geladenen Model-Namen. Kein 404 bei Mismatch — llama.cpp/Ollama machen es genauso. Begründung: viele Clients senden hardcoded `gpt-3.5-turbo` o.ä. |
| `messages` | array | ✅ | Pflicht; alternierende user/assistant; optional 1× system am Anfang |
| `messages[].role` | enum | ✅ | `system`/`user`/`assistant`; `developer` wird als `system` behandelt (OpenAI-Alias seit Dezember 2024); `tool` → 400 in v0.4 |
| `messages[].content` | string \| array | partial | String: 1:1. Array von content-parts: nur `type: "text"` Parts werden konkateniert; alle anderen (image_url, tool_use) → 400 in v0.4 |
| `stream` | bool | ✅ | true → SSE; false → JSON |
| `max_tokens` | int | ✅ | → `Sampling.max_tokens` (capped at `max_context - prompt_tokens`) |
| `temperature` | float | ✅ | → `Sampling.temperature` |
| `top_p` | float | ✅ | → `Sampling.top_p` |
| `frequency_penalty` | float | ✅ | → `Sampling.repetition_penalty` mit `freq → 1.0 + freq/2` Mapping (OpenAIs [-2, 2] → VFs [>1.0]); siehe §7 |
| `presence_penalty` | float | ⏭️ ignoriert | VF hat kein presence-getrenntes Modell; OpenAI-Wert wird gelesen, Warn-Log, nicht angewendet |
| `stop` | string \| array | ✅ | bis 4 stop-strings (OpenAI-Limit); text-level matching nach Decode |
| `stream_options` | obj | partial | Nur `include_usage: true` unterstützt. Wenn gesetzt: zusätzliches SSE-Chunk vor `[DONE]` mit `usage`-Block (siehe §4.3). Andere Sub-Felder ignoriert. |
| `seed` | int \| null | ✅ | → `Sampling.seed`; null → wall-clock |
| `user` | string | ⏭️ ignoriert | Telemetrie-Feld, kein Effect auf Generation |
| `n` | int | ⏭️ nur n=1 | n>1 → 400 |
| `logit_bias` | map | ⏭️ ignoriert | Optional v0.5 |
| `logprobs` | bool | ⏭️ ignoriert | v0.5 |
| `top_logprobs` | int | ⏭️ ignoriert | v0.5 |
| `response_format` | obj | ⏭️ ignoriert (text-only) | JSON-Mode: v0.5+ via grammar |
| `tools` / `tool_choice` | array | ⏭️ ignoriert | v0.5+ |
| `parallel_tool_calls` | bool | ⏭️ ignoriert | v0.5+ |
| `function_call` / `functions` | — | ⏭️ deprecated | v0.5+ via tools API |
| `top_k` | int | ✅ VF-ext | OpenAI hat das nicht; vLLM/llama.cpp haben es im `extra_body` |
| `repetition_penalty` | float | ✅ VF-ext | Direkt → `Sampling.repetition_penalty` (überschreibt `frequency_penalty` falls beide gesetzt) |
| `min_p` | float | ⏭️ v0.5 | Aktuell hat VFs Sampler kein min_p; default disabled |

**Response (Non-Streaming, `stream: false`):**

```json
{
  "id": "chatcmpl-<base32-uuid>",
  "object": "chat.completion",
  "created": 1715420000,
  "model": "qwen3-8b",
  "system_fingerprint": "vulkanforge-0.4.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "string"
      },
      "logprobs": null,
      "finish_reason": "stop|length|content_filter"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 17,
    "total_tokens": 59
  }
}
```

**Response (Streaming, `stream: true`):**

Content-Type: `text/event-stream; charset=utf-8`

Eventfolge:

```
data: {"id":"chatcmpl-XXX","object":"chat.completion.chunk","created":1715420000,"model":"qwen3-8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-XXX","object":"chat.completion.chunk","created":1715420000,"model":"qwen3-8b","choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}

data: {"id":"chatcmpl-XXX","object":"chat.completion.chunk","created":1715420000,"model":"qwen3-8b","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-XXX","object":"chat.completion.chunk","created":1715420000,"model":"qwen3-8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Regeln:
- Erstes Chunk emittiert `delta.role = "assistant"` (sonst leer)
- Folge-Chunks emittieren `delta.content` mit der **inkrementellen**
  Text-Differenz seit letztem Chunk (Token-decoded String, nach
  ThinkFilter wenn aktiv)
- Letztes Chunk emittiert `delta: {}` (leer) und das passende
  `finish_reason`
- **Wenn `stream_options.include_usage: true`:** ein zusätzliches
  Chunk **nach** dem Final-Chunk und **vor** `[DONE]`, mit leerem
  `choices: []` und einem `usage`-Block:
  ```
  data: {"id":"chatcmpl-XXX","object":"chat.completion.chunk","created":...,"model":"qwen3-8b","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17,"total_tokens":59}}
  ```
- Final separater Marker `data: [DONE]\n\n` (OpenAI-Spec)
- Jedes Event endet mit `\n\n`

**Finish-Reasons:**

| Wert | Trigger |
|---|---|
| `stop` | EOS-Token vom Modell ODER `stop`-String matched ODER `<\|im_end\|>` / `<turn\|>` boundary |
| `length` | `max_tokens` erreicht ODER `current_pos + max_tokens > max_context` |
| `content_filter` | nicht implementiert in v0.4; reserved für v0.5+ |

**Error-Responses:**

| HTTP | Code | Wann | Body-Format |
|---|---|---|---|
| 400 | `invalid_request_error` | Schema-Validierung failed, ungültige Rolle, n>1, content-array mit non-text part | OpenAI-Error-JSON |
| 400 | `invalid_request_error` (code `context_length_exceeded`) | prompt + max_tokens > max_context | OpenAI-kompatibel (Decision: OpenAI-Kompat > HTTP-Semantik) |
| 429 | `server_busy` | Concurrent request während Queue voll | OpenAI-Error-JSON |
| 500 | `internal_error` | GPU-Crash, Tokenizer-Fehler, IO | OpenAI-Error-JSON |
| 503 | `model_loading` | Server gestartet aber Modell-Load nicht fertig | OpenAI-Error-JSON |

OpenAI-Error-JSON-Format:

```json
{
  "error": {
    "message": "Human-readable description",
    "type": "invalid_request_error",
    "param": "messages[0].role",
    "code": "invalid_role"
  }
}
```

---

### 2.2 `GET /v1/models`

**Request:** keine Parameter

**Response 200:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-8b",
      "object": "model",
      "created": 1715420000,
      "owned_by": "vulkanforge",
      "permission": [],
      "root": "qwen3-8b",
      "parent": null
    }
  ]
}
```

Da v0.4 single-model ist, ist `data[]` immer ein Array mit genau
einem Eintrag. Die `id` wird beim Server-Start aus dem
Model-Pfad-Basename abgeleitet ODER explizit gesetzt via
`--served-model-name`.

### 2.3 `GET /health`

**Request:** keine Parameter

**Response 200 (Modell loaded + ready):**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "qwen3-8b",
  "version": "0.4.0",
  "kv_cache": {
    "max_seq_len": 2048,
    "current_pos": 0
  }
}
```

**Response 503 (Modell lädt noch):**

```json
{
  "status": "loading",
  "model_loaded": false,
  "version": "0.4.0"
}
```

`/health` ist **nicht authentifiziert** und immer reachable
unabhängig vom Modell-State (damit Health-Probes funktionieren).

---

## §3 Request/Response Types (Rust-Struct-Pseudo-Code)

Diese Definitionen leiten direkt die Implementation. Module-Layout
siehe §9 (kommt nach STOP). Alle Types in `src/server/types.rs`
(oder Sub-Module).

### 3.1 Chat Completion Request

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(default)]
    pub stream: bool,

    // OpenAI sampling fields (all optional)
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,  // accepted, ignored
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,            // accepted, ignored
    #[serde(default)]
    pub n: Option<u32>,                   // only 1 supported

    // Ignored-but-accepted (no 400 if present, just no-op + log)
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    // VF extensions (also accepted from clients that know us)
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,               // accepted, v0.5 honors it
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[serde(alias = "developer")]    // OpenAI alias since Dec 2024
    System,
    User,
    Assistant,
    // We accept "tool" syntactically for forward-compat but error in v0.4
    Tool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    // ImageUrl/etc. parsed for forward-compat but rejected in v0.4 handler
    ImageUrl { image_url: serde_json::Value },
}
```

Bemerkung zu `serde(deny_unknown_fields)`: bewusst weglassen am
Top-Level, weil OpenAI-Clients regelmäßig Felder durchschleusen
die wir nicht kennen (`stream_options`, `service_tier`, …). Strict
Reject auf Tier-1-Felder, lax auf Tier-2.

### 3.2 Chat Completion Response (Non-Streaming)

```rust
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,                    // "chatcmpl-<id>"
    pub object: &'static str,          // "chat.completion"
    pub created: u64,                  // unix epoch seconds
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,            // always "assistant"
    pub content: String,
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    // ToolCalls reserved for v0.5+
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

### 3.3 Chat Completion Chunk (Streaming)

```rust
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,                    // same for all chunks of one request
    pub object: &'static str,          // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,    // only set on first chunk: "assistant"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
```

Stream-Sequenz:

| Chunk # | `delta.role` | `delta.content` | `finish_reason` |
|---|---|---|---|
| 1 (Header) | `"assistant"` | `None` | `None` |
| 2..N-1 | `None` | `Some("…")` | `None` |
| N (Final) | `None` | `None` | `Some(FinishReason)` |
| (Marker) | `data: [DONE]\n\n` (kein JSON) | — | — |

### 3.4 Models List Response

```rust
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,           // "list"
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,           // "model"
    pub created: u64,
    pub owned_by: &'static str,         // "vulkanforge"
    pub permission: Vec<serde_json::Value>,  // empty
    pub root: String,                    // == id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,          // None
}
```

### 3.5 Health Response

```rust
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub model_loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache: Option<KvCacheInfo>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    Loading,
    Error,
}

#[derive(Debug, Serialize)]
pub struct KvCacheInfo {
    pub max_seq_len: u32,
    pub current_pos: u32,
}
```

### 3.6 Error Response

```rust
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorInner,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorInner {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}
```

`error_type` Werte (OpenAI-kompatibel):
- `invalid_request_error` → 400 (schema, semantik, context_length_exceeded)
- `server_error` → 500
- `engine_unavailable` → 503 (model loading)
- `rate_limit_exceeded` → 429

### 3.7 Beispiel-Mapping: Request → Internal Config

Pseudo-Code (gehört nach §6/§7, hier nur zur Type-Validation):

```rust
fn build_generate_config(req: &ChatCompletionRequest, model_max_ctx: u32, prompt_len: u32)
    -> Result<(GenerateConfig, Option<Vec<String>>), ApiError>
{
    // Cap max_tokens at remaining context
    let req_max = req.max_tokens.unwrap_or(200);
    let avail = model_max_ctx.saturating_sub(prompt_len);
    let max_tokens = req_max.min(avail).max(1);

    // Resolve sampling (VF-ext takes precedence over OpenAI mapping)
    let temperature = req.temperature.unwrap_or(0.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let top_k = req.top_k.unwrap_or(0);
    let rep_penalty = req.repetition_penalty
        .or_else(|| req.frequency_penalty.map(|f| 1.0 + f.max(0.0) * 0.5))
        .unwrap_or(1.0);
    let seed = req.seed.unwrap_or_else(seed_from_clock);

    Ok((GenerateConfig {
        max_tokens,
        print_stream: false,
        think_filter: detect_think_filter(model),
        sampling: Sampling { temperature, top_k, top_p, repetition_penalty: rep_penalty, seed },
    }, parse_stop(req.stop.as_ref())))
}
```

---

## §4 Streaming-Architektur

### 4.1 Kanal-Topologie

Drei Threads/Tasks bilden eine ein-Producer/ein-Consumer Pipeline:

```
GPU-Forward-Thread          tokio mpsc                axum SSE-Handler
─────────────────────       ───────────               ───────────────────
ChatSession.send_streaming  ─ event_tx.send(Token) ─►  event_rx.recv()
  on_visible(text) → push                              → Sse::Event::data(json)
                                                       → yield aus Stream
```

Begründung: Forward läuft auf einem dedizierten Blocking-Thread
(via `tokio::task::spawn_blocking`) weil VFs Forward synchrone
Vulkan-FFI macht und nicht in einen Future passt. SSE-Handler ist
ein Async-Task der vom Channel pollt und `Sse::Event` emittiert.

### 4.2 Channel-Type

```rust
use tokio::sync::mpsc;

enum StreamEvent {
    Header,                                    // emits first chunk with delta.role
    Delta(String),                             // text chunk (post-think-filter)
    Final { finish: FinishReason, usage: Usage },
    UsageOnly(Usage),                          // emitted when stream_options.include_usage=true (after Final, before [DONE])
    Error(ApiError),                           // mid-stream GPU/tokenizer error
}

// Bounded channel: 64 tokens of buffer is enough headroom for
// network jitter without unbounded growth if client is slow.
let (tx, rx) = mpsc::channel::<StreamEvent>(64);
```

**Backpressure:** Channel `capacity=64`. Wenn der Client langsam
liest, blockiert `tx.blocking_send()` im Forward-Thread nach 64
gepufferten Tokens. Das pausiert GPU-Decode bis der Client wieder
liest. **Bewusst gewählt:** OOM-Risiko durch un-bounded queue ist
schlimmer als Decode-Slowdown bei Slow-Client.

### 4.3 SSE-Format-Adapter

```rust
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio_stream::wrappers::ReceiverStream;
use futures_util::StreamExt;

pub fn build_sse_stream(
    rx: mpsc::Receiver<StreamEvent>,
    chunk_meta: ChunkMeta,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(move |event| {
        let chunk = match event {
            StreamEvent::Header => header_chunk(&chunk_meta),
            StreamEvent::Delta(t) => delta_chunk(&chunk_meta, &t),
            StreamEvent::Final { finish, .. } => final_chunk(&chunk_meta, finish),
            StreamEvent::UsageOnly(usage) => usage_chunk(&chunk_meta, usage),
            StreamEvent::Error(e) => return Ok(Event::default().event("error")
                                                    .data(serde_json::to_string(&e).unwrap())),
        };
        Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
    });

    // After the inner stream ends we append the [DONE] marker.
    let with_done = stream.chain(futures_util::stream::once(async {
        Ok(Event::default().data("[DONE]"))
    }));

    // Keep-alive every 15s emits an SSE comment ":\n\n" which most
    // clients silently ignore — prevents intermediate proxies from
    // closing idle connections during long prefill.
    Sse::new(with_done).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}
```

### 4.4 Client-Disconnect → GPU-Cancel

Bei Client-Abbruch (TCP-Close mid-stream) emittiert axum keinen
direkten Callback, aber der `mpsc::Sender::send()` Call im
Forward-Thread liefert `Err(SendError)` weil der Receiver
gedropped wurde. Das ist unser Cancel-Signal:

```rust
// In the spawn_blocking forward task:
let result = chat_session.send_streaming(..., |visible| {
    if tx.blocking_send(StreamEvent::Delta(visible.into())).is_err() {
        // Client disconnected — set the cancellation flag.
        cancel_flag.store(true, Relaxed);
    }
});
```

Plus eine Cancel-Check **innerhalb** der Decode-Loop. VFs
`GenerateConfig` braucht ein optionales Cancel-Hook (siehe
**§11 Open Decisions #1**). Ohne Hook läuft der aktuelle
Decode-Batch zu Ende (gemäß `max_tokens`), aber kein Re-Submit
für KV-Append nach Disconnect — Memory ist sicher.

### 4.5 ThinkFilter im Stream

VFs `ChatSession.send_streaming` ruft `on_visible(text)` bereits
**nach** ThinkFilter auf (siehe `chat.rs:162-176`). Wir nutzen
denselben Hook — der `<think>...</think>` Block fließt nie in
einen SSE-Delta. Defaults:
- Qwen3, DeepSeek-R1, Gemma-4-26B → ThinkFilter **on**
- Sonst off
- Override via Request-Feld `chat_template_kwargs.enable_thinking`
  (llama.cpp-kompatibel) ODER VF-Extension `think_filter: bool`

### 4.6 Non-Streaming-Pfad (`stream: false`)

Trivial: derselbe Forward-Thread läuft synchron, sammelt alle
Tokens in einem `String`, returnt ein einzelnes `ChatCompletionResponse`
JSON. Keine mpsc nötig. Implementation re-uses dieselbe
`build_generate_config` + `chat_session.send` (non-streaming Variante,
existiert bereits in `chat.rs:229`).

---

## §5 Server-Architektur

### 5.1 Crate-Auswahl

| Crate | Version | Begründung |
|---|---|---|
| `axum` | `0.8` | Standard-Rust-HTTP-Framework, tower-kompatibel; SSE-Helper builtin; mature |
| `tokio` | `1` (features = ["full"]) | Async-Runtime; nötig für mpsc + spawn_blocking |
| `tower-http` | `0.6` | `CorsLayer` + `TraceLayer` (logging); kein Auth nötig in v0.4 |
| `serde` + `serde_json` | bereits in VF | Request/Response-Serde |
| `futures-util` + `tokio-stream` | `0.3` / `0.1` | Stream-Adapter für SSE |
| `uuid` | `1` (mit feature `v4`) | Chat-Completion-ID-Generierung |

**Warum axum, nicht actix-web?** axum baut auf tower/hyper auf,
ist die de-facto Default-Wahl 2025/2026, und integriert nahtlos mit
tokio das wir eh brauchen. actix hat eigene Runtime, mehr deps,
gegen den State-of-Art-Sound seit axum 0.7+ obsolet. tower-http
liefert CORS/Tracing/Compression als drop-in Layers.

**Warum nicht hyper direkt?** Routing, Extractor-Pattern,
SSE-Helper wären 2-3× mehr Boilerplate für Null-Vorteil.

### 5.2 AppState

```rust
pub struct AppState {
    /// The loaded model + tokenizer + chat template + config.
    /// Wrapped in Mutex because Forward is !Send-friendly only for
    /// one inflight request.
    pub session: tokio::sync::Mutex<ServerSession>,

    /// Single-permit semaphore — enforces concurrency=1 with explicit
    /// 429 on overflow (rather than head-of-line blocking).
    pub request_permit: tokio::sync::Semaphore,

    /// Model-id exposed via /v1/models (lowercased basename, or
    /// --served-model-name override).
    pub model_id: String,

    /// Server startup time (for /health uptime, optional in v0.4).
    pub started_at: std::time::Instant,
}

pub struct ServerSession {
    pub forward: Forward,
    pub tokenizer: Tokenizer,
    pub chat_template: ChatTemplate,
    pub model_cfg: ModelConfig,
    pub gguf: Option<GgufFile>,             // None for SafeTensors loads
    pub model: LoadedModel,
}
```

Wrapped in `Arc<AppState>`; geklont in jeden Handler via
`State(state): State<Arc<AppState>>`.

### 5.3 Concurrency-Model

**Eins-zu-einer-Zeit, mit explizitem Reject statt Queue.**

```rust
// In the chat handler:
let permit = match state.request_permit.try_acquire() {
    Ok(p) => p,
    Err(_) => return Err(busy_error()),  // 429
};
let mut session = state.session.lock().await;
// ... process request, hold permit + lock for the duration ...
```

**Warum kein Queueing?** Streaming-LLM-Requests dauern Sekunden bis
Minuten. Eine queue mit 10 Wartenden bedeutet up to 10× max-Latenz
fürs letzte Request. 429 schiebt die Queue-Verantwortung an den
Client zurück (typische OpenAI-Clients retryen mit Exponential
Backoff). Spätere Version kann opt-in `--max-queue N` bekommen.

**KV-Reset zwischen Requests (decision §3):**

Stateless: jeder Request startet mit `session.forward.kv_cache.reset()`
und re-prefilled das gesamte `messages[]`-Array. Kein Prefix-Cache.

### 5.4 CLI-Integration

```rust
// src/main.rs Commands enum gets a new variant:
Serve {
    #[arg(short, long)]
    model: Option<PathBuf>,
    #[arg(long)]
    tokenizer_from: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Override served-model name; default = lowercased basename.
    #[arg(long)]
    served_model_name: Option<String>,
    #[arg(long)]
    max_context: Option<u32>,
    /// Enable CORS with `Access-Control-Allow-Origin: *`.
    /// Required for browser-based UIs (Open WebUI, SillyTavern) on
    /// different ports/origins. Default: OFF (same-origin only).
    #[arg(long)]
    cors: bool,
},
```

Entry-Point (Pseudo):

```rust
fn serve_cmd(args: ServeArgs) -> Result<()> {
    // 1. Load model (current logic from Chat command extracted into
    //    a shared helper `load_model_session`).
    let session = load_model_session(&args.model, &args.tokenizer_from, args.max_context)?;
    let model_id = resolve_model_id(&args.model, args.served_model_name);

    // 2. Build AppState.
    let state = Arc::new(AppState {
        session: Mutex::new(session),
        request_permit: Semaphore::new(1),
        model_id,
        started_at: Instant::now(),
    });

    // 3. Build axum Router (see §5.5).
    let app = build_router(state, args.cors);

    // 4. Start tokio runtime + serve.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let addr = format!("{}:{}", args.host, args.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("VulkanForge API server listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
    })?;
    Ok(())
}
```

### 5.5 Router

```rust
fn build_router(state: Arc<AppState>, cors_enabled: bool) -> Router {
    let cors_layer = if cors_enabled {
        CorsLayer::permissive()
    } else {
        CorsLayer::new().allow_origin(AllowOrigin::mirror_request())
    };

    Router::new()
        // Primary OpenAI-Standard paths
        .route("/v1/chat/completions", post(handlers::chat::completions))
        .route("/v1/models", get(handlers::models::list))
        // Alias paths without /v1/ prefix — manche Clients (z.B. ältere
        // SillyTavern-Versionen, custom-tools) lassen /v1/ weg.
        // llama.cpp/Ollama akzeptieren beide Varianten.
        .route("/chat/completions", post(handlers::chat::completions))
        .route("/models", get(handlers::models::list))
        // Operations / introspection
        .route("/health", get(handlers::health::get))
        .layer(cors_layer)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
```

### 5.6 Graceful Shutdown

```rust
async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok(); };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM").recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    eprintln!("Shutting down…");
}
```

axum's `with_graceful_shutdown` lässt aktive Requests zu Ende laufen
(bounded by SSE-stream-completion), schließt dann den Listener.
**Hardstop nach 30s** (timer fallback) falls ein Request hängt:

```rust
tokio::select! {
    res = axum::serve(...).with_graceful_shutdown(shutdown_signal()) => res?,
    _ = tokio::time::sleep(Duration::from_secs(30 + max_decode_time())) => {
        eprintln!("Forced shutdown after 30s grace period");
    }
}
```

### 5.7 Logging

`tower-http::TraceLayer` + `tracing-subscriber` (Reuse falls VF schon
tracing nutzt; sonst neu rein). Env-var-gated:

- `VF_SERVE_LOG=info` (default)
- `VF_SERVE_LOG=debug` für Request-Body-Dumps
- `VF_SERVE_LOG_FILE=/path/server.log` für File-Sink

Logs umfassen:
- Request-Methode/Path/Status/Dauer
- `model_id` + Token-Count (prompt + completion)
- TTFT (Zeit Request-Empfang bis erstes SSE-Chunk)
- 4xx/5xx mit error-type

---

## §6 Chat-Template-Integration

### 6.1 messages[] → Token-Sequenz

Da v0.4 stateless ist, kommt bei jedem Request das komplette
`messages[]` Array. Das wird in eine flache Token-Sequenz gerendert:

```
[BOS]
<template-header-für-system>(messages[0].content if role=system)<template-sep>
<template-header-für-user>(messages[1].content if role=user)<template-sep>
<template-header-für-assistant>(messages[2].content if role=assistant)<template-sep>
... (alternierend) ...
<template-header-für-assistant>      ← generation-prompt (no content)
```

VFs aktueller `ChatTemplate::render_first_turn` rendert nur
system+user; für full multi-turn fügen wir hinzu:

```rust
// src/backend/vulkan/chat_template.rs (neu):
pub fn render_full_history(
    self,
    tokenizer: &Tokenizer,
    messages: &[RenderMessage],   // role + content pairs
) -> Vec<u32> {
    match self {
        ChatTemplate::ChatML => render_chatml_full(tokenizer, messages),
        ChatTemplate::Llama3 => render_llama3_full(tokenizer, messages),
        ChatTemplate::Gemma4 => render_gemma4_full(tokenizer, messages, false),
        ChatTemplate::Gemma4WithThoughtChannel => render_gemma4_full(tokenizer, messages, true),
        ChatTemplate::DeepSeekR1 => render_deepseek_full(tokenizer, messages),
        ChatTemplate::Mistral => render_mistral_full(tokenizer, messages),
        ChatTemplate::Raw => render_raw_full(tokenizer, messages),
    }
}
```

`RenderMessage` ist eine flache Server-side Variante von `Message`
(content bereits zu String konkateniert, role als Enum).

**Coding-Standards-Notiz:** Das ist eine **additive** Erweiterung
von `chat_template.rs`, nicht im `forward/` Modul. Der `forward/`
Coding-Standard (§4 dieses Memory) gilt für Forward-Dispatcher,
nicht für Template-Rendering. Trotzdem: keine modellspezifischen
`if cfg.gemma4` in der Server-Schicht — nur via `ChatTemplate`
enum dispatchen.

### 6.2 System-Message-Handling

- 0 oder 1 system-Messages erlaubt, **muss messages[0]** sein wenn
  vorhanden (sonst 400)
- Mehrere system-Messages: nur die erste verwenden, Warn-Log
  (manche Clients machen das — defensive)
- Kein system-Header im Template: System-Content wird vor dem
  ersten User-Header prepended (Mistral-Pfad) oder als getrenntes
  Pseudo-Turn gerendert (Llama-3-Pfad)

### 6.3 Multi-Turn-Abfolge

Standard-Pattern:
```
[system?, user, assistant, user, assistant, ..., user]
```

Validation:
- Mindestens 1 user-Message vorhanden (sonst 400)
- Letzte Message muss `role: "user"` sein (sonst 400 — der Server
  muss wissen worauf er antworten soll)
- Alternierung user↔assistant nach dem optionalen System-Header
  ist **nicht erzwungen** — manche Clients schicken 2× user
  hintereinander (Edit-Workflows). Wir rendern sie als
  getrennte User-Turns; das Modell lernt damit umzugehen.

### 6.4 ThinkFilter & assistant-Messages aus messages[]

Wenn `messages[]` ein assistant-Turn enthält der vom letzten Call
zurückkam, kann es ein `<think>` Block enthalten (wenn der Client
unfiltered Text zurück sendet) oder nicht (wenn der Client schon
gefiltert hat). Beides handlen:

- Beim Encoden eines assistant-Turns: **als-ist** rendern (kein
  Server-side Strip). Reasoning der Client-Side.
- Beim Streamen der Server-Antwort: ThinkFilter on (§4.5).

### 6.5 Generation-Prompt

Nach allen messages[] **immer** das assistant-Header-Token
emittieren (z.B. `<|im_start|>assistant\n` für ChatML), damit das
Modell weiß dass es jetzt generieren soll. Das ist analog zu
HuggingFace's `apply_chat_template(add_generation_prompt=True)`.

Für Gemma-4-26B (siehe Memory `project_v0318_gemma4_26b_status`):
zusätzlich `<|channel>thought\n<channel|>` anhängen — das ist die
`ChatTemplate::Gemma4WithThoughtChannel` Variante, die der
`detect_hf`-Path automatisch wählt.

---

## §7 Sampling-Parameter-Mapping (vollständig)

### 7.1 Mapping-Tabelle

| OpenAI-Feld | Default | VF-Sampling-Feld | Mapping |
|---|---|---|---|
| `temperature` | OpenAI default 1.0, VF default 0.0 | `temperature` | 1:1, geklemmt auf `[0.0, 2.0]` |
| `top_p` | 1.0 | `top_p` | 1:1, geklemmt auf `(0.0, 1.0]` |
| `frequency_penalty` | 0.0 | `repetition_penalty` | `1.0 + max(0.0, f) * 0.5` (negative → 1.0, no encourage path) |
| `presence_penalty` | 0.0 | — | **ignoriert**, Warn-Log wenn ≠ 0 |
| `seed` | null | `seed` | null → `seed_from_clock()`; sonst 1:1 |
| `max_tokens` | null (model-max) | `max_tokens` | `min(req, max_context - prompt_tokens)`, geklemmt `≥ 1` |
| `stop` | null | (text-level, post-decode) | bis 4 strings; siehe §7.3 |
| `top_k` (VF-ext) | 0 | `top_k` | 1:1, `0` = disabled |
| `repetition_penalty` (VF-ext) | 1.0 | `repetition_penalty` | 1:1; **überschreibt** das frequency_penalty-Mapping wenn beide gesetzt |
| `min_p` (VF-ext) | 0.0 | — | accepted, ignored in v0.4 |

### 7.2 frequency_penalty-Mapping (geklärt)

```rust
fn map_freq_penalty(freq: Option<f32>, rep: Option<f32>) -> f32 {
    // VF-Extension wins if both are present.
    if let Some(r) = rep { return r.max(1.0); }
    let f = freq.unwrap_or(0.0);
    // Negative values are accepted by OpenAI (encourage repetition)
    // but VF has no encourage path; clamp to 0 → identity penalty.
    1.0 + f.max(0.0) * 0.5
}
```

Beispiele:
- `frequency_penalty: 0.0` → `repetition_penalty: 1.0` (kein Effekt)
- `frequency_penalty: 1.0` → `repetition_penalty: 1.5`
- `frequency_penalty: 2.0` → `repetition_penalty: 2.0`
- `frequency_penalty: -1.0` → `repetition_penalty: 1.0` (no encourage)

### 7.3 stop-Strings

OpenAI erlaubt bis 4 stop-Strings. VFs Decode-Loop hat aktuell
keine eingebauten stop-strings — die werden **post-decode** am
text-Level matched:

```rust
let mut accumulated = String::new();
// In the on_visible callback:
|visible_chunk: &str| {
    accumulated.push_str(visible_chunk);
    for stop in &stop_strings {
        if let Some(idx) = accumulated.rfind(stop.as_str()) {
            // Truncate output at the stop boundary and cancel.
            send_delta(&accumulated[..idx]);
            cancel_flag.store(true, Relaxed);
            finish_reason = FinishReason::Stop;
            return;
        }
    }
    // No stop hit — flush all accumulated text since last flush.
    send_delta(&accumulated[last_flush..]);
    last_flush = accumulated.len();
}
```

**Achtung:** Token-boundary kann mitten in einem Stop-String liegen.
Wir akkumulieren bevor wir Streamen → kleine Latenz-Erhöhung
(`max_stop_len` Bytes hold-back). Default-Hold-Back = 32 Bytes
(genug für `<|im_end|>`-style Sentinels).

### 7.4 EOS und Stop-Token

EOS-Tokens kommen aus dem Tokenizer (`Tokenizer::is_eos(id)`). Das
sind tokenizer-spezifisch: ChatML-`<|im_end|>`, Llama-3-`<|eot_id|>`,
Gemma-4-`<turn|>`. Diese gelten **immer** als Stop, unabhängig vom
`stop`-Feld im Request → `finish_reason: "stop"`.

### 7.5 max_tokens Cap

```rust
let max_tokens = req.max_tokens.unwrap_or(200)
    .min(state.session.lock().await.forward.kv_cache.config.max_seq_len
         .saturating_sub(prompt_tokens))
    .max(1);
```

Wenn `prompt_tokens >= max_seq_len`: **400 mit
`context_length_exceeded`** (decision §2), nicht erst beim Decode
auflaufen lassen.

---

## §8 Fehlerbehandlung

### 8.1 Error-Mapping-Tabelle

| Fehlerklasse | HTTP | `type` | `code` | Wann |
|---|---|---|---|---|
| Schema-Validation (serde) | 400 | `invalid_request_error` | `invalid_body` | Top-Level JSON-Parse oder Field-Type-Mismatch |
| Ungültige Rolle | 400 | `invalid_request_error` | `invalid_role` | `messages[i].role` nicht in {system,user,assistant} |
| Tool-Role | 400 | `invalid_request_error` | `unsupported_role` | `role: "tool"` (v0.5+) |
| Image-Content-Part | 400 | `invalid_request_error` | `unsupported_content_type` | `messages[].content[].type == "image_url"` |
| n > 1 | 400 | `invalid_request_error` | `unsupported_n` | Mehrere Choices nicht unterstützt |
| Kein User-Message | 400 | `invalid_request_error` | `no_user_message` | messages[] ohne user-Turn |
| Letzte Message ≠ user | 400 | `invalid_request_error` | `last_message_not_user` | |
| Model-Mismatch | — | — | — | **NICHT validiert** — `model`-Feld wird ignoriert (Decision §2). Response trägt den geladenen Model-Namen unabhängig vom Request. |
| Context-Overflow | 400 | `invalid_request_error` | `context_length_exceeded` | `prompt_tokens + max_tokens > max_seq_len` (decision §2) |
| Server-Busy | 429 | `rate_limit_exceeded` | `concurrent_limit` | request_permit.try_acquire failed |
| Tokenizer-Error | 500 | `server_error` | `tokenizer_error` | Encode/decode failure |
| GPU-Error | 500 | `server_error` | `gpu_error` | Vulkan-FFI failure |
| Model-Loading | 503 | `engine_unavailable` | `model_loading` | Im Startup-Window vor Modell-Ready |

### 8.2 Error-Wrapper

```rust
pub struct ApiErrorResponse {
    pub status: StatusCode,
    pub body: ApiError,
}

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

// Helpers:
pub fn invalid_request(message: impl Into<String>, code: &'static str,
                        param: Option<String>) -> ApiErrorResponse { ... }
pub fn context_length_exceeded(prompt: u32, max_ctx: u32) -> ApiErrorResponse { ... }
pub fn server_busy() -> ApiErrorResponse { ... }
// ...
```

### 8.3 Streaming-Mid-Request-Errors

Wenn ein Error **mid-stream** auftritt (z.B. GPU-OOM beim 50ten
Token), kann man kein neues HTTP-Status setzen — die Response-Header
sind schon raus. SSE-konformer Workaround:

```
data: {"error": {"message": "GPU OOM during decode", "type": "server_error", "code": "gpu_error"}}

data: [DONE]

```

Plus passendes `finish_reason` in einem Final-Chunk (best-effort).
Manche Clients lesen `error`-Events nicht; Decision: Log-Warn + emit
trotzdem, Client ist verantwortlich.

### 8.4 Validierung-Reihenfolge

```
1. JSON-Parse                          → 400 invalid_body
2. messages[]-Schema-Validation        → 400 invalid_role / invalid_content
3. messages[]-Semantik (user-last)     → 400 no_user_message / last_message_not_user
4. Render-To-Tokens                    → 500 tokenizer_error
5. Context-Length-Check                → 400 context_length_exceeded
6. Acquire request_permit              → 429 concurrent_limit
7. Acquire session lock                → (always succeeds after permit)
8. KV-Reset + Prefill + Decode         → streamed errors per §8.3

(Model-ID-Match wurde aus der Reihenfolge entfernt — Decision §2:
`model`-Feld wird ignoriert.)
```

---

## §9 Datei-Struktur

```
src/
  server/                              ← NEUES MODUL
    mod.rs              — pub use für serve_cmd + AppState
    state.rs            — AppState, ServerSession, model_id resolution
    routes.rs           — build_router() + middleware composition
    handlers/
      mod.rs            — pub mod chat, models, health
      chat.rs           — POST /v1/chat/completions (streaming + non-streaming)
      models.rs         — GET /v1/models
      health.rs         — GET /health
    types/
      mod.rs            — pub re-exports
      request.rs        — ChatCompletionRequest, Message, ContentPart, Role
      response.rs       — ChatCompletionResponse, Choice, Usage, Delta, ChunkChoice
      health.rs         — HealthResponse, KvCacheInfo
      error.rs          — ApiError, ApiErrorResponse, ApiErrorInner
    stream.rs           — build_sse_stream, StreamEvent, ChunkMeta
    template.rs         — render_messages_to_tokens (wraps ChatTemplate::render_full_history)
    sampling.rs         — map_request_to_sampling (impl from §3.7 / §7)
    cancel.rs           — CancelToken (Arc<AtomicBool>) used by Forward decode loop
  main.rs               ← +1 Subcommand (Commands::Serve)
  lib.rs                ← pub mod server (für integration tests)

src/backend/vulkan/
  chat_template.rs      ← +pub fn render_full_history + role-aware helpers
  decode.rs             ← +optional cancel_token: Option<Arc<AtomicBool>> in GenerateConfig
```

**Begründung Sub-Module:**

- `handlers/{chat,models,health}.rs` — ein File pro Endpoint, jeder
  unter 200 LOC. Folgt axum-Convention.
- `types/{request,response,health,error}.rs` — geteilt nach
  Verantwortlichkeit, damit niemand 800-Zeilen-`types.rs` öffnen muss.
- `stream.rs` und `cancel.rs` getrennt — Streaming-Adapter ist
  reine Mechanik; CancelToken ist ein Cross-Concern.
- `chat_template.rs` Erweiterung ist additiv und im bestehenden
  Modul (`render_first_turn` und `render_continuation` bleiben für
  CLI/REPL relevant).

**Coding-Standards-Compliance:**
- `forward/`-Standard (`docs/vf_forward_coding_standards.md`) wird
  **nicht berührt**: kein neues LayerStep, kein neuer GPU-Shader,
  keine Änderung an Executor/Dispatch-Pfaden.
- Einzige Ausnahme: optionales `cancel_token` Feld in
  `GenerateConfig` (decode.rs). Das ist ein NEUES Feld auf einem
  bestehenden Struct, keine LayerStep-Erweiterung. Der Decode-Loop
  liest es alle 8 Tokens und bricht bei `true` ab.

---

## §10 Test-Strategie

### 10.1 Unit-Tests (`src/server/types/*` + `src/server/sampling.rs`)

- `frequency_penalty` Mapping-Tabelle: Test pro Edge-Case (negative,
  zero, positive, both freq+rep set, rep wins)
- Stop-Sequence-Parsing: single-string, array, null, oversized (>4)
- Schema-Roundtrip: serialize → deserialize bit-equal für jeden
  Request/Response/Chunk Type
- `context_length_exceeded` Trigger-Logik
- Model-ID-Match Case-Sensitivity (lowercased vs original)

Ziel: ≥30 neue Tests; vollständige Coverage der `types/` und
`sampling.rs` Module. Bestehende 67+ VF Tests bleiben grün
(Forward-Pfad unverändert).

### 10.2 Integration-Tests (`tests/server_integration.rs`)

Cargo-Test mit `#[tokio::test]` der den Server auf `127.0.0.1:0`
(Auto-Port) startet und mit `reqwest` Requests schickt:

- `GET /health` vor Modell-Load → 503
- `GET /health` nach Load → 200
- `GET /v1/models` → korrekte ID
- `POST /v1/chat/completions` non-stream → 200 + valides JSON
- `POST /v1/chat/completions` stream=true → SSE-Frames + `[DONE]`
- 400 für invalides Body / unsupported role / context_length_exceeded
- Request mit `model: "gpt-3.5-turbo"` (anderer Name als geladen) → 200 OK (Decision §2: `model` wird ignoriert)
- 429 für Second-Concurrent-Request
- Request mit `messages[].role: "developer"` → 200 OK (Alias auf `system`)
- Request mit `stream_options.include_usage: true` → Usage-Chunk vor `[DONE]`
- Request gegen `/chat/completions` (ohne `/v1/`) → 200 OK (Alias-Route)

Kein GPU-Bedarf für Schema-Validation-Tests (Stub für Forward).
GPU-Tests gated mit `#[cfg(feature = "vulkan-tests")]`.

### 10.3 Curl-basierte Smoke-Tests (`scripts/server_smoke.sh`)

```bash
# Non-streaming
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}]}' | jq

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"stream":true}'

# Health
curl -s http://localhost:8080/health | jq

# Models
curl -s http://localhost:8080/v1/models | jq

# Error: model mismatch
curl -s -w "%{http_code}\n" http://localhost:8080/v1/chat/completions \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}'

# Error: oversize context
curl -s -w "%{http_code}\n" http://localhost:8080/v1/chat/completions \
  -d "{\"model\":\"qwen3-8b\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python -c 'print("x"*100000)')\"}]}"
```

### 10.4 Kompatibilitäts-Smoke (manuell, dokumentiert in Acceptance-Test)

1. **Open WebUI** — Container starten, in Admin-Settings OpenAI-Backend
   `http://host.docker.internal:8080/v1` (Server muss mit
   `--host 0.0.0.0` laufen), 5-Turn-Chat fahren, Stream-Anzeige OK.
2. **SillyTavern** — Custom Chat-Completion Endpoint, Stream-Mode an,
   System-Prompt + 3-Turn Conversation.
3. **OpenAI Python SDK**:
   ```python
   from openai import OpenAI
   c = OpenAI(base_url="http://localhost:8080/v1", api_key="not-used")
   r = c.chat.completions.create(model="qwen3-8b",
       messages=[{"role": "user", "content": "Hi"}], stream=True)
   for chunk in r:
       print(chunk.choices[0].delta.content or "", end="")
   ```
4. **LangChain `ChatOpenAI`** — base_url + dummy key, `.invoke()` und
   `.stream()` smoke.

### 10.5 Performance-Validation

Re-bench mit Server-Layer aktiv:

```bash
# TTFT measurement:
time (curl -N localhost:8080/v1/chat/completions -d '{...,"stream":true}' \
  | head -c 200)

# Decode tok/s (manual count tokens vs wall-clock):
curl -N localhost:8080/v1/chat/completions -d '{...,"max_tokens":200,"stream":true}' \
  > /tmp/sse.log
# Count "delta.content" frames in /tmp/sse.log, divide by wall-time.
```

Akzeptanz-Ziele aus §1.3:
- TTFT pp=64 < 200 ms
- Decode ≥ 100 tok/s (Q4_K_M 8B, gfx1201)
- SSE-Chunk-Overhead < 2 ms vs nackt VF chat

Falls Δ > 10 % vs nacktem `vulkanforge chat`: Profiler-Analyse
(Channel-Send-Overhead? JSON-Serialize-pro-Token?).

### 10.6 Regression-Gate (analog `vf_forward_coding_standards.md` §5.2)

Vor v0.4-Release:
```
cargo test --release --lib                        → 67+ bestehende Tests grün
cargo test --release --test server_integration    → neu (30+ Tests)
Qwen3-8B Q4_K_M decode (chat-CLI)                 → ≥ 100 tok/s (unchanged)
Qwen3-8B Q4_K_M decode (server)                   → ≥ 95 tok/s (≤5% overhead)
Open WebUI + SillyTavern + Python-SDK smoke       → alle 3 grün
```

---

## §11 Offene Entscheidungen

Punkte für Owner-Entscheidung **vor oder während** Implementation:

1. **Cancel-Hook im Decode-Loop:** Soll der GPU-Decode bei Client-Disconnect
   sofort abbrechen (mid-batch, ~1-token Verlust an Latenz-Wahrnehmung)
   oder erst nach dem nächsten Token-Submit-Punkt? Erstes braucht
   `Arc<AtomicBool>` Check **innerhalb** der Decode-Iteration; zweites
   ist trivial einzubauen aber wirkt bei langen Prefills schwach.

2. **CORS-Default:** v0.4 default ist `AllowOrigin::mirror_request()`
   (Same-Origin reflektiert, kein `*`). Open WebUI im selben Browser
   funktioniert; cross-origin Webapps brauchen `--cors-any` Flag.
   Wenn das viele User auf SECURITY-Tickets schickt → `--cors-any`
   als Default umstellen.

3. **logprobs-Support:** v0.5 Feature, aber für Eval-Workflows (lm-eval,
   AlpacaEval) wichtig. Wenn der Use-Case schon in v0.4 auftaucht
   → priorisieren.

4. **JSON-Mode / Structured-Output:** v0.5+ via Grammar-Sampling;
   braucht aber Sampler-Changes (grammar-constrained next-token).
   Größere Sache, eigener Sprint.

5. **Tool/Function-Calling:** v0.5+. Erfordert Template-Erweiterung
   (Qwen3/Llama3 haben function-call templates), Output-Parser für
   JSON-Tool-Calls, OpenAI tools/tool_choice request mapping. Sprint-Material.

6. **Multi-Model + dynamisches Modell-Swap:** v0.6+. Erfordert
   GPU-Memory-Management für Load-on-Demand; v0.4 hat das nicht.

7. **Auth (Bearer-Token):** v0.5 wenn Tunneling/Cloudflare-Use-Cases
   auftauchen. Simple Env-Var `VF_API_KEY=xxx`; Middleware-Layer.

8. **Prefix-Cache für Multi-Turn:** v0.5+ (explizit gesetzt §3 oben).
   Würde ChatSession-Lebenszyklus verlängern und per-Request
   `messages[]`-Hash-Vergleich nötig machen.

9. **Async-Loading bei Server-Start:** Aktuell synchron im
   serve_cmd-Entry. Bei großen Modellen (26B → 130-160s laden,
   siehe Memory) blockiert das den Listener. Alternative:
   spawn Load-Task, /health gibt 503 bis ready. Empfehlung:
   ja, async-load, ist 30 LOC zusätzlich.

10. **Token-Streaming-Granularität:** Aktuell 1-Token-pro-SSE-Frame.
    Manche UIs (Continue.dev) bevorzugen "Stream-Buffered" mit
    50-100ms-Batches. Default 1-token, optionales Flag
    `--stream-buffer-ms 50`?

11. **Server-Logging-Format:** plain text (default), JSON
    (`--log-format json` für Struktured-Logging in
    Loki/Elastic)? v0.4 plain reicht.

---

## §12 Referenz-Links

### 12.1 OpenAI-Spec
- Chat Completions API: <https://platform.openai.com/docs/api-reference/chat/create>
- Models API: <https://platform.openai.com/docs/api-reference/models/list>
- Streaming-Format: <https://platform.openai.com/docs/api-reference/chat-streaming>
- Error-Format: <https://platform.openai.com/docs/guides/error-codes/api-errors>

### 12.2 Referenz-Implementations
- llama.cpp server: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- vLLM OpenAI server: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- Ollama OpenAI compat: <https://github.com/ollama/ollama/blob/main/docs/openai.md>

### 12.3 Rust-Stack
- axum: <https://docs.rs/axum/latest/axum/>
- axum SSE: <https://docs.rs/axum/latest/axum/response/sse/index.html>
- tokio: <https://docs.rs/tokio/latest/tokio/>
- tower-http (CORS, Trace): <https://docs.rs/tower-http/latest/tower_http/>
- serde + serde_json: <https://docs.rs/serde/latest/serde/>
- reqwest (für integration tests): <https://docs.rs/reqwest/latest/reqwest/>

### 12.4 Web-Standards
- SSE Spec (W3C): <https://html.spec.whatwg.org/multipage/server-sent-events.html>
- MDN EventSource: <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>
- HTTP Status Codes (RFC 9110): <https://www.rfc-editor.org/rfc/rfc9110.html>

### 12.5 VulkanForge Internals
- Chat-Session: `src/backend/vulkan/chat.rs` (multi-turn KV-state)
- Chat-Template: `src/backend/vulkan/chat_template.rs` (rendering)
- Generate-Config + Sampling: `src/backend/vulkan/decode.rs` (lines 30-110)
- Tokenizer: `src/backend/vulkan/tokenizer.rs`
- Forward-Coding-Standards: `docs/vf_forward_coding_standards.md`
- VF CLI: `src/main.rs` (Commands enum)

---

## Anhang A — Vollständige Request/Response Beispiele

### A.1 Non-Streaming Chat

**Request:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is a mutex?"}
    ],
    "temperature": 0.3,
    "max_tokens": 100
  }'
```

**Response (200 OK):**
```json
{
  "id": "chatcmpl-7e2a8c4f9b1d4e3a8c2f9b7d1e4a3c8f",
  "object": "chat.completion",
  "created": 1715420000,
  "model": "qwen3-8b-q4_k_m",
  "system_fingerprint": "vulkanforge-0.4.0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "A mutex (mutual exclusion lock) is a synchronization primitive that prevents multiple threads from simultaneously accessing a shared resource. Only one thread can hold the lock at a time; others block until release."
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 43,
    "total_tokens": 71
  }
}
```

### A.2 Streaming Chat

**Request:**
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    "max_tokens": 20
  }'
```

**Response (200 OK, Content-Type: text/event-stream):**
```
data: {"id":"chatcmpl-a1","object":"chat.completion.chunk","created":1715420010,"model":"qwen3-8b-q4_k_m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1","object":"chat.completion.chunk","created":1715420010,"model":"qwen3-8b-q4_k_m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1","object":"chat.completion.chunk","created":1715420010,"model":"qwen3-8b-q4_k_m","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1","object":"chat.completion.chunk","created":1715420010,"model":"qwen3-8b-q4_k_m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

### A.3 Streaming mit Usage-Chunk

**Request:**
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 20
  }'
```

**Response (Diff vs. A.2):** ein zusätzliches Chunk vor `[DONE]`:
```
... (Final-Chunk wie oben) ...

data: {"id":"chatcmpl-a1","object":"chat.completion.chunk","created":1715420010,"model":"qwen3-8b-q4_k_m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

data: [DONE]

```

### A.4 Health

**Request:**
```bash
curl http://localhost:8080/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "qwen3-8b-q4_k_m",
  "version": "0.4.0",
  "kv_cache": {"max_seq_len": 2048, "current_pos": 0}
}
```

### A.5 Models

**Request:**
```bash
curl http://localhost:8080/v1/models
```

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [{
    "id": "qwen3-8b-q4_k_m",
    "object": "model",
    "created": 1715420000,
    "owned_by": "vulkanforge",
    "permission": [],
    "root": "qwen3-8b-q4_k_m"
  }]
}
```

### A.6 Hardcoded-Model-Name Tolerance (Decision §2)

**Request mit OpenAI-Style hardcoded `gpt-3.5-turbo`:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Response (200 OK):** Request wird normal verarbeitet; das `model`-Feld
in der Response trägt den tatsächlich geladenen Namen
(`qwen3-8b-q4_k_m`), nicht `gpt-3.5-turbo`. Kein 404.

### A.7 `developer` Role als `system`-Alias

**Request:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"x","messages":[
    {"role":"developer","content":"You are concise."},
    {"role":"user","content":"Hi"}
  ]}'
```

**Response (200 OK):** Server behandelt `developer` als `system`.

### A.8 Error — Context-Length-Exceeded

**Request (Prompt >> max_context):**
```bash
curl -w "%{http_code}\n" http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python -c 'print("x"*100000)')\"}]}"
```

**Response (400):**
```json
{
  "error": {
    "message": "Prompt tokens (24832) + max_tokens (200) exceeds context length (2048)",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
```

### A.9 Error — Concurrent-Request

**Request 2** während Request 1 noch streamt:
```bash
curl -w "%{http_code}\n" http://localhost:8080/v1/chat/completions ...
```

**Response (429):**
```json
{
  "error": {
    "message": "Server busy: another request is in progress",
    "type": "rate_limit_exceeded",
    "code": "concurrent_limit"
  }
}
```

---

## Anhang B — Implementierungs-Reihenfolge

### Sprint 1 (≈4h): Foundation

```
□ src/server/mod.rs          — Modul-Skelett, pub use
□ src/server/types/          — alle Structs aus §3:
    □ request.rs             — ChatCompletionRequest, Message, ContentPart, Role
    □ response.rs            — ChatCompletionResponse, Choice, Usage, Delta, ChunkChoice
    □ health.rs              — HealthResponse, KvCacheInfo
    □ error.rs               — ApiError, ApiErrorInner
□ src/server/error.rs        — ApiErrorResponse + IntoResponse + Helpers
□ src/server/sampling.rs     — map_request_to_sampling (impl §7)
□ Cargo.toml deps:           axum 0.8, tokio "full", tower-http, futures-util,
                             tokio-stream, uuid v4
□ Unit-Tests:                ≥30 Tests für types/ + sampling (frequency_penalty
                             Mapping, Stop-Sequence-Parsing, developer alias,
                             context_length_exceeded Trigger)
```

**Gate:** `cargo test --release --lib` grün; bestehende 67+ VF-Tests
weiter grün.

### Sprint 2 (≈4h): Non-Streaming-Endpoint

```
□ src/server/state.rs        — AppState, ServerSession, resolve_model_id()
□ src/server/routes.rs       — build_router() inkl. /v1-Aliase + CORS-Layer
□ src/server/handlers/health.rs   — GET /health
□ src/server/handlers/models.rs   — GET /v1/models + /models
□ src/server/template.rs     — render_messages_to_tokens (multi-turn)
□ src/backend/vulkan/chat_template.rs  — pub fn render_full_history hinzu
□ src/server/handlers/chat.rs     — POST /v1/chat/completions
                                    (NUR non-streaming Pfad)
□ src/main.rs                — Commands::Serve variant + serve_cmd()
□ Integration-Test:          curl non-streaming → 200 + valides JSON
```

**Gate:** End-to-end roundtrip funktioniert; `messages[]` mit
system+user → assistant-Response.

### Sprint 3 (≈4h): Streaming + Cancel

```
□ src/server/stream.rs       — StreamEvent, ChunkMeta, build_sse_stream(),
                               header_chunk/delta_chunk/final_chunk/usage_chunk
□ src/server/cancel.rs       — CancelToken (Arc<AtomicBool>)
□ src/backend/vulkan/decode.rs  — optionales cancel_token: Option<Arc<AtomicBool>>
                                  in GenerateConfig + Check alle 8 Tokens
□ src/server/handlers/chat.rs   — Streaming-Pfad via spawn_blocking + mpsc
□ ThinkFilter im Stream:     Wiederverwendung von ChatSession.send_streaming
                              (on_visible kommt schon post-filter)
□ stream_options.include_usage Unterstützung
□ Integration-Test:          curl streaming, OpenAI Python SDK smoke
```

**Gate:** SSE-Frames + `[DONE]` korrekt; Usage-Chunk wenn requested;
TTFT < 200 ms (pp=64).

### Sprint 4 (≈2h): Polish + Compat

```
□ CORS:                      --cors Flag, mirror_request default
□ Graceful Shutdown:         SIGINT/SIGTERM + 30s Hardstop-Timer
□ Async Model-Load:          spawn(load_task), /health 503 bis ready
                              (siehe §11 Open-Decision #9)
□ Open WebUI Smoke:          Container-Setup, 5-Turn-Chat
□ SillyTavern Smoke:         Custom-Endpoint, 3-Turn + Stream
□ OpenAI Python SDK Smoke:   chat.completions.create stream + sync
□ LangChain ChatOpenAI Smoke
□ Performance-Vergleich:     vulkanforge chat (CLI) vs vulkanforge serve
                              + curl → Δ ≤ 5 % (Regression-Gate §10.6)
□ README.md Update:          neuer Abschnitt "API Server"
□ docs/v0.4/usage.md (neu):  Beispiele + Compat-Liste
```

**Gate:** Alle 4 Kompat-Smokes grün; Server-Overhead ≤ 5 % vs nackt CLI.

### Geschätzter Gesamtaufwand

| Sprint | Aufwand | Output |
|---|---|---|
| 1 | 4 h | Foundation + Tests |
| 2 | 4 h | Non-Streaming-Endpoint Live |
| 3 | 4 h | Streaming + Cancel + Usage-Chunk |
| 4 | 2 h | Polish + Compat-Smokes |
| **Total** | **~14 h** | **v0.4 API-Server Release-Ready** |

≈ 2 Tage fokussierte Arbeit. Sprints sind sequentiell — Sprint 2
braucht Sprint 1's Types, etc. Innerhalb eines Sprints können
Sub-Tasks parallelisiert werden.
