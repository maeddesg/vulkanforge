# VulkanForge v0.4 — OpenAI-kompatible API Server: Architektur-Dokument

**Version:** 1.0.0-draft  
**Datum:** 2026-05-11  
**Autor:** mg (Architektur), Claude (Dokument)  
**Status:** ENTWURF — Freigabe durch Owner ausstehend  
**Basis:** VulkanForge v0.3.18, OpenAI Chat Completions API, llama.cpp Server

---

## Inhaltsverzeichnis

- §1 Scope + Ziele
- §2 Endpunkte
- §3 Rust Types
- §4 Streaming-Architektur
- §5 Server-Architektur
- §6 Chat-Template-Integration
- §7 Sampling-Parameter-Mapping
- §8 Fehlerbehandlung
- §9 Datei-Struktur
- §10 Test-Strategie
- §11 Offene Entscheidungen
- §12 Referenz-Links

---

## §1 Scope + Ziele

### Was v0.4 liefert

1. **OpenAI-kompatibler HTTP-Server** mit `POST /v1/chat/completions` (Streaming + Non-Streaming), `GET /v1/models`, `GET /health`.
2. **Ein Modell pro Server-Instanz**, geladen beim Start.
3. **Sequentielle Request-Verarbeitung** — ein Request gleichzeitig, Queue für wartende Requests.
4. **CLI-Integration** über `vulkanforge serve` Subcommand.
5. **SSE-Streaming** im OpenAI-Format (`data: {...}\n\n`).
6. **Kompatibilität** mit: Open WebUI, SillyTavern, text-generation-webui, Continue.dev, Cursor, OpenAI Python SDK (`openai.OpenAI(base_url=...)`).

### Was v0.4 NICHT liefert

| Feature | Grund | Ziel-Version |
|---|---|---|
| Multi-Model | GPU-Context single-threaded | v0.6+ |
| Concurrent Requests | GPU-Context single-threaded | v0.5 (Queue-Parallelism mit Prefill/Decode Interleave) |
| `POST /v1/completions` (Legacy) | Kein UI nutzt das | Nicht geplant |
| `POST /v1/embeddings` | Braucht separaten Embedding-Modus | v0.5+ |
| Tool/Function Calling | OpenAI-spezifisches Feature, kein lokales Modell unterstützt das nativ | v0.5+ (DEFERRED) |
| Auth (API-Key, Bearer) | localhost-only Annahme | v0.5 |
| CORS | Entscheidung offen (§11) | v0.4.1 |
| TLS/HTTPS | Localhost, ggf. Reverse-Proxy | Nicht geplant |
| `response_format` (JSON Schema) | Komplexes Constrained Decoding | v0.5+ |
| `logprobs` | Braucht Top-K Logit-Extraktion | v0.5 |
| Image/Audio Input | VF ist text-only | Nicht geplant |
| `n > 1` (Multi-Choice) | Braucht Batch-Decode | v0.5+ |

### Performance-Ziele

| Metrik | Ziel | Begründung |
|---|---|---|
| TTFT (Time to First Token) | < 100ms (cached KV), < 500ms (cold prefill 512 tok) | UIs zeigen "typing..." an, User erwartet < 1s |
| Decode Throughput | Identisch zum CLI (105-121 tok/s Q4_K, 52 tok/s Gemma-4) | Server-Overhead darf < 1% sein |
| Server-Start (Model Load) | Identisch zum CLI | Kein zusätzlicher Overhead |
| HTTP-Overhead pro Token | < 50µs | SSE-Serialisierung + Send |
| Memory Overhead | < 20 MB über CLI hinaus | axum + Buffers |

### Vergleich mit llama.cpp Server

| Aspekt | llama.cpp | VulkanForge v0.4 |
|---|---|---|
| Concurrent Slots | Ja (bis zu N) | Nein (sequentiell, Queue) |
| `/v1/completions` | Ja | Nein |
| `/v1/embeddings` | Ja | Nein |
| `/v1/responses` | Ja (neu) | Nein |
| Tool Calling | Ja (via Jinja) | Nein |
| `response_format` | Ja (Grammar) | Nein |
| Grammar/GBNF | Ja | Nein |
| Built-in Web UI | Ja | Nein (verweist auf Open WebUI) |
| Think-Filter | Nein | Ja (VF Alleinstellung) |
| Quality Monitor | Nein | Ja (VF Alleinstellung) |
| Model Introspection | Nein | Ja (VF Alleinstellung) |
| On-the-fly Quantization | Nein | Ja (VF Alleinstellung) |

**Strategie:** Minimaler, korrekter OpenAI-kompatibler Server. Clients die `/v1/chat/completions` + `/v1/models` nutzen funktionieren sofort. Alles andere wird mit `501 Not Implemented` oder Feld-Ignoring behandelt.

---

## §2 Endpunkte

### 2.1 `POST /v1/chat/completions`

**Primärer Endpunkt.** Akzeptiert OpenAI-kompatible Chat Completion Requests.

#### Request

```
POST /v1/chat/completions
Content-Type: application/json
```

**Request Body (vollständig):**

```json
{
  "model": "string",
  "messages": [
    {
      "role": "system" | "user" | "assistant",
      "content": "string"
    }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": null,
  "stream": false,
  "stop": null,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "seed": null,
  "user": "string"
}
```

**Feld-Behandlung:**

| Feld | Typ | Erforderlich | VF Verhalten |
|---|---|---|---|
| `model` | `string` | Ja | Akzeptiert, aber IGNORIERT (Single-Model Server). Wird im Response mit dem tatsächlich geladenen Modell-Namen überschrieben. llama.cpp macht es genauso. |
| `messages` | `array` | Ja | Vollständig unterstützt. Siehe §6. |
| `messages[].role` | `string` | Ja | `"system"`, `"user"`, `"assistant"`. `"developer"` wird als `"system"` behandelt (OpenAI Alias). `"tool"` wird mit 400 abgelehnt. |
| `messages[].content` | `string` | Ja | Nur Plaintext. Multimodal-Content-Arrays (`[{type:"text",...}]`) werden mit 400 abgelehnt. |
| `temperature` | `f32` | Nein | Default: 0.7. Range: [0.0, 2.0]. Direkt an VF-Sampler. |
| `top_p` | `f32` | Nein | Default: 1.0. Range: (0.0, 1.0]. Direkt an VF-Sampler. |
| `max_tokens` | `u32 \| null` | Nein | Default: `null` → Modell-Maximum (Context-Länge minus Prompt). `0` = Fehler. |
| `stream` | `bool` | Nein | Default: `false`. `true` → SSE Response (§4). |
| `stop` | `string \| [string] \| null` | Nein | Bis zu 4 Stop-Sequenzen. Tokenisiert und in Sampler eingespeist. |
| `presence_penalty` | `f32` | Nein | Default: 0.0. Range: [-2.0, 2.0]. Gemappt auf VF `repeat_penalty`. |
| `frequency_penalty` | `f32` | Nein | Default: 0.0. Range: [-2.0, 2.0]. Gemappt auf VF `frequency_penalty`. |
| `seed` | `i64 \| null` | Nein | Akzeptiert, an VF-Sampler weitergeleitet. Determinismus nicht garantiert (wie OpenAI). |
| `user` | `string` | Nein | Akzeptiert, geloggt, nicht verarbeitet. |

**Ignorierte OpenAI-Felder (akzeptiert, aber ohne Wirkung):**

| Feld | Warum ignoriert |
|---|---|
| `n` | Single-Choice only (v0.4). Wert > 1 → 400 Error. |
| `logprobs` | Nicht implementiert. Feld wird ignoriert. |
| `top_logprobs` | Nicht implementiert. Feld wird ignoriert. |
| `response_format` | Kein Constrained Decoding. Feld wird ignoriert. |
| `tools` | Kein Tool Calling. Feld wird ignoriert. |
| `tool_choice` | Kein Tool Calling. Feld wird ignoriert. |
| `stream_options` | Akzeptiert. `include_usage: true` → Usage-Chunk am Ende (implementiert). |
| `service_tier` | OpenAI-spezifisch. Ignoriert. |
| `store` | OpenAI-spezifisch. Ignoriert. |
| `metadata` | OpenAI-spezifisch. Ignoriert. |
| `logit_bias` | Nicht implementiert. Ignoriert. |

**Abgelehnte Felder (lösen 400 Error aus):**

| Feld/Wert | Grund |
|---|---|
| `n > 1` | Nicht unterstützt. |
| `messages[].role = "tool"` | Kein Tool Calling. |
| `messages[].content` als Array | Kein Multimodal. |

#### Response (Non-Streaming)

```
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1715500000,
  "model": "Qwen3-8B-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The response text here."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

#### Response (Streaming)

Siehe §4 für das vollständige SSE-Format.

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1715500000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1715500000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1715500000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":" response"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1715500000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 2.2 `GET /v1/models`

**Liefert die Liste geladener Modelle (immer genau 1).**

#### Request

```
GET /v1/models
```

Kein Body.

#### Response

```
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-8B-Q4_K_M",
      "object": "model",
      "created": 1715500000,
      "owned_by": "vulkanforge"
    }
  ]
}
```

**Modell-ID-Ableitung:** Dateiname ohne Extension und Pfad. `~/models/Qwen3-8B-Q4_K_M.gguf` → `"Qwen3-8B-Q4_K_M"`. Bei SafeTensors: Verzeichnisname.

**Vergleich llama.cpp:** llama.cpp gibt ebenfalls genau ein Modell zurück. Format identisch. llama.cpp nutzt einen optionalen `--alias` Parameter — VF leitet den Namen aus dem Dateinamen ab, erlaubt aber Override via `--model-name`.

### 2.3 `GET /health`

**Health-Check für Load-Balancer und UIs.**

#### Request

```
GET /health
```

#### Response (Modell geladen, bereit)

```
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "model": "Qwen3-8B-Q4_K_M",
  "vulkanforge_version": "0.4.0",
  "gpu": "AMD Radeon RX 9070 XT",
  "vram_used_mb": 5800,
  "vram_total_mb": 16384
}
```

#### Response (Modell noch beim Laden)

```
HTTP/1.1 503 Service Unavailable
Content-Type: application/json
```

```json
{
  "status": "loading",
  "model": "Qwen3-8B-Q4_K_M",
  "progress": "loading weights (42%)"
}
```

**Vergleich llama.cpp:** llama.cpp hat `GET /health` mit `{"status": "ok" | "loading_model" | "error"}` und optionalem `slots_idle`/`slots_processing`. VF ist einfacher (kein Slot-Konzept) und reicher (GPU-Info, VRAM).

### 2.4 Nicht-implementierte Endpunkte

Alle anderen Pfade:

```
HTTP/1.1 404 Not Found
Content-Type: application/json
```

```json
{
  "error": {
    "message": "Unknown endpoint: POST /v1/completions. VulkanForge v0.4 supports: POST /v1/chat/completions, GET /v1/models, GET /health",
    "type": "not_found",
    "param": null,
    "code": "unknown_endpoint"
  }
}
```

---

## §3 Rust Types

Alle Structs nutzen `serde` mit `#[serde(rename_all = "snake_case")]`. Optionale Felder nutzen `#[serde(skip_serializing_if = "Option::is_none")]` um das Response-JSON minimal zu halten.

### 3.1 Request Types

```rust
use serde::{Deserialize, Serialize};

/// POST /v1/chat/completions Request Body
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Modell-ID. Akzeptiert aber ignoriert (Single-Model Server).
    /// Clients senden typischerweise den Modellnamen, den /v1/models liefert.
    pub model: String,

    /// Conversation Messages. Mindestens 1 Message erforderlich.
    pub messages: Vec<ChatMessage>,

    /// Sampling-Temperatur [0.0, 2.0]. Default: 0.7.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Nucleus-Sampling (0.0, 1.0]. Default: 1.0 (deaktiviert).
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Maximale Tokens in der Completion. null = Modell-Maximum.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// SSE-Streaming aktivieren.
    #[serde(default)]
    pub stream: bool,

    /// Streaming-Optionen.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    /// Stop-Sequenzen (max 4). String oder Array von Strings.
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,

    /// Presence-Penalty [-2.0, 2.0]. Default: 0.0.
    #[serde(default)]
    pub presence_penalty: f32,

    /// Frequency-Penalty [-2.0, 2.0]. Default: 0.0.
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Seed für reproduzierbare Ergebnisse. Determinismus nicht garantiert.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Anzahl Choices. MUSS 1 oder absent sein. > 1 → 400 Error.
    #[serde(default = "default_n")]
    pub n: u32,

    /// User-ID für Logging.
    #[serde(default)]
    pub user: Option<String>,

    // --- Ignorierte Felder (akzeptiert, ohne Wirkung) ---
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
    pub service_tier: Option<String>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 1.0 }
fn default_n() -> u32 { 1 }

/// Einzelne Message in der Conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// Rolle: "system", "user", "assistant", "developer" (→ system alias).
    pub role: String,

    /// Plaintext-Content. Multimodal-Arrays werden abgelehnt.
    pub content: String,
}

/// Streaming-Optionen (OpenAI-Feld).
#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    /// true → letzter Chunk enthält Usage-Objekt.
    #[serde(default)]
    pub include_usage: bool,
}
```

**`deserialize_stop`**: Custom Deserializer der sowohl `"stop_word"` (String) als auch `["stop1", "stop2"]` (Array) akzeptiert und als `Vec<String>` normalisiert. Maximal 4 Einträge, sonst 400 Error.

### 3.2 Response Types

```rust
/// Non-Streaming Response (object = "chat.completion").
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique ID. Format: "chatcmpl-{nanoid}".
    pub id: String,

    /// Immer "chat.completion".
    pub object: &'static str,

    /// Unix-Timestamp (Sekunden) der Erstellung.
    pub created: u64,

    /// Geladenes Modell (aus Server-State, NICHT aus Request).
    pub model: String,

    /// Immer genau 1 Choice (v0.4).
    pub choices: Vec<ChatCompletionChoice>,

    /// Token-Usage Statistik.
    pub usage: Usage,

    /// System-Fingerprint. Konstant pro Server-Instanz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Ein Choice in der Response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    /// Immer 0 (v0.4, n=1).
    pub index: u32,

    /// Die generierte Message.
    pub message: ChatMessage,

    /// Warum die Generation gestoppt hat.
    pub finish_reason: FinishReason,
}

/// Finish-Reason als Enum.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Normales Ende (EOS-Token oder Stop-Sequenz).
    Stop,
    /// max_tokens erreicht.
    Length,
    /// Content-Filter (aktuell nicht implementiert, reserviert).
    ContentFilter,
}

/// Token-Usage.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Tokens im Prompt (nach Template-Anwendung).
    pub prompt_tokens: u32,
    /// Generierte Tokens.
    pub completion_tokens: u32,
    /// Summe.
    pub total_tokens: u32,
}
```

### 3.3 Streaming Chunk Types

```rust
/// Streaming-Chunk (object = "chat.completion.chunk").
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// Gleiche ID wie alle Chunks dieser Completion.
    pub id: String,

    /// Immer "chat.completion.chunk".
    pub object: &'static str,

    /// Unix-Timestamp.
    pub created: u64,

    /// Modell-Name.
    pub model: String,

    /// Immer genau 1 Choice.
    pub choices: Vec<ChunkChoice>,

    /// Nur im letzten Chunk, wenn stream_options.include_usage=true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Choice in einem Streaming-Chunk.
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<FinishReason>,
}

/// Delta-Content in einem Streaming-Chunk.
/// Erster Chunk: role + leerer content.
/// Mittlere Chunks: nur content.
/// Letzter Chunk: leeres Delta + finish_reason.
#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
```

### 3.4 Models-Endpunkt Types

```rust
/// GET /v1/models Response.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Immer "list".
    pub object: &'static str,
    /// Genau 1 Modell (v0.4).
    pub data: Vec<ModelObject>,
}

/// Einzelnes Modell in der Liste.
#[derive(Debug, Serialize)]
pub struct ModelObject {
    /// Modell-ID (abgeleitet aus Dateiname).
    pub id: String,
    /// Immer "model".
    pub object: &'static str,
    /// Unix-Timestamp des Server-Starts.
    pub created: u64,
    /// Immer "vulkanforge".
    pub owned_by: String,
}
```

### 3.5 Health-Endpunkt Types

```rust
/// GET /health Response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// "ok" | "loading" | "error".
    pub status: String,
    /// Modell-Name.
    pub model: String,
    /// VulkanForge Version.
    pub vulkanforge_version: String,
    /// GPU-Name (aus Vulkan Device Properties).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// VRAM belegt (MB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_used_mb: Option<u64>,
    /// VRAM total (MB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total_mb: Option<u64>,
    /// Lade-Fortschritt (nur wenn status="loading").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<String>,
}
```

### 3.6 Error Types

```rust
/// OpenAI-kompatible Error-Response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    /// Menschenlesbarer Fehlertext.
    pub message: String,
    /// Fehler-Kategorie (OpenAI-kompatibel).
    #[serde(rename = "type")]
    pub error_type: String,
    /// Welcher Parameter den Fehler verursacht hat (oder null).
    pub param: Option<String>,
    /// Maschinen-lesbarer Error-Code.
    pub code: Option<String>,
}
```

---

## §4 Streaming-Architektur

### 4.1 SSE-Format

Das SSE-Protokoll sendet Events als Textzeilen mit `data:` Prefix:

```
data: {JSON}\n\n
```

Jedes Event ist ein vollständiges JSON-Objekt. Die Sequenz endet mit:

```
data: [DONE]\n\n
```

**Header:**

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

`X-Accel-Buffering: no` ist kritisch für nginx Reverse-Proxies. Ohne diesen Header puffert nginx die gesamte Response und liefert sie am Ende auf einmal aus.

### 4.2 Chunk-Sequenz

```
Chunk 0 (Role):
  delta: { role: "assistant", content: "" }
  finish_reason: null

Chunk 1..N (Tokens):
  delta: { content: "token_text" }
  finish_reason: null

Chunk N+1 (Ende):
  delta: {}
  finish_reason: "stop" | "length"

[Optional] Chunk N+2 (Usage, wenn stream_options.include_usage=true):
  choices: []
  usage: { prompt_tokens: X, completion_tokens: Y, total_tokens: Z }

data: [DONE]
```

**Warum Chunk 0 mit leerem Content:** OpenAI-Konvention. Clients wie Open WebUI erwarten den Role-Chunk als Signal dass die Response beginnt. Ohne diesen Chunk zeigen manche UIs keinen "typing..." Indikator.

### 4.3 Token-Loop → Stream Mapping

```
┌──────────────────────────────────────────┐
│  HTTP Handler (axum)                      │
│                                           │
│  1. Parse Request                         │
│  2. Validate                              │
│  3. Acquire InferenceLock (Mutex)         │
│  4. Apply Chat-Template (→ token_ids)     │
│  5. Start Prefill                         │
│  6. Create tokio mpsc::channel(32)        │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │ Inference Task (tokio::spawn)       │  │
│  │                                     │  │
│  │  loop {                             │  │
│  │    token = decode_one_token();      │  │
│  │    text = detokenize(token);        │  │
│  │    if tx.send(text).is_err() {      │  │
│  │      // Client disconnected         │  │
│  │      break;                         │  │
│  │    }                                │  │
│  │    if token == EOS || stop_match {  │  │
│  │      tx.send(FINISH_SIGNAL);        │  │
│  │      break;                         │  │
│  │    }                                │  │
│  │  }                                  │  │
│  │  release InferenceLock              │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Return Sse::new(ReceiverStream::new(rx)) │
│    .keep_alive(KeepAlive::new()           │
│      .interval(Duration::from_secs(15))   │
│      .text(""))                           │
└──────────────────────────────────────────┘
```

**Channel-Kapazität 32:** Tokenisierung + JSON-Serialisierung ist schneller als TCP-Send. 32 Tokens Buffer verhindert dass der Decode-Loop auf den Netzwerk-Sender wartet. Bei 121 tok/s ≈ 260ms Puffer.

### 4.4 Connection-Abbruch

Wenn der Client die SSE-Verbindung trennt (Tab schließen, Ctrl-C, Netzwerkfehler):

1. `tx.send()` gibt `Err(SendError)` zurück (Receiver dropped).
2. Der Decode-Loop bricht sofort ab.
3. `InferenceLock` wird released.
4. KV-Cache wird invalidiert (kein Resume möglich).

**Vergleich llama.cpp:** llama.cpp nutzt Slots die bei Abbruch freigegeben werden. VF hat kein Slot-Konzept, aber das Ergebnis ist identisch: Request wird abgebrochen, GPU wird frei.

### 4.5 Backpressure

Wenn der TCP-Buffer voll ist und der Client nicht schnell genug liest:

1. `tx.send()` blockiert (bounded channel, cap=32).
2. Der Decode-Loop pausiert automatisch.
3. Tokens gehen NICHT verloren.
4. GPU-Compute pausiert (kein Wasted Compute).

Wenn der Channel 15 Sekunden nicht entwässert wird → Timeout → Connection Close → Cleanup wie bei Abbruch.

### 4.6 Think-Filter bei Streaming

Qwen3 und andere Modelle emittieren `<think>...</think>` Tags. VFs Think-Filter ist eine VF-Alleinstellung:

1. Der Decode-Loop detektiert `<think>` Opening-Tag.
2. Tokens innerhalb des Think-Blocks werden NICHT über SSE gesendet.
3. `</think>` schließt den Block, danach werden Tokens wieder gesendet.
4. `stream_options` könnte in Zukunft `include_thinking: true` unterstützen (v0.5).

**Implementierung:** Der bestehende `ThinkFilter` aus VFs CLI-Streaming wird wiederverwendet. Er arbeitet als stateful Token-Filter zwischen Decode-Loop und Channel-Send.

---

## §5 Server-Architektur

### 5.1 Framework-Wahl: axum

**Entscheidung: `axum` (aktuelle stabile Version).**

| Alternative | Grund für Ablehnung |
|---|---|
| `actix-web` | Schwerer, eigene Runtime (optional tokio). axum ist First-Party tokio. |
| `warp` | Filter-basierte API ist schwer zu debuggen. Weniger aktiv. |
| `hyper` direkt | Zu low-level. SSE-Support müsste manuell implementiert werden. |
| `rocket` | Proc-Macro-heavy, langsame Compile-Zeiten. |

**Begründung axum:**

1. First-Party tokio-Ökosystem (tokio-rs Maintainer).
2. `axum::response::sse::Sse` + `Event` — nativer SSE-Support.
3. `tower` Middleware-Kompatibilität (CORS, Logging, Compression).
4. Minimaler Overhead — kein Proc-Macro, keine Runtime-Magie.
5. VF nutzt bereits tokio für async I/O.
6. Starke Community + aktive Entwicklung.

**Crate-Abhängigkeiten (Cargo.toml Ergänzungen):**

```toml
[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tokio-stream = "0.1"
tower-http = { version = "0.6", features = ["cors", "trace"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
nanoid = "0.4"
```

### 5.2 Concurrency-Model

```
┌──────────────────────────────────────────────────────┐
│                    axum Router                        │
│                                                       │
│  GET  /health              → health_handler           │
│  GET  /v1/models           → models_handler           │
│  POST /v1/chat/completions → chat_completions_handler │
│  fallback                  → not_found_handler        │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              AppState (Arc<AppState>)                  │
│                                                       │
│  engine: Arc<InferenceEngine>     // Modell + GPU     │
│  inference_lock: Arc<Mutex<()>>   // 1 Request/Zeit   │
│  model_name: String               // z.B. "Qwen3-8B"  │
│  model_id: String                 // "Qwen3-8B-Q4_K_M"|
│  created_at: u64                  // Server-Start      │
│  config: ServerConfig             // CLI-Parameter      │
└──────────────────────────────────────────────────────┘
```

**Warum `Mutex<()>` statt Semaphore:**

VFs GPU-Context ist single-threaded. Es kann EXAKT ein Inference-Request gleichzeitig laufen. Ein `Mutex<()>` ist die einfachste korrekte Lösung. `tokio::sync::Mutex` (nicht `std::sync::Mutex`!) damit der Lock über `.await`-Punkte gehalten werden kann.

**Request-Queue:** Wartende Requests warten auf den `Mutex`. tokio's Mutex ist fair (FIFO). Kein expliziter Queue-Mechanismus nötig. Bei 121 tok/s Decode und 512 max_tokens dauert ein Request ~4.2s. Ein zweiter Request wartet maximal so lange.

**Timeout:** Wenn ein Request nach 120 Sekunden den Lock nicht bekommt → 503 Service Unavailable mit `"Server busy, try again later"`.

### 5.3 CLI-Integration

Neuer Subcommand `serve`:

```bash
vulkanforge serve \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  --temperature 0.7 \
  --max-tokens 4096 \
  --system "You are a helpful assistant." \
  --model-name "qwen3-8b" \
  --context-size 8192 \
  --fp8-kv \
  --quantize-on-load
```

| Parameter | Default | Beschreibung |
|---|---|---|
| `--model` | Erforderlich | Pfad zu GGUF oder SafeTensors-Verzeichnis. |
| `--port` | `8080` | Listen-Port. |
| `--host` | `127.0.0.1` | Listen-Adresse. `0.0.0.0` für Netzwerk-Zugriff. |
| `--temperature` | `0.7` | Default-Temperatur (per-Request überschreibbar). |
| `--max-tokens` | Model-Context | Server-weites Maximum für `max_tokens`. |
| `--system` | `""` (kein System-Prompt) | Default System-Prompt, falls Request keinen sendet. |
| `--model-name` | Auto (aus Dateiname) | Override für die Modell-ID in `/v1/models`. |
| `--context-size` | Model-Default | KV-Cache Kontextlänge. |
| `--fp8-kv` | `false` | FP8 KV-Cache aktivieren. |
| `--quantize-on-load` | `false` | On-the-fly Q4_K Quantisierung. |
| `--threads` | Auto (Zen4 P-Cores) | CPU-Threads für Tokenizer + lm_head. |

**Startup-Sequenz:**

```
1. CLI-Parameter parsen
2. Modell laden (GGUF/SafeTensors, Shader kompilieren, VRAM allozieren)
3. InferenceEngine initialisieren
4. axum Router aufbauen
5. TCP-Listener binden
6. Log: "VulkanForge v0.4.0 serving Qwen3-8B-Q4_K_M on http://127.0.0.1:8080"
7. Log: "  Decode: ~121 tok/s | Prefill: ~1197 tok/s | VRAM: 5.8 GiB / 16 GiB"
8. Log: "  Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health"
9. Server läuft, akzeptiert Connections
```

### 5.4 Graceful Shutdown

Auf `SIGINT` (Ctrl-C) oder `SIGTERM`:

1. Stoppe Accept von neuen Connections.
2. Warte bis laufender Inference-Request fertig ist (max 30s Timeout).
3. Laufende SSE-Verbindungen: Sende `data: [DONE]\n\n` und schließe.
4. GPU-Ressourcen freigeben (VkDevice cleanup).
5. Log: "VulkanForge server shut down gracefully."
6. Exit 0.

**Implementierung:** `tokio::signal::ctrl_c()` + `axum::serve(...).with_graceful_shutdown(shutdown_signal)`.

---

## §6 Chat-Template-Integration

### 6.1 OpenAI Messages → VF Internal Format

OpenAI's `messages[]` Array wird auf VFs Chat-Template-System gemappt:

```
OpenAI Request:
  messages: [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "What is Rust?" },
    { role: "assistant", content: "Rust is a systems language." },
    { role: "user", content: "Tell me more." }
  ]

VF Internal (nach Template-Anwendung, Beispiel Qwen3):
  <|im_start|>system
  You are helpful.<|im_end|>
  <|im_start|>user
  What is Rust?<|im_end|>
  <|im_start|>assistant
  Rust is a systems language.<|im_end|>
  <|im_start|>user
  Tell me more.<|im_end|>
  <|im_start|>assistant
```

### 6.2 Template-Auswahl

VF erkennt das Chat-Template automatisch aus dem GGUF-Metadata oder dem tokenizer_config.json (bei SafeTensors). Das ist identisch zum CLI-Verhalten. Unterstützte Templates:

| Modell-Familie | Template | Erkennung |
|---|---|---|
| Qwen3, Qwen2.5 | ChatML (`<\|im_start\|>`) | GGUF `tokenizer.chat_template` |
| Llama-3.1 | Llama-3 Format (`<\|begin_of_text\|>`) | GGUF `tokenizer.chat_template` |
| Gemma-4 | Gemma Format (`<start_of_turn>`) | GGUF `tokenizer.chat_template` |
| DeepSeek-R1 | ChatML-Variante | GGUF `tokenizer.chat_template` |
| Mistral | Mistral Format (`[INST]`) | GGUF `tokenizer.chat_template` |

### 6.3 System-Message Handling

| Szenario | Verhalten |
|---|---|
| Request hat System-Message | System-Message wird in Template eingesetzt. |
| Request hat KEINE System-Message, `--system` CLI-Flag gesetzt | CLI-Default wird als System-Message verwendet. |
| Request hat KEINE System-Message, kein CLI-Flag | Kein System-Prompt (modell-abhängiges Default-Verhalten). |
| Request hat MEHRERE System-Messages | Werden konkateniert (Newline-getrennt). Warnung im Log. |
| `role: "developer"` | Wird als `role: "system"` behandelt (OpenAI-Kompatibilität). |

### 6.4 Multi-Turn KV-Cache

In v0.4 gibt es **KEINEN** KV-Cache-Reuse zwischen Requests. Jeder Request startet mit leerem KV-Cache und vollem Prefill.

**Begründung:** KV-Cache-Reuse erfordert Conversation-Tracking (welcher Request gehört zu welcher Conversation). Das ist komplex (Session-Management, Cache-Eviction, Memory-Management) und wird auf v0.5 geschoben.

**Auswirkung:** Für Multi-Turn Conversations wird der gesamte Kontext bei jedem Request neu prefilled. Bei 1197 tok/s Prefill (Qwen3 Q4K) und 2000 Token Kontext ≈ 1.7s TTFT. Das ist akzeptabel für v0.4.

### 6.5 Think-Filter

Für Modelle mit `<think>...</think>` Tags (Qwen3 `/think` Mode):

1. VFs bestehender ThinkFilter wird in den Streaming-Path eingehängt.
2. Think-Tokens werden aus dem SSE-Stream gefiltert.
3. Die `usage.completion_tokens` zählt ALLE generierten Tokens (inkl. Think).
4. `finish_reason` bezieht sich auf den sichtbaren Output.

**Offene Entscheidung (§11):** Soll der Think-Filter per Default AN oder AUS sein? Soll er per Request steuerbar sein?

---

## §7 Sampling-Parameter-Mapping

### 7.1 Direkte Mappings

| OpenAI Parameter | VF Sampler Parameter | Transformation |
|---|---|---|
| `temperature` | `temperature` | 1:1. `temperature=0` → Greedy. |
| `top_p` | `top_p` | 1:1. |
| `max_tokens` | `max_tokens` | 1:1. `null` → `context_size - prompt_length`. |
| `seed` | `seed` | 1:1. |

### 7.2 Penalty-Mapping

OpenAI und VF haben unterschiedliche Penalty-Systeme. Die Transformation:

**OpenAI:**
- `presence_penalty` ∈ [-2.0, 2.0]: Flat-Penalty auf Tokens die im bisherigen Output vorkommen.
- `frequency_penalty` ∈ [-2.0, 2.0]: Pro-Occurrence-Penalty proportional zur Häufigkeit.

**VF (aktuell):**
- `repeat_penalty` ∈ [1.0, ∞): Multiplikator auf Logits wiederholter Tokens. Default: 1.1.
- `repeat_last_n` : Fenster für Repeat-Penalty-Berechnung.

**Mapping-Strategie:**

```rust
fn map_penalties(presence: f32, frequency: f32) -> VfSamplerConfig {
    // Wenn beide 0.0 → keine Penalty
    if presence == 0.0 && frequency == 0.0 {
        return VfSamplerConfig { repeat_penalty: 1.0, ..default() };
    }
    
    // Approximation: OpenAI addiert Penalties auf Logits,
    // VF multipliziert. Wir konvertieren:
    // effective_penalty = 1.0 + max(presence, frequency) * 0.5
    // Range: [0.0, 2.0] → [1.0, 2.0] für VF repeat_penalty
    let combined = (presence.abs() + frequency.abs()) / 2.0;
    VfSamplerConfig {
        repeat_penalty: 1.0 + combined.clamp(0.0, 1.0),
        ..default()
    }
}
```

**Anmerkung:** Das ist eine Approximation. Für exakte OpenAI-Kompatibilität müsste VF additive Penalties auf Logit-Ebene implementieren. Das ist ein v0.5 Feature.

### 7.3 Stop-Sequenzen

```rust
fn process_stop_sequences(stop: &[String], tokenizer: &Tokenizer) -> StopConfig {
    // Jede Stop-Sequenz wird tokenisiert
    let token_sequences: Vec<Vec<u32>> = stop.iter()
        .map(|s| tokenizer.encode(s))
        .collect();
    
    // Zusätzlich: String-Matching auf detokenisierten Output
    // (für Stop-Sequenzen die Token-Grenzen überspannen)
    StopConfig {
        token_sequences,
        string_patterns: stop.to_vec(),
    }
}
```

### 7.4 Parameter-Validierung

```rust
fn validate_request(req: &ChatCompletionRequest) -> Result<(), ErrorResponse> {
    if req.messages.is_empty() {
        return Err(error_400("messages", "'messages' must contain at least 1 message"));
    }
    if req.temperature < 0.0 || req.temperature > 2.0 {
        return Err(error_400("temperature", "must be between 0 and 2"));
    }
    if req.top_p <= 0.0 || req.top_p > 1.0 {
        return Err(error_400("top_p", "must be between 0 (exclusive) and 1"));
    }
    if req.n > 1 {
        return Err(error_400("n", "n > 1 is not supported in VulkanForge v0.4"));
    }
    if let Some(max) = req.max_tokens {
        if max == 0 {
            return Err(error_400("max_tokens", "must be greater than 0"));
        }
    }
    if req.stop.len() > 4 {
        return Err(error_400("stop", "maximum 4 stop sequences"));
    }
    if req.presence_penalty < -2.0 || req.presence_penalty > 2.0 {
        return Err(error_400("presence_penalty", "must be between -2 and 2"));
    }
    if req.frequency_penalty < -2.0 || req.frequency_penalty > 2.0 {
        return Err(error_400("frequency_penalty", "must be between -2 and 2"));
    }
    // Multimodal-Check: content darf kein Array sein
    for msg in &req.messages {
        if msg.role == "tool" {
            return Err(error_400("messages", "role 'tool' is not supported"));
        }
    }
    Ok(())
}
```

---

## §8 Fehlerbehandlung

### 8.1 OpenAI-kompatibles Error-Format

Alle Fehler folgen dem OpenAI-Format:

```json
{
  "error": {
    "message": "Menschenlesbare Beschreibung.",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_parameter_value"
  }
}
```

### 8.2 HTTP Status Codes

| Status | Bedeutung | Wann |
|---|---|---|
| `200` | OK | Erfolgreiche Completion. |
| `400` | Bad Request | Ungültige Parameter, fehlende Pflichtfelder, ungültiges JSON. |
| `404` | Not Found | Unbekannter Endpunkt. |
| `405` | Method Not Allowed | GET auf `/v1/chat/completions`, POST auf `/v1/models`. |
| `413` | Payload Too Large | Request Body > 10 MB. |
| `422` | Unprocessable Entity | JSON parst, aber Schema-Validierung schlägt fehl. |
| `500` | Internal Server Error | GPU-Fehler, Vulkan-Error, unerwarteter Panic. |
| `503` | Service Unavailable | Modell noch beim Laden, Server überlastet (Queue-Timeout). |

### 8.3 Error-Type Mapping

| Error-Type (OpenAI-kompatibel) | Beschreibung |
|---|---|
| `invalid_request_error` | Ungültige Parameter, fehlende Felder. |
| `not_found` | Endpunkt nicht gefunden. |
| `server_error` | Interner Fehler (GPU, OOM, Panic). |
| `service_unavailable` | Server nicht bereit. |

### 8.4 Finish-Reasons

| `finish_reason` | Bedeutung |
|---|---|
| `"stop"` | EOS-Token oder Stop-Sequenz getroffen. Normal-Fall. |
| `"length"` | `max_tokens` erreicht. Output ist abgeschnitten. |
| `"content_filter"` | Reserviert. Nicht implementiert in v0.4 (VF hat keinen Content-Filter). |

### 8.5 Error während Streaming

Tritt ein Fehler WÄHREND des Streamings auf (GPU-Timeout, Vulkan-Error):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1715500000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Pragmatische Entscheidung:** Bei einem Mid-Stream-Error wird der Stream sauber geschlossen (wie ein normales Ende). Der Client sieht einen abrupt endenden Text. OpenAI macht es genauso — es gibt kein "Error-Event" im SSE-Standard.

**Alternative (verworfen):** Ein Custom-Event `data: {"error": {...}}` senden. Das bricht die OpenAI-Kompatibilität — kein Client erwartet das.

**Logging:** Der Fehler wird serverseitig geloggt mit Request-ID und Fehlerbeschreibung.

---

## §9 Datei-Struktur

### 9.1 Modul-Layout

```
src/
├── server/
│   ├── mod.rs              // pub mod Deklarationen, re-exports
│   ├── app.rs              // Router-Setup, AppState, run_server()
│   ├── handlers.rs         // Endpunkt-Handler (chat_completions, models, health)
│   ├── types.rs            // Request/Response Structs (§3 komplett)
│   ├── streaming.rs        // SSE-Chunk-Generator, Token→Chunk Mapping
│   ├── validation.rs       // Request-Validierung (§7.4)
│   ├── error.rs            // ErrorResponse Builder, Status-Code Mapping
│   └── config.rs           // ServerConfig, CLI-Parameter-Mapping
├── cli/
│   ├── mod.rs
│   ├── chat.rs             // Bestehend: chat Subcommand
│   └── serve.rs            // NEU: serve Subcommand (CLI-Parsing → server::run_server)
└── ...                     // Bestehende VF-Module (forward/, backend/, etc.)
```

### 9.2 Modul-Verantwortlichkeiten

| Modul | Zeilen (geschätzt) | Verantwortung |
|---|---|---|
| `types.rs` | ~250 | Alle Structs aus §3. Keine Logik. |
| `handlers.rs` | ~200 | 3 Handler-Funktionen. Orchestration: Validation → Lock → Inference → Response. |
| `streaming.rs` | ~150 | `InferenceStream` Struct, `fn token_to_chunk()`, Think-Filter Integration. |
| `app.rs` | ~120 | `AppState`, `fn build_router()`, `async fn run_server()`, Graceful Shutdown. |
| `validation.rs` | ~100 | `validate_request()`, Range-Checks, Typ-Checks. |
| `error.rs` | ~80 | `error_400()`, `error_500()`, `error_503()`, `impl IntoResponse for ErrorResponse`. |
| `config.rs` | ~60 | `ServerConfig` Struct, Default-Werte, CLI→Config Mapping. |
| `serve.rs` | ~50 | CLI-Argument-Parsing für `serve` Subcommand. |
| **Total** | **~1010** | |

**Vergleich llama.cpp:** llama.cpp's `server.cpp` ist eine einzelne Datei mit ~4000+ Zeilen. VFs modularer Ansatz ist wartbarer und folgt dem VF-Coding-Standard (kein File > 2074 LOC, aus v0.3.14).

### 9.3 Abhängigkeits-Graph

```
cli/serve.rs ──→ server/config.rs ──→ server/app.rs
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                   server/handlers.rs  server/streaming.rs  server/error.rs
                          │               │
                          ▼               ▼
                   server/types.rs   server/validation.rs
                          │
                          ▼
                   (bestehende VF-Module: forward/, backend/, tokenizer/)
```

Keine zirkulären Abhängigkeiten. `types.rs` und `error.rs` sind Blatt-Module.

---

## §10 Test-Strategie

### 10.1 Unit-Tests

| Test-Modul | Was getestet wird | Anzahl Tests (geschätzt) |
|---|---|---|
| `types.rs` | Serde Deserialisierung: gültige Requests, Defaults, fehlende Felder, Stop als String vs Array | 8 |
| `validation.rs` | Alle Range-Checks, leere Messages, n>1, ungültige Rollen | 12 |
| `streaming.rs` | Chunk-Generierung: erster Chunk (Role), mittlere Chunks, letzter Chunk, Usage-Chunk, Think-Filter | 8 |
| `error.rs` | Error-Serialisierung, Status-Code Mapping | 4 |
| **Total** | | **32** |

### 10.2 Integration-Tests (curl)

Manuell ausführbar. Werden als `tests/server_integration.sh` gespeichert:

```bash
#!/bin/bash
# VulkanForge API Server Integration Tests
BASE="http://127.0.0.1:8080"

# 1. Health Check
echo "=== Health Check ==="
curl -s $BASE/health | jq .

# 2. Models
echo "=== Models ==="
curl -s $BASE/v1/models | jq .

# 3. Non-Streaming Completion
echo "=== Non-Streaming ==="
curl -s $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anything",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }' | jq .

# 4. Streaming Completion
echo "=== Streaming ==="
curl -sN $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anything",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "max_tokens": 50,
    "stream": true
  }'

# 5. Error: Empty Messages
echo "=== Error: Empty Messages ==="
curl -s -w "\nHTTP Status: %{http_code}\n" $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "x", "messages": []}' | jq .

# 6. Error: Invalid Temperature
echo "=== Error: Invalid Temperature ==="
curl -s -w "\nHTTP Status: %{http_code}\n" $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "x", "messages": [{"role": "user", "content": "hi"}], "temperature": 5.0}' | jq .

# 7. Error: Unknown Endpoint
echo "=== Error: Unknown Endpoint ==="
curl -s -w "\nHTTP Status: %{http_code}\n" $BASE/v1/completions \
  -H "Content-Type: application/json" \
  -d '{}' | jq .

# 8. OpenAI Python SDK Kompatibilität
echo "=== OpenAI Python SDK ==="
python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://127.0.0.1:8080/v1', api_key='not-needed')
r = client.chat.completions.create(
    model='anything',
    messages=[{'role': 'user', 'content': 'Say hello.'}],
    max_tokens=10
)
print(r.choices[0].message.content)
print(f'Usage: {r.usage}')
"
```

### 10.3 Kompatibilitäts-Tests

| UI/Tool | Test-Methode | Akzeptanz-Kriterium |
|---|---|---|
| **Open WebUI** | `docker run -e OPENAI_API_BASE_URL=http://host:8080/v1 ...` | Chat funktioniert, Streaming flüssig, keine Fehler in UI-Console. |
| **SillyTavern** | Custom API Endpoint konfigurieren | Chat + Streaming funktioniert. |
| **Continue.dev** | VS Code Extension: `base_url` konfigurieren | Code-Completion Streaming funktioniert. |
| **Cursor** | Settings → Custom Model: base_url + model-name | Completion funktioniert. |
| **OpenAI Python SDK** | `openai.OpenAI(base_url=..., api_key="not-needed")` | `chat.completions.create()` + Streaming funktioniert. |
| **curl** | Direkte HTTP-Requests | Korrekte JSON-Responses. |

### 10.4 Performance-Tests

| Metrik | Test-Methode | Akzeptanz |
|---|---|---|
| TTFT | `time curl ... --max-time 5 -o /dev/null -w "%{time_starttransfer}"` | < 100ms (warm), < 500ms (cold pp512) |
| Decode Throughput | `completion_tokens / (last_chunk_time - first_chunk_time)` | ≥ 100 tok/s (Q4_K_M Qwen3-8B) |
| Server Overhead | `CLI tok/s - Server tok/s` | < 1% Differenz |
| Memory Overhead | `RSS(server) - RSS(cli)` | < 20 MB |
| Concurrent Rejection | 2 parallele curl Requests | Erster completes, Zweiter wartet (nicht 500) |

---

## §11 Offene Entscheidungen

Entscheidungen die der Owner (mg) treffen muss vor Implementierung:

### OE-1: CORS

**Frage:** Soll CORS für v0.4 aktiviert werden?

**Kontext:** Open WebUI und andere Browser-basierte UIs brauchen CORS wenn sie auf einem anderen Port laufen als VF. SillyTavern (Electron) braucht kein CORS.

**Optionen:**
- **A:** CORS OFF (Default). User muss Reverse-Proxy nutzen oder UI auf gleichem Origin hosten.
- **B:** CORS ON mit `Access-Control-Allow-Origin: *`. Einfach, aber Security-Implikation bei non-localhost.
- **C:** `--cors` CLI-Flag (opt-in). Default OFF.

**Empfehlung:** Option C. `--cors` Flag, Default OFF.

### OE-2: Think-Filter Default

**Frage:** Soll der Think-Filter per Default AN oder AUS sein?

**Kontext:** Im CLI ist der Think-Filter AN. Im Server-Kontext könnte ein Client den Think-Output wollen (z.B. für Debug).

**Optionen:**
- **A:** AN (wie CLI). Konsistent.
- **B:** AUS. Client bekommt Raw-Output inkl. `<think>` Tags.
- **C:** CLI-Flag `--think-filter` (Default: ON) + zukünftig per-Request Header.

**Empfehlung:** Option C.

### OE-3: API-Key

**Frage:** Soll VF einen API-Key akzeptieren (auch wenn nicht validiert)?

**Kontext:** Viele Clients senden zwingend einen `Authorization: Bearer sk-...` Header. llama.cpp akzeptiert aber ignoriert ihn. Wenn VF den Header ablehnt, brechen diese Clients.

**Optionen:**
- **A:** Header ignorieren (wie llama.cpp). Einfach, korrekt.
- **B:** Optionaler `--api-key` Flag. Wenn gesetzt: validieren. Wenn nicht: ignorieren.

**Empfehlung:** Option A für v0.4, Option B für v0.5.

### OE-4: Request-Timeout

**Frage:** Maximale Dauer pro Request?

**Kontext:** Ein `max_tokens=32768` Request bei 121 tok/s dauert ~270 Sekunden.

**Optionen:**
- **A:** Kein Timeout. Request läuft bis fertig.
- **B:** `--timeout` CLI-Flag, Default 300s (5 Minuten).

**Empfehlung:** Option B. 300s Default.

### OE-5: Logging-Format

**Frage:** Welches Log-Format für Server-Requests?

**Optionen:**
- **A:** Strukturiertes JSON-Logging (maschinenlesbar).
- **B:** Menschenlesbares Format (wie aktuell im CLI).
- **C:** `--log-format json|text` Flag.

**Empfehlung:** Option B für v0.4, Option C für v0.5.

### OE-6: Default-Port

**Frage:** 8080 (wie llama.cpp) oder 11434 (wie Ollama)?

**Empfehlung:** 8080 (llama.cpp Konvention, weiter verbreitet in UI-Defaults).

### OE-7: `/v1/` Prefix Optional?

**Frage:** Soll `/chat/completions` (ohne `/v1/`) auch funktionieren?

**Kontext:** Manche Clients senden `/v1/chat/completions`, manche nur `/chat/completions`. llama.cpp unterstützt beides.

**Empfehlung:** Ja. Beide Pfade routen zum selben Handler.

---

## §12 Referenz-Links

| Quelle | URL | Verwendung |
|---|---|---|
| OpenAI Chat Completions API | https://platform.openai.com/docs/api-reference/chat/create | Request/Response Schema, Feld-Definitionen |
| OpenAI Models API | https://platform.openai.com/docs/api-reference/models/list | `/v1/models` Response-Format |
| OpenAI Error Format | https://platform.openai.com/docs/guides/error-codes | Error-Response Struktur |
| OpenAI Streaming Guide | https://platform.openai.com/docs/api-reference/streaming | SSE-Chunk-Format |
| llama.cpp Server README | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | Kompatibilitäts-Referenz, Feature-Vergleich |
| vLLM OpenAI-kompatibler Server | https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html | Kompatibilitäts-Referenz |
| axum Dokumentation | https://docs.rs/axum/latest/axum/ | Framework-API |
| axum SSE Modul | https://docs.rs/axum/latest/axum/response/sse/ | SSE-Implementation |
| MDN Server-Sent Events | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events | SSE-Protokoll-Spezifikation |
| tower-http CORS | https://docs.rs/tower-http/latest/tower_http/cors/ | CORS-Middleware |
| Open WebUI Docs | https://docs.openwebui.com/ | Kompatibilitäts-Anforderungen |
| SillyTavern Docs | https://docs.sillytavern.app/ | Kompatibilitäts-Anforderungen |

---

## Anhang A: Vollständiges Request/Response Beispiel

### Non-Streaming

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

**Response:**
```json
{
  "id": "chatcmpl-vf_Xa7bC2dEf",
  "object": "chat.completion",
  "created": 1715500042,
  "model": "Qwen3-8B-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A mutex (mutual exclusion) is a synchronization primitive that ensures only one thread can access a shared resource at a time. When a thread locks a mutex, other threads attempting to lock it will block until the mutex is released."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 43,
    "total_tokens": 71
  }
}
```

### Streaming

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

**Response:**
```
data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":" How"},"finish_reason":null}]}

data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-vf_Xa7bC2dEf","object":"chat.completion.chunk","created":1715500042,"model":"Qwen3-8B-Q4_K_M","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}

data: [DONE]
```

---

## Anhang B: Implementierungs-Reihenfolge

Empfohlene Sprint-Sequenz für 2-3 Tage Implementierung:

```
Sprint 1 (4h): Foundation
  □ server/types.rs — alle Structs aus §3
  □ server/error.rs — ErrorResponse + IntoResponse
  □ server/validation.rs — validate_request()
  □ server/config.rs — ServerConfig
  □ Unit-Tests für types + validation

Sprint 2 (4h): Non-Streaming
  □ server/app.rs — AppState, Router, run_server()
  □ server/handlers.rs — health_handler, models_handler
  □ server/handlers.rs — chat_completions_handler (non-streaming)
  □ cli/serve.rs — serve Subcommand
  □ Integration-Test: curl non-streaming

Sprint 3 (4h): Streaming
  □ server/streaming.rs — SSE-Chunk-Generator
  □ server/handlers.rs — Streaming-Path in chat_completions_handler
  □ Think-Filter Integration
  □ Integration-Test: curl streaming
  □ Integration-Test: OpenAI Python SDK

Sprint 4 (2h): Polish + Compat
  □ CORS (wenn OE-1 entschieden)
  □ Graceful Shutdown
  □ Open WebUI Test
  □ Performance-Vergleich CLI vs Server
  □ README Update
```

**Geschätzter Gesamtaufwand:** ~14h (2 Tage fokussierte Arbeit).

---

*Ende des Dokuments. Freigabe durch Owner ausstehend.*
