# LLM Context-Store — Ergänzung: Real-Time Ingestion + Storage-Scope

**Ergänzt:** 2026-05-19
**Basis:** context_store_konzept.md (Graph-Schema, Lifecycle, Query-Patterns, Integration, Skalierung, Semantische Suche)

---

## 10. Real-Time Ingestion

### 10.1 Das Grundproblem

Das bisherige `ctx-commit`-am-Session-Ende-Modell hat eine stille Annahme: Sessions enden kontrolliert. In der Praxis:

```
Session-Typ              Dauer      Crash-Risiko    Nuggets
──────────────────────────────────────────────────────────────
Quick-Fix (1 Bug)        15-30 min  niedrig         2-5
Sprint (Feature)         2-4 h      mittel          10-20
Marathon (Refactor)      4-8 h      hoch            20-40
Debug-Rabbit-Hole        1-3 h      hoch            5-15
```

Bei Marathon-Sessions mit 30+ Nuggets und Crash-Risiko ist End-of-Session-Commit Datenverlust-Roulette. Das hat VulkanForge real betroffen — Sprint 44A-44C (forward.rs Refactor, 7 Sprints, 17 Commits) hätte bei einem Crash zwischendurch Wochen an Entscheidungsdokumentation verloren.

### 10.2 Architektur: Write-Ahead Knowledge Queue

Der Mechanismus ist bewusst simpel gehalten — kein eigener Daemon, kein Message-Broker, keine Async-Runtime. Stattdessen: eine FIFO-Datei und ein Background-Consumer.

```
┌─────────────────────────┐
│  Agent (OpenCode)       │
│  oder Stream-Parser     │
│                         │
│  erkennt Nugget         │
│       │                 │
│       ▼                 │
│  echo JSON >> fifo      │  ← non-blocking, <1ms
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  ~/.ctx/ingestion.fifo  │  ← named pipe (mkfifo)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ctx-consumer (fish)    │  ← Background-Loop, liest FIFO
│                         │
│  1. JSON parsen         │
│  2. Dedup prüfen        │
│  3. sqlitegraph write   │
│  4. Embedding queuen    │
└─────────────────────────┘
```

**Warum FIFO statt direkt schreiben?** SQLite-Writes sind serialisiert (WAL-Mode erlaubt concurrent reads, aber nur einen Writer). Wenn der Agent direkt schreibt, blockiert er bei concurrent Writes. Die FIFO entkoppelt Produzent und Consumer — der Agent schreibt in <1ms in die Pipe und macht weiter, der Consumer schreibt sequentiell in die DB.

### 10.3 Implementierung

**Producer-Seite (Agent schreibt Nuggets):**

```fish
# ctx-emit: Non-blocking Nugget in die Queue schreiben
function ctx-emit
  set kind $argv[1]    # Learning|Decision|Benchmark|Bug|Hypothesis
  set name $argv[2]
  set payload $argv[3] # JSON-String mit kind-spezifischen Feldern
  
  set nugget_id (uuidgen | head -c 8)
  set timestamp (date -Iseconds)
  
  # Atomar in FIFO schreiben, non-blocking
  echo "{
    \"id\": \"$nugget_id\",
    \"kind\": \"$kind\",
    \"name\": \"$name\",
    \"data\": $payload,
    \"session\": \"$CTX_CURRENT_SESSION\",
    \"timestamp\": \"$timestamp\",
    \"status\": \"draft\"
  }" >> ~/.ctx/ingestion.fifo &
end
```

**Consumer (Background-Prozess):**

```fish
# ctx-consumer: Läuft im Hintergrund, liest FIFO, schreibt in SG
function ctx-consumer
  while true
    # Blockiert bis Daten in der FIFO sind
    while read -l line < ~/.ctx/ingestion.fifo
      # Dedup: Prüfe ob nugget_id schon existiert
      set nugget_id (echo $line | jq -r '.id')
      set exists (sqlitegraph --db $CTX_DB query "
        MATCH (n) WHERE n.data.nugget_id = '$nugget_id'
        RETURN count(n)
      " | jq '.[0]')
      
      if test "$exists" = "0"
        set kind (echo $line | jq -r '.kind')
        set name (echo $line | jq -r '.name')
        set data (echo $line | jq -c '.data')
        set session (echo $line | jq -r '.session')
        set timestamp (echo $line | jq -r '.timestamp')
        
        # In Graph schreiben
        sqlitegraph --db $CTX_DB query "
          CREATE (n:$kind {
            name: '$name',
            data: $data
          })
        "
        
        # Session-Edge
        if test -n "$session"
          sqlitegraph --db $CTX_DB query "
            MATCH (s:Session {name: '$session'})
            MATCH (n:$kind {name: '$name'})
            WHERE n.data.nugget_id = '$nugget_id'
            CREATE (s)-[:PRODUCED]->(n)
          "
        end
        
        # Embedding in separater Queue (Batch, nicht pro Nugget)
        echo "$nugget_id" >> ~/.ctx/embed_queue.txt
      end
    end
  end
end
```

**Consumer starten (Session-Start):**

```fish
# In ctx-load integriert:
function ctx-load
  # ... bestehender Context-Load ...
  
  # Session-Variable setzen
  set -gx CTX_CURRENT_SESSION "sprint-$argv[2]"
  
  # Consumer starten falls nicht läuft
  if not pgrep -f ctx-consumer > /dev/null
    ctx-consumer &
    disown
  end
end
```

### 10.4 Agent-gesteuert vs. automatisch

Beides — aber mit klarer Aufgabenteilung:

```
Methode              Was                           Wann
──────────────────────────────────────────────────────────────
Agent-gesteuert      Decisions, Learnings,         Agent erkennt aktiv:
                     Hypotheses                    "Das ist eine Entscheidung"
                     
Automatisch          Benchmarks, Commits,          Pattern-Match auf
                     Bugs (aus Compiler-Output)    strukturiertem Output
                     
Semi-automatisch     Session-Summary               End-of-Session oder
                                                   nach N Minuten Inaktivität
```

**Warum nicht rein automatisch?** Weil die Unterscheidung "ist das eine Erkenntnis oder nur ein Debug-Schritt?" semantisches Verständnis erfordert. Ein Regex-Parser kann `121 tok/s` aus einem Benchmark-Output extrahieren, aber er kann nicht entscheiden ob "VRAM-Alloc-Reihenfolge beeinflusst BW" eine fundamentale Erkenntnis oder ein Zwischenschritt ist. Der Agent hat dieses Verständnis.

**Warum nicht rein Agent-gesteuert?** Weil Benchmarks und Commits ein vorhersagbares Format haben. Die automatische Extraktion ist zuverlässiger als darauf zu hoffen, dass der Agent am Ende jeder Messung `ctx-emit` aufruft.

**Automatischer Benchmark-Parser (Beispiel):**

```fish
# In VulkanForge Build-Script integriert
# Parst Inference-Output und emittiert Benchmarks automatisch
function vf-bench-parse
  # VulkanForge output format: "decode: 121.3 tok/s (209W)"
  cat - | while read -l line
    echo $line  # Passthrough zum Terminal
    
    if string match -rq 'decode:\s+([0-9.]+)\s+tok/s\s+\(([0-9.]+)W\)' -- $line
      set tps $match[1]
      set watts $match[2]
      ctx-emit Benchmark "bench-(date +%s)" "{
        \"metric\": \"decode_tok_s\",
        \"value\": $tps,
        \"power_w\": $watts,
        \"nugget_id\": \"bench-(uuidgen | head -c 8)\",
        \"status\": \"draft\"
      }"
    end
  end
end

# Nutzung:
cargo run --release -- --model qwen3-8b 2>&1 | vf-bench-parse
```

### 10.5 Draft → Confirmed Lifecycle

Jedes Nugget das während der Session geschrieben wird, hat `status: "draft"`. Am Session-Ende gibt es drei Möglichkeiten:

```
                    ctx-emit (während Session)
                           │
                           ▼
                    ┌──────────────┐
                    │ status: draft │
                    └──────┬───────┘
                           │
              Session-Ende │
                           │
             ┌─────────────┼─────────────────┐
             │             │                 │
             ▼             ▼                 ▼
      ┌────────────┐ ┌──────────┐    ┌────────────┐
      │ confirmed  │ │ promoted │    │ discarded  │
      │ (default)  │ │ (+pinned)│    │ (manuell)  │
      └────────────┘ └──────────┘    └────────────┘
```

**Default: Alle Drafts → confirmed.** Am Session-Ende führt `ctx-finalize` ein Bulk-Update durch:

```fish
function ctx-finalize
  # 1. Alle Drafts dieser Session → confirmed
  sqlitegraph --db $CTX_DB query "
    MATCH (s:Session {name: '$CTX_CURRENT_SESSION'})-[:PRODUCED]->(n)
    WHERE n.data.status = 'draft'
    SET n.data.status = 'confirmed',
        n.data.confirmed_at = datetime('now')
  "
  
  # 2. Agent kann optional Nuggets discarden
  # (wird im Session-Summary-Prompt angeboten)
  
  # 3. Batch-Embedding für alle neuen confirmed Nodes
  cat ~/.ctx/embed_queue.txt | sort -u | while read nugget_id
    embed-node-by-nugget-id $nugget_id
  end
  echo -n > ~/.ctx/embed_queue.txt  # Queue leeren
  
  # 4. Session-Summary generieren
  set nugget_count (sqlitegraph --db $CTX_DB query "
    MATCH (s:Session {name: '$CTX_CURRENT_SESSION'})-[:PRODUCED]->(n)
    WHERE n.data.status = 'confirmed'
    RETURN count(n)
  " | jq '.[0]')
  
  echo "Session $CTX_CURRENT_SESSION finalisiert: $nugget_count Nuggets confirmed."
end
```

**Bei Crash (kein `ctx-finalize`):** Drafts bleiben Drafts. Beim nächsten `ctx-load` werden verwaiste Drafts angezeigt:

```fish
# In ctx-load:
set orphan_drafts (sqlitegraph --db $CTX_DB query '
  MATCH (n)
  WHERE n.data.status = "draft"
  RETURN n.kind, n.name, n.data.timestamp
  ORDER BY n.data.timestamp
')

if test (echo $orphan_drafts | jq length) -gt 0
  echo "⚠ $(echo $orphan_drafts | jq length) unfinalisierte Drafts aus vorheriger Session:"
  echo $orphan_drafts | jq -r '.[] | "  \(.kind): \(.name) (\(.timestamp))"'
  echo "Finalisieren mit: ctx-recover-drafts"
end
```

### 10.6 Idempotenz

Drei Schichten Dedup-Schutz:

**Schicht 1: Nugget-ID.** Jedes Nugget bekommt eine UUID (8-char Kurzform). Der Consumer prüft vor dem Write ob diese ID existiert. Identischer Inhalt mit neuer ID wird als neues Nugget behandelt — das ist korrekt, weil der Agent die ID kontrolliert.

**Schicht 2: Content-Hash für Benchmarks.** Automatisch extrahierte Benchmarks (aus dem Stream-Parser) bekommen einen Hash über `(metric, model, value, timestamp_minute)`. Gleicher Benchmark innerhalb derselben Minute wird dedupliziert:

```fish
# Im Benchmark-Parser:
set content_hash (echo "$metric|$model|$tps|$(date +%Y%m%dT%H%M)" | sha256sum | head -c 12)
# content_hash als nugget_id verwenden
```

**Schicht 3: Session-Scope.** `ctx-finalize` prüft auf inhaltliche Duplikate innerhalb einer Session (zwei Learnings mit identischem `claim`-Text) und merged sie. Das fängt den Fall ab, dass der Agent dieselbe Erkenntnis zweimal formuliert:

```fish
# In ctx-finalize: Dedup innerhalb Session
sqlitegraph --db $CTX_DB query "
  MATCH (s:Session {name: '$CTX_CURRENT_SESSION'})-[:PRODUCED]->(l1:Learning)
  MATCH (s)-[:PRODUCED]->(l2:Learning)
  WHERE l1.data.claim = l2.data.claim
    AND id(l1) < id(l2)
  DELETE l2
"
```

### 10.7 Backpressure: Was wenn SG langsamer ist als der Token-Stream?

In der Praxis ist das kein Problem. Rechnung:

```
Token-Stream:       ~100 tok/s (VulkanForge Decode)
Nugget-Rate:        ~1 pro 2-5 Minuten (Agent-gesteuert)
                    ~1 pro 30 Sekunden (automatische Benchmarks, Peak)
SG Write-Latenz:    ~5-10ms pro INSERT (SQLite WAL)
SG Throughput:      ~100-200 Writes/s

→ Nugget-Rate (max ~2/min) << SG-Throughput (12.000/min)
→ Faktor 6.000× Headroom
```

Die FIFO ist trotzdem richtig, nicht wegen Throughput, sondern wegen Latenz-Entkopplung: der Agent soll nicht auf den 10ms SQLite-Write warten wenn er gerade einen 4096-Token-Prefill verarbeitet. Fire-and-forget in die FIFO (<1ms) ist die richtige Semantik.

**Edge-Case: Batch-Import.** Wenn jemand 500 historische Commits importiert, könnte der Consumer hinterherhinken. Lösung: Batch-Import geht direkt über `sqlitegraph` CLI, nicht über die FIFO. Die FIFO ist nur für den Live-Pfad.

---

## 11. Storage-Scope: Was kommt rein, was nicht

### 11.1 Die Knowledge-Distillation-Pyramide

Nicht alles was in einer Session passiert ist Wissen. Die Pyramide zeigt das Verhältnis:

```
                    ▲
                   ╱ ╲
                  ╱   ╲         Persistentes Wissen
                 ╱ 5%  ╲        (Decisions, Learnings, Benchmarks)
                ╱───────╲
               ╱         ╲      Strukturierte Fakten
              ╱    15%    ╲     (Commits, Configs, Bug-Status)
             ╱─────────────╲
            ╱               ╲   Kontext-Informationen
           ╱      30%       ╲   (Hypothesen, Zwischenergebnisse)
          ╱───────────────────╲
         ╱                     ╲ Noise
        ╱         50%          ╲ (Debug-Logs, Smalltalk,
       ╱                        ╲ Compiler-Output, Reasoning)
      ╱──────────────────────────╲
```

Der Context-Store speichert die oberen ~20%. Alles darunter lebt im Terminal-Scrollback, in Git, oder nirgends.

### 11.2 Taxonomie: Was wird gespeichert

```
┌────────────────┬─────────────────────────────────────────────────────┬────────┐
│ Kategorie      │ Beispiel                                           │ Speich.│
├────────────────┼─────────────────────────────────────────────────────┼────────┤
│                │                                                     │        │
│ DECISIONS      │ "LayerStep enum statt funktionaler Split"           │ IMMER  │
│                │ "GPU-direct statt mid_frame_submit für MoE"         │        │
│                │ "MiniLM für Embeddings, nicht Qwen3"                │        │
│                │                                                     │        │
│ LEARNINGS      │ "Dispatch-Overhead ist NICHT der Bottleneck"        │ IMMER  │
│  (positive)    │ "VRAM-Alloc-Reihenfolge beeinflusst BW"            │        │
│                │ "Touch-on-Read verhindert falsches Aging"           │        │
│                │                                                     │        │
│ HONEST-NEG     │ "mul_mm_id Down driftet, 2× gescheitert"           │ IMMER  │
│  (negative)    │ "Q8_1 Noise-Hypothese war FALSCH"                  │        │
│                │ "Dynamic Gating ε=0.01 FALSIFIZIERT"               │        │
│                │                                                     │        │
│ BENCHMARKS     │ "Qwen3 decode: 121 tok/s @ 209W"                   │ IMMER  │
│                │ "26B prefill: 40→93 tok/s (+133%)"                 │        │
│                │                                                     │        │
│ BUGS           │ "FMA Race in MoE Expert Dispatch"                   │ IMMER  │
│  (+ Status)    │ "gemv_up 2× langsamer als gemv_gate: OPEN"         │        │
│                │                                                     │        │
│ HYPOTHESES     │ "Cache-Thrashing als Ursache für BW-Drop"           │ JA*    │
│  (+ Ergebnis)  │ → FALSIFIZIERT in Sprint I                         │        │
│                │ *nur mit Ergebnis oder wenn >1 Stunde investiert    │        │
│                │                                                     │        │
│ COMMITS        │ "b2cb0d4: lm_head alloc first, +34% decode"        │ AUTO   │
│                │ (Git-Hook, automatisch)                             │        │
│                │                                                     │        │
│ ARCH-CHANGES   │ "Neuer Shader sigmoid_mul.comp für Gated Attn"     │ JA     │
│                │ "forward.rs → 13 Module, LayerStep enum"            │        │
│                │                                                     │        │
│ GEN-PROMPTS    │ "cyberpunk city, neon, volumetric lighting"         │ JA     │
│  (Bild/Musik)  │ "synthwave 128bpm melancholic retro"               │        │
│                │                                                     │        │
│ CONFIGS        │ "steps=30, cfg=7.5, seed=42"                       │ JA     │
│                │ "VF_MOE_GROUPED=1, VULKANFORGE_KV_FP8=1"           │        │
│                │                                                     │        │
│ INTERACTIONS   │ "HN Post: Why Gemma-4 26B MoE breaks in prod"      │ JA     │
│                │ "Mesa #15396 Update: LOAD_TR"                       │        │
│                │                                                     │        │
├────────────────┼─────────────────────────────────────────────────────┼────────┤
│                │                                                     │        │
│ CONVERSATIONS  │ Volle Frage-Antwort-Paare                           │ NEIN   │
│                │                                                     │        │
│ DEBUG-OUTPUT   │ Compiler-Fehler, Stack-Traces, rocprof-Logs         │ NEIN   │
│                │                                                     │        │
│ REASONING      │ "Lass mich überlegen..." / Zwischen-Analyse         │ NEIN   │
│                │                                                     │        │
│ CODE           │ Quellcode-Snippets (leben in Git)                   │ NEIN   │
│                │                                                     │        │
│ TEMP-METRICS   │ Einzelne rocprof-Counter, Zwischenmessungen         │ NEIN   │
│                │                                                     │        │
│ META           │ "ok", "weiter", "gut", "verstanden"                │ NEIN   │
│                │                                                     │        │
│ TOOL-OUTPUT    │ ls, cat, grep-Ergebnisse                           │ NEIN   │
│                │                                                     │        │
└────────────────┴─────────────────────────────────────────────────────┴────────┘
```

### 11.3 Speicher-Entscheidung: Agent-Heuristiken

Der Agent braucht klare Regeln wann er `ctx-emit` aufruft. Diese Regeln gehören in den System-Prompt:

```
KNOWLEDGE-DISTILLATION-REGELN

Du MUSST ctx-emit aufrufen wenn:
  1. Du eine Architektur- oder Design-Entscheidung triffst oder empfiehlst
  2. Du eine Performance-Messung durchführst (Benchmark)
  3. Du einen Bug findest oder fixst
  4. Du eine Erkenntnis gewinnst die für zukünftige Sessions relevant ist
  5. Du eine Hypothese falsifizierst (Honest-Negative)

Du DARFST ctx-emit aufrufen wenn:
  6. Du eine Hypothese aufstellst die du NICHT in dieser Session testen kannst
  7. Du eine Architektur-Änderung vornimmst (neuer Shader, neues Modul)
  8. Du eine Config/Env-Variable einführst oder änderst

Du DARFST NICHT ctx-emit aufrufen für:
  9. Debug-Schritte die nicht zu einer Erkenntnis führen
  10. Code-Änderungen (die leben in Git)
  11. Compiler-Output, Logs, Tool-Ergebnisse
  12. Deine eigenen Überlegungen und Zwischen-Analysen
  13. Bestätigungen, Statusmeldungen, Smalltalk

DER TEST: Würde ein neuer Agent, der nächste Woche an diesem Projekt
arbeitet, dieses Wissens-Nugget brauchen? Wenn ja → ctx-emit.
Wenn nein → nicht speichern.
```

**Zusätzliche Heuristik: Marker-Wörter.** Bestimmte Formulierungen im Agent-Output korrelieren stark mit speicherwürdigem Wissen:

```
Starke Signale (fast immer speichern):
  "Entscheidung:", "Decision:", "Erkenntnis:", "LEKTION:"
  "FALSIFIZIERT", "BESTÄTIGT", "VERIFIZIERT"
  "Root Cause:", "Fix:", "Workaround:"
  "tok/s", "t/s", "ms", "GB" (Benchmark-Zahlen)
  "→" (Verbesserungs-Pfeil: "40→93 tok/s")
  "Bug:", "OPEN", "FIXED", "SHIPPED"
  
Schwache Signale (Kontext prüfen):
  "Hypothese:", "könnte sein dass", "vermutlich"
  "Test zeigt:", "Messung:"
  "Alternative:", "Option A vs B"
  
Keine Signale (nicht speichern):
  "Lass mich...", "Ich werde...", "Als nächstes..."
  "Kompiliert.", "Test bestanden.", "Sieht gut aus."
  "```" (Code-Blöcke)
```

### 11.4 Granularität: Ein Nugget pro Erkenntnis

Die richtige Einheit ist das **atomare Wissens-Nugget** — ein Fakt, eine Entscheidung, eine Messung. Nicht eine Zusammenfassung, nicht ein Paragraph, nicht eine Session.

Warum atomar und nicht aggregiert? Weil die Embedding-Suche auf atomaren Nuggets präziser ist. "FMA Race war der Bug, 1 LOC Barrier Fix" findet sich über HNSW besser als ein 500-Wort-Session-Summary in dem dieser Fakt ein Nebensatz ist.

**Größen-Limits pro Nugget:**

```
Feld             Limit          Rationale
──────────────────────────────────────────────────────────────
name             80 chars       Soll als Identifier lesbar sein
claim/choice     200 chars      Ein Satz. Wenn länger → splitten
summary          500 chars      2-3 Sätze Kontext
rationale        500 chars      Begründung für Decisions
evidence         500 chars      Beleg für Learnings
prompt.text      2000 chars     Generative Prompts können lang sein
──────────────────────────────────────────────────────────────
GESAMT pro Node  ~4 KB          data-JSON mit allen Properties
```

**Wenn es nicht in 200 chars passt:** Das Nugget ist nicht atomar genug. Splitten in mehrere Nuggets und mit `RELATED_TO`-Edges verbinden.

```
FALSCH (zu breit):
  Learning: "VRAM-Alloc-Reihenfolge beeinflusst BW auf RDNA4,
   weil der AMD/RADV Driver physisch hohe Adressen für spät
   allokierte Tensoren nutzt, was zu GDDR6 BW-Degradation führt,
   gemessen an lm_head mit 7.8% Peak-BW statt 80%"

RICHTIG (atomar, verknüpft):
  Learning: "lm_head bei ≥12GB Modellen degradiert auf 7.8% Peak-BW"
    └─ SUPPORTS → Learning: "RADV platziert spät allokierte Tensoren physisch hoch"
    └─ SUPPORTS → Learning: "GDDR6 BW-Degradation bei hohen physischen Adressen"
    └─ SUPPORTS → Decision: "VF_LMHEAD_ALLOC_FIRST=1 (output.weight zuerst)"
    └─ SUPPORTS → Benchmark: "26B: 27→36.5 tok/s (+34%) mit ALLOC_FIRST"
```

### 11.5 Session-Summary: Ergänzung, nicht Ersatz

Atomare Nuggets ersetzen die Session-Summary nicht — sie ergänzen sie. Die Summary ist ein Index-Node der auf die Nuggets zeigt:

```cypher
-- Am Session-Ende generiert der Agent eine Summary
CREATE (sum:Document {
  name: "summary-sprint-61g",
  data: {
    "artifact_type": "session_summary",
    "text": "Expert-Grouped Dispatch für 26B MoE implementiert.
             Prefill +133%, Decode +35%. FMA Race als Root Cause
             identifiziert (1 LOC Fix). Q8_1 Noise-Hypothese
             falsifiziert. Async Decode re-enabled.",
    "nugget_count": 7,
    "duration_min": 180,
    "created_at": "2026-05-16T18:00:00Z"
  }
})

-- Summary zeigt auf alle Nuggets dieser Session
MATCH (s:Session {name: "sprint-61g"})-[:PRODUCED]->(n)
WHERE n.data.status = "confirmed"
MATCH (sum:Document {name: "summary-sprint-61g"})
CREATE (sum)-[:REFERENCES]->(n)
```

Die Summary dient zwei Zwecken: sie ist selbst embedding-fähig (für grobe semantische Suche), und sie ist der menschenlesbare Einstiegspunkt wenn jemand "was ist in Sprint 61G passiert?" fragt.

### 11.6 Die Grauzone: Debugging-Prozesse

Das Debugging-Szenario aus der Aufgabenstellung ist der häufigste Grauzonenfall. Hier die Regel:

**Der Prozess ist temporär. Das Ergebnis ist persistent. Die falsifizierten Hypothesen sind persistent wenn sie zukünftige Fehler verhindern.**

Konkretes Beispiel — der 26B Decode-Bug (Sprint 54D-56C, real passiert):

```
TEMPORÄR (nicht speichern):
  - "Schauen wir uns mal die Barrier-Reihenfolge an"
  - rocprof-Output mit 400 Zeilen Counter
  - 3× Code-Änderung → kompilieren → testen
  - "Hmm, das war es nicht, zurück zum Ausgangszustand"

PERSISTENT (speichern):
  ✅ Bug("26b-decode-race") — "V-Race + Async MoE Race"
  ✅ Learning("q8-1-noise-wrong") — "Q8_1 noise hypothesis WRONG"
     category: "negative"
  ✅ Learning("isolation-test-first") — "isolation test (top_k=1) FIRST,
     nicht analytisches dispatch debugging"
     category: "positive"
  ✅ Decision("gpu-direct-expert-ffn") — "GPU-direct eliminiert
     mid_frame_submit entirely"
  ✅ Benchmark("26b-decode-27tps") — "20→27 tok/s nach Fix"
```

Die 5 falsifizierten Hypothesen im Debugging-Prozess werden nur gespeichert wenn sie eigenständigen Wert haben (d.h. jemand könnte in Zukunft die gleiche falsche Hypothese aufstellen). "Q8_1 Noise als Ursache für Decode-Drift" ist eigenständig wertvoll — das ist eine plausible Hypothese die ein neuer Agent auch aufstellen würde. "Vielleicht ist der Barrier-Offset falsch" ist zu spezifisch und wird nicht gespeichert.

**Der Test:** Würde ein neuer Agent ohne Kontext diese Hypothese auch formulieren? Wenn ja, ist die Falsifizierung speicherwürdig. Wenn die Hypothese nur aus dem lokalen Debugging-Kontext Sinn ergibt, nicht.

---

## 12. Interaktion: Real-Time Ingestion × Storage-Scope

### 12.1 Das 30-Minuten-Debug-Szenario

Konkretes Zeitprotokoll eines Debug-Vorgangs mit den Entscheidungen wann was gespeichert wird:

```
t=0     Bug entdeckt: "26B Decode produziert Müll"
        → ctx-emit Bug "26b-decode-garbage" {..., status: "open"}      ✅ SOFORT

t=3     Hypothese 1: "KV-Cache Corruption bei FP8"
        → NICHT speichern (zu früh, nicht getestet)

t=8     H1 falsifiziert: FP8 OFF → gleiches Problem
        → NICHT speichern (H1 war trivial zu testen)

t=12    Hypothese 2: "Q8_1 Quantisierungs-Rauschen"
        → NICHT speichern (noch nicht getestet)

t=18    Benchmark: "Q8_1 vs FP32 Vergleich: identische Drift-Rate"
        → ctx-emit Benchmark "26b-q8-vs-fp32" {value: "identical"}     ✅ SOFORT

t=20    H2 falsifiziert
        → ctx-emit Learning "q8-1-noise-wrong" {                       ✅ SOFORT
            category: "negative",
            claim: "Q8_1 noise hypothesis was WRONG for 26B decode"
          }
        (Weil: plausible Hypothese die andere auch aufstellen würden)

t=25    Isolation-Test: top_k=1 → Problem verschwindet
        → ctx-emit Learning "isolation-test-method" {                   ✅ SOFORT
            category: "positive",
            claim: "Isolation test top_k=1 als Debug-Methode für
                    Sampling-related Bugs"
          }

t=28    Root Cause gefunden: FMA Race in Expert Dispatch
        → ctx-emit Learning "fma-race-root-cause" {                    ✅ SOFORT
            claim: "FMA Race: Expert-FFN Output wird gelesen bevor
                    vorheriger Dispatch fertig ist"
          }

t=30    Fix: 1 LOC Barrier
        → ctx-emit Decision "barrier-after-expert-dispatch" {          ✅ SOFORT
            choice: "vkCmdPipelineBarrier nach jedem Expert-Dispatch",
            rationale: "FMA Race Prevention, 1 LOC"
          }
        → ctx-emit Benchmark "26b-post-fix" {value: 27, metric: "tok/s"} ✅ SOFORT
        → Bug-Update: status → "fixed"
```

**Resultat:** 6 Nuggets gespeichert, 3 Hypothesen verworfen (H1 zu trivial, H2 erst nach Falsifizierung gespeichert als Learning), 0 Debug-Output gespeichert. Alles in Real-Time, nichts davon wäre bei Crash verloren.

### 12.2 Benchmark-Granularität: Alle oder nur die letzte?

Alle Benchmarks speichern, aber mit unterschiedlichem Aging-Verhalten:

```
Benchmark-Typ            Speichern    Aging
──────────────────────────────────────────────────
Baseline (Release)       Ja           Pinned (nie)
Vergleichs-Messung       Ja           Normal (90d)
Zwischen-Messung         Ja           Schnell (30d)
Identische Wiederholung  Dedup        —
```

**Warum alle?** Weil Benchmark-Reihen Trends zeigen. "Decode war 20, dann 22, dann 27 tok/s" ist wertvoller als nur die finale 27. Die Zwischen-Werte altern schneller, aber sie sind für Regressionsanalyse nützlich solange man am selben Feature arbeitet.

**Implementierung:** Der automatische Benchmark-Parser taggt Zwischen-Messungen mit `benchmark_class: "intermediate"`, der Aging-Job behandelt sie anders:

```cypher
-- Intermediate Benchmarks altern nach 30 Tagen statt 90
MATCH (b:Benchmark)
WHERE b.data.benchmark_class = "intermediate"
  AND b.data.updated_at < datetime("now", "-30 days")
  AND b.data.status = "active"
SET b.data.status = "stale"
```

### 12.3 Hypothesen-Granularität: Wann speichern?

Die Regel hat zwei Dimensionen:

```
                          ┌──────────────────────────┐
                          │  Wurde getestet?          │
                          ├─────────┬────────────────┤
                          │  JA     │  NEIN          │
  ┌───────────┬───────────┼─────────┼────────────────┤
  │           │ Plausibel │ ✅ IMMER │ ✅ wenn >1h    │
  │ Würde ein │ (andere   │ Learning │ investiert     │
  │ neuer     │ würden    │ negative │ Hypothesis     │
  │ Agent die │ die auch  │          │ status: open   │
  │ gleiche   │ aufst.)   │          │                │
  │ Hypothese ├───────────┼─────────┼────────────────┤
  │ aufst.?   │ Spezifisch│ ❌ NEIN  │ ❌ NEIN        │
  │           │ (nur aus  │ (kein    │ (kein          │
  │           │ lokalem   │ Zukunfts-│ Zukunfts-      │
  │           │ Debug-    │ wert)    │ wert)          │
  │           │ Kontext)  │          │                │
  └───────────┴───────────┴─────────┴────────────────┘
```

### 12.4 Timing: Real-Time vs. End-of-Session

Die beiden Mechanismen sind komplementär, nicht alternativ:

```
Real-Time (ctx-emit, während Session):
  → Atomare Nuggets, sofort persistent
  → Schutz gegen Crash-Verlust
  → Status "draft" bis Finalisierung
  → Agent-gesteuert + automatisch (Benchmarks)
  
End-of-Session (ctx-finalize):
  → Draft → Confirmed Bulk-Update
  → Session-Summary generieren
  → Dedup über Session-Nuggets
  → Batch-Embedding berechnen
  → Verwaiste Drafts erkennen
```

Es gibt kein "entweder Real-Time oder End-of-Session". Beides zusammen ergibt das Sicherheitsnetz: Real-Time garantiert dass nichts verloren geht, End-of-Session garantiert dass alles sauber finalisiert wird.

### 12.5 Zusammenfassung: Entscheidungsbaum für den Agent

```
Agent produziert Output-Token
         │
         ▼
  ┌──────────────────────────┐
  │ Enthält der Output ein   │
  │ Marker-Signal?           │
  │ (Entscheidung, Messung,  │──── NEIN ──→ Nicht speichern.
  │  Erkenntnis, Bug, Fix)   │              Weiter generieren.
  └──────────┬───────────────┘
             │ JA
             ▼
  ┌──────────────────────────┐
  │ Würde ein neuer Agent    │
  │ dieses Wissen brauchen?  │──── NEIN ──→ Nicht speichern.
  └──────────┬───────────────┘
             │ JA
             ▼
  ┌──────────────────────────┐
  │ Passt es in 200 chars    │──── NEIN ──→ Splitten in
  │ (claim/choice)?          │              atomare Nuggets.
  └──────────┬───────────────┘
             │ JA
             ▼
  ┌──────────────────────────┐
  │ ctx-emit Kind Name Data  │
  │ (non-blocking, <1ms)     │
  └──────────────────────────┘
```
