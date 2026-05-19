# LLM Context-Store — Architektur-Konzept

**Basis:** SQLiteGraph 3.0.1 (oldnordic/sqlitegraph, GPL-3.0)
**Ziel:** Persistentes Wissen über Sessions, Projekte und Workflows hinweg
**Erstellt:** 2026-05-19

---

## 1. Graph-Schema

### 1.1 Design-Philosophie

Das Schema folgt drei Prinzipien:

**Einheitlicher Kern, domänenspezifische Daten.** Alle Workflows teilen dieselben Node- und Edge-Typen. Domänenspezifik lebt ausschließlich in den `data`-JSON-Properties — nicht in der Topologie. Das verhindert Schema-Explosion bei neuen Workflows und ermöglicht Cross-Domain-Queries ohne Joins über verschiedene Tabellen.

**Temporal by Default.** Jeder Node trägt `created_at` und `updated_at`. Jede Edge trägt `created_at`. Das ermöglicht zeitbasierte Traversals ("was wusste der Graph vor Sprint 44?") ohne separate Versionierung.

**Embedding-Ready.** Nodes mit textueller Substanz (Decisions, Learnings, Artifacts) bekommen ein Embedding-Vektor. Reine Strukturknoten (Project, Sprint) nicht — sie sind über Graph-Traversal erreichbar und brauchen keine semantische Suche.

### 1.2 Node-Typen (kind)

```
┌─────────────────────────────────────────────────────────────────┐
│  STRUKTUR (kein Embedding)                                      │
├─────────────────────────────────────────────────────────────────┤
│  Project        Oberster Container. 1 pro Workflow-Kontext.     │
│  Sprint         Zeitlich begrenzte Arbeitseinheit.              │
│  Session        Einzelne Agent-Interaktion.                     │
│  Agent          LLM/Tool-Identität (Claude, OpenCode, Udio...) │
│  Model          LLM-Modell-Version (Qwen3-8B, FLUX-dev, ...)   │
│  Tag            Freie Kategorisierung, wiederverwendbar.        │
├─────────────────────────────────────────────────────────────────┤
│  INHALT (mit Embedding)                                         │
├─────────────────────────────────────────────────────────────────┤
│  Decision       Getroffene Entscheidung mit Begründung.         │
│  Learning       Erkenntnis, Lektion, Honest-Negative.           │
│  Artifact       Generiertes Objekt (Code, Bild, Track, Video).  │
│  Benchmark      Performance-Messung mit Zahlenwerten.           │
│  Bug            Gefundener Fehler mit Status.                   │
│  Prompt         Input-Text für ein Modell.                      │
│  Config         Parameter-Set (Sampler, Style, BPM...).         │
│  Interaction    Community-Post (Reddit, HN, GitHub Issue).      │
│  Document       Architektur-Doc, Spec, README.                  │
│  Concept        Abstraktes Wissenskonzept (z.B. "FP8-WMMA").   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Node-Properties (data JSON)

Jeder Node hat `kind`, `name`, und ein `data`-JSON-Objekt. Die Properties innerhalb von `data` sind konventionsbasiert, nicht erzwungen — der Graph ist schemaless.

**Gemeinsame Properties (alle Nodes):**

```json
{
  "created_at": "2026-05-19T14:30:00Z",
  "updated_at": "2026-05-19T15:00:00Z",
  "status": "active",
  "summary": "Einzeiler für Quick-Lookup"
}
```

**Project:**
```json
{
  "domain": "software|image|music|video",
  "repo": "https://github.com/oldnordic/vulkanforge",
  "current_version": "v0.4.4",
  "tech_stack": ["rust", "vulkan", "glsl"]
}
```

**Artifact (domänenübergreifend):**
```json
{
  "artifact_type": "code|image|track|video|document|shader",
  "version": 3,
  "file_path": "/home/mg/projects/vulkanforge/src/forward/dispatch.rs",
  "file_hash": "sha256:abc123...",
  "generator": "claude|flux|udio|runway",
  "parent_prompt": "<node-id des Prompts>",
  "metrics": {
    "loc": 2074,
    "tok_s": 121.0,
    "resolution": "1024x1024",
    "bpm": 128,
    "duration_s": 240
  }
}
```

**Decision:**
```json
{
  "choice": "forward.rs Refactor → 13 Module statt Single-File",
  "alternatives_considered": ["funktionaler Split", "macro-basiert"],
  "rationale": "LayerStep enum = compile-time Bug-Prevention",
  "reversible": true,
  "impact": "high"
}
```

**Learning:**
```json
{
  "category": "positive|negative|neutral",
  "claim": "Dispatch-Overhead ist NICHT der Bottleneck auf RDNA4",
  "evidence": "Sprint 12: Cost-Modell 30× daneben",
  "applies_to": ["rdna4", "vulkan", "inference"],
  "confidence": 0.95
}
```

**Benchmark:**
```json
{
  "target": "Qwen3-8B Q4_K_M",
  "metric": "decode_tok_s",
  "value": 121.0,
  "unit": "tok/s",
  "reference": "llama.cpp",
  "reference_value": 114.2,
  "ratio": 1.06,
  "environment": {
    "gpu": "RX 9070 XT",
    "driver": "RADV Mesa 26.1",
    "power_w": 209
  }
}
```

**Prompt (für alle generativen Workflows):**
```json
{
  "text": "A cyberpunk cityscape at golden hour, volumetric lighting...",
  "system_prompt": "You are a Rust expert...",
  "model_used": "flux-dev",
  "parameters": {
    "temperature": 0.7,
    "steps": 30,
    "cfg_scale": 7.5,
    "seed": 42
  },
  "result_quality": "good|mediocre|failed",
  "iteration_of": "<node-id der vorherigen Version>"
}
```

### 1.4 Edge-Typen

```
┌──────────────────────┬──────────────────────────────────────────┐
│  Edge-Type           │  Semantik (Source → Target)              │
├──────────────────────┼──────────────────────────────────────────┤
│  CONTAINS            │  Project → Sprint, Sprint → Session     │
│  PRODUCED            │  Session → Artifact/Decision/Learning   │
│  USED_MODEL          │  Session → Model                        │
│  USED_AGENT          │  Session → Agent                        │
│  GENERATED_BY        │  Artifact → Prompt                      │
│  ITERATION_OF        │  Artifact(v2) → Artifact(v1)            │
│  DECIDED_IN          │  Decision → Session                     │
│  SUPERSEDES          │  Decision(neu) → Decision(alt)          │
│  CONTRADICTS         │  Learning → Learning                    │
│  SUPPORTS            │  Learning → Decision                    │
│  INVALIDATES         │  Learning → Decision                    │
│  REFERENCES          │  Document → Concept                     │
│  TAGGED              │  Any → Tag                              │
│  DEPENDS_ON          │  Artifact → Artifact                    │
│  BENCHMARKED_WITH    │  Benchmark → Config                     │
│  RELATED_TO          │  Any → Any (schwache Assoziation)       │
│  TRIGGERED_BY        │  Bug → Session/Artifact                 │
│  FIXED_IN            │  Bug → Session/Artifact                 │
│  RESPONDED_TO        │  Interaction → Interaction (Threads)    │
│  ABOUT               │  Interaction → Project/Concept          │
│  USED_CONFIG         │  Prompt → Config                        │
│  CROSS_POLLINATED    │  Learning(Proj A) → Decision(Proj B)    │
└──────────────────────┴──────────────────────────────────────────┘
```

**Edge-data (JSON):** Optional. Wird genutzt für Metadaten wie `confidence`, `timestamp`, oder `context`:

```json
{
  "created_at": "2026-05-19T14:30:00Z",
  "context": "Sprint 44A-44C forward.rs refactor",
  "confidence": 0.9
}
```

### 1.5 Schema-Diagramm (vereinfacht)

```
                          ┌─────────┐
                          │ Project │
                          └────┬────┘
                    CONTAINS   │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
            ┌────────┐   ┌────────┐    ┌─────────┐
            │ Sprint │   │  Tag   │    │Document │
            └───┬────┘   └────────┘    └─────────┘
        CONTAINS│              ▲ TAGGED
                ▼              │
           ┌─────────┐        ╔══════════╗
           │ Session ├────────╢ Artifact ║
           └──┬──┬───┘ PRODUCED ╚═══╤══════╝
    USED_MODEL│  │USED_AGENT        │ ITERATION_OF
              ▼  ▼                  ▼
         ┌───────┐ ┌───────┐  ╔══════════╗
         │ Model │ │ Agent │  ║ Artifact ║ (vorherige Version)
         └───────┘ └───────┘  ╚══════════╝

           ┌──────────┐    ┌──────────┐    ┌─────────┐
           │ Decision │    │ Learning │    │ Concept │
           └──────────┘    └──────────┘    └─────────┘
                │ SUPPORTS      │ CROSS_POLLINATED
                ▼               ▼
           ┌──────────┐    ┌──────────┐
           │ Decision │    │ Decision │ (anderes Projekt)
           └──────────┘    └──────────┘
```

---

## 2. Daten-Lifecycle

### 2.1 Ingest: Wie kommen Daten rein?

Es gibt vier Ingest-Pfade, sortiert nach Automatisierungsgrad:

**Pfad 1: Agent-Auto-Commit (primär für Software-Workflows)**

Der Coding-Agent (OpenCode) schreibt am Ende jeder Session automatisch. Das passiert über ein Shell-Script das der Agent als Tool aufruft:

```bash
# ctx-commit: Atomares Session-Commit
# Wird vom Agent am Session-Ende aufgerufen

sqlitegraph --db ~/context.db query '
  CREATE (s:Session {
    name: "sprint-56a-moe-dispatch",
    data: {
      "created_at": "2026-05-19T14:30:00Z",
      "agent": "opencode",
      "model": "qwen3-8b-q4k",
      "duration_min": 45,
      "summary": "Expert-Grouped Dispatch für 26B MoE",
      "git_commits": ["abc1234", "def5678"],
      "files_changed": 12,
      "tests_added": 3,
      "tests_passing": "67/67"
    }
  })
'
```

**Pfad 2: Post-Hoc Batch-Import**

Für bestehende Daten (Git-History, alte Sprint-Reports, Benchmark-CSV-Dateien). Ein Rust- oder Python-Script parsed die Quelle und erzeugt Cypher-Statements:

```bash
# Beispiel: Git-Log → Context-Store
git log --format='%H|%s|%ai' --since="2026-04-01" | while IFS='|' read hash msg date; do
  sqlitegraph --db ~/context.db query "
    CREATE (a:Artifact {
      name: '$hash',
      data: {
        \"artifact_type\": \"commit\",
        \"message\": \"$msg\",
        \"created_at\": \"$date\"
      }
    })
  "
done
```

**Pfad 3: Interaktiver CLI-Eintrag**

Für Erkenntnisse die während der Arbeit anfallen und nicht automatisiert erfasst werden:

```bash
# Alias in fish shell
function ctx-learn
  sqlitegraph --db ~/context.db query "
    CREATE (l:Learning {
      name: '$argv[1]',
      data: {
        \"category\": \"$argv[2]\",
        \"claim\": \"$argv[1]\",
        \"created_at\": \"$(date -Iseconds)\",
        \"confidence\": 0.9
      }
    })
  "
end

# Nutzung:
ctx-learn "VRAM-Alloc-Reihenfolge beeinflusst BW auf RDNA4" "positive"
```

**Pfad 4: Generative-Tool-Hooks**

Für Bild/Musik/Video-Workflows. Wrapper-Scripts um die eigentlichen Tools:

```bash
# flux-gen: FLUX-Wrapper der Prompt+Result im Graph speichert
function flux-gen
  set prompt $argv[1]
  set output (flux --prompt "$prompt" --steps 30 --output ~/gen/)
  
  sqlitegraph --db ~/context.db query "
    CREATE (p:Prompt {
      name: 'flux-$(date +%s)',
      data: {
        \"text\": \"$prompt\",
        \"model_used\": \"flux-dev\",
        \"parameters\": {\"steps\": 30},
        \"created_at\": \"$(date -Iseconds)\"
      }
    })
    CREATE (a:Artifact {
      name: '$(basename $output)',
      data: {
        \"artifact_type\": \"image\",
        \"file_path\": \"$output\",
        \"generator\": \"flux\",
        \"created_at\": \"$(date -Iseconds)\"
      }
    })
    CREATE (a)-[:GENERATED_BY]->(p)
  "
end
```

### 2.2 Lifecycle-Stufen

Jedes Wissens-Objekt durchläuft implizite Stufen, gesteuert über das `status`-Feld:

```
active ──────► stale ──────► archived ──────► (deleted)
  │               │               │
  │  90 Tage      │  180 Tage     │  manuell
  │  ohne Zugriff │  ohne Zugriff │
  │               │               │
  └─ Zugriff ─────┘               │
    resettet auf                   │
    "active"                       └─ Daten bleiben im Graph,
                                      werden aber nicht mehr
                                      in Context-Loads inkludiert
```

**Ausnahmen vom Aging:** Nodes mit `status: "pinned"` altern nie. Dafür qualifizieren sich: Architektur-Entscheidungen, fundamentale Learnings, und Benchmark-Baselines.

**Aging-Job (Cron, täglich):**

```bash
# Täglich: active → stale nach 90 Tagen ohne Edge-Traversal
sqlitegraph --db ~/context.db query '
  MATCH (n)
  WHERE n.data.status = "active"
    AND n.data.updated_at < datetime("now", "-90 days")
    AND n.data.status != "pinned"
  SET n.data.status = "stale"
'
```

### 2.3 Touch-on-Read

Jeder Query der einen Node liest, aktualisiert `updated_at`. Das verhindert, dass häufig abgefragtes Wissen veraltet. Implementierung über einen Wrapper:

```bash
# ctx-query: Query + Touch
function ctx-query
  set result (sqlitegraph --db ~/context.db query "$argv[1]")
  # Touch alle gelesenen Nodes
  echo $result | jq -r '.[].id' | while read id
    sqlitegraph --db ~/context.db query "
      MATCH (n) WHERE id(n) = $id
      SET n.data.updated_at = datetime('now')
    "
  end
  echo $result
end
```

---

## 3. Query-Patterns

### 3.1 Context-Load für Agent-Session

Das ist der kritischste Query — er läuft VOR jeder Agent-Session und bestimmt was der Agent "weiß". Das Context-Budget ist begrenzt (typisch: 4000–8000 Tokens im System-Prompt), also muss der Load priorisieren.

**Dreistufiger Context-Load:**

```
Stufe 1: Projekt-State (immer, ~500 Tokens)
  → Aktuelles Projekt, Version, offene Bugs, letzte 3 Sessions

Stufe 2: Relevante Learnings + Decisions (~1500 Tokens)
  → Semantische Suche mit Task-Beschreibung als Query
  → Alle "pinned" Learnings des Projekts
  → Aktive Decisions die zum Task passen

Stufe 3: Artefakt-Kontext (~1000 Tokens)
  → Zuletzt geänderte Dateien und ihre Benchmark-Werte
  → Offene Bugs an den betroffenen Komponenten
```

**Stufe 1 — Projekt-Snapshot:**

```cypher
MATCH (p:Project {name: "vulkanforge"})
RETURN p.data.current_version, p.data.tech_stack

MATCH (p:Project {name: "vulkanforge"})-[:CONTAINS]->(sp:Sprint)
WHERE sp.data.status = "active"
RETURN sp.name, sp.data.summary

MATCH (p:Project {name: "vulkanforge"})-[:CONTAINS]->(:Sprint)-[:CONTAINS]->(s:Session)
WHERE s.data.status = "active"
RETURN s.name, s.data.summary, s.data.created_at
ORDER BY s.data.created_at DESC LIMIT 3

MATCH (b:Bug)-[:TRIGGERED_BY]->()-[:PRODUCED]->()<-[:CONTAINS]-(:Project {name: "vulkanforge"})
WHERE b.data.status = "open"
RETURN b.name, b.data.summary
```

**Stufe 2 — Semantische Relevanz (HNSW):**

```bash
# Task-Beschreibung als Embedding → HNSW-Suche
sqlitegraph --db ~/context.db vector-search \
  --index learnings_idx \
  --query-text "MoE expert dispatch optimization RDNA4" \
  --top-k 10 \
  --metric cosine

# Ergänzend: Pinned Learnings
sqlitegraph --db ~/context.db query '
  MATCH (l:Learning)-[:TAGGED]->(t:Tag)
  WHERE l.data.status = "pinned"
    AND t.name IN ["rdna4", "moe", "performance"]
  RETURN l.name, l.data.claim, l.data.category
'
```

**Stufe 3 — Artefakt-Lokus:**

```cypher
MATCH (a:Artifact)
WHERE a.data.artifact_type = "code"
  AND a.data.file_path CONTAINS "dispatch"
  AND a.data.status = "active"
RETURN a.name, a.data.file_path, a.data.metrics
ORDER BY a.data.updated_at DESC LIMIT 5
```

### 3.2 Post-Session Wissens-Extraktion

Am Ende einer Session extrahiert der Agent die wichtigsten Punkte:

```cypher
-- Neue Erkenntnis speichern
CREATE (l:Learning {
  name: "fma-race-1loc-barrier",
  data: {
    "category": "positive",
    "claim": "FMA Race in MoE Expert Dispatch war der Bug — 1 LOC Barrier Fix",
    "evidence": "Sprint 61G: Q8_1 noise hypothesis war WRONG, isolation test bewies es",
    "applies_to": ["moe", "vulkan", "synchronization"],
    "confidence": 1.0,
    "created_at": "2026-05-16T18:00:00Z",
    "status": "active"
  }
})

-- Verknüpfung zur Session
MATCH (s:Session {name: "sprint-61g"})
CREATE (s)-[:PRODUCED]->(l)

-- Verknüpfung zu existierender falscher Hypothese
MATCH (old:Learning {name: "q8-1-noise-hypothesis"})
CREATE (l)-[:CONTRADICTS {
  data: {"context": "Isolation test top_k=1 bewies FMA Race, nicht Q8_1 Noise"}
}]->(old)
SET old.data.status = "superseded"
```

### 3.3 Cross-Projekt-Queries

Erkenntnisse aus einem Workflow in einem anderen finden:

```cypher
-- "Gibt es Learnings aus der Musik-Produktion über Iteration,
--  die für meinen Software-Workflow relevant sein könnten?"
MATCH (l:Learning)-[:TAGGED]->(t:Tag {name: "iteration"})
MATCH (l)<-[:PRODUCED]-(:Session)<-[:CONTAINS]-(:Sprint)<-[:CONTAINS]-(p:Project)
WHERE p.data.domain = "music"
  AND l.data.status IN ["active", "pinned"]
RETURN l.name, l.data.claim, p.name
```

Oder über Semantische Suche domänenübergreifend:

```bash
# Suche über ALLE Projekte nach "diminishing returns bei Iteration"
sqlitegraph --db ~/context.db vector-search \
  --index learnings_idx \
  --query-text "diminishing returns iteration refinement" \
  --top-k 5 \
  --metric cosine
```

### 3.4 Iterations-Tracking

```cypher
-- Vollständige Versions-Kette eines Artefakts
MATCH path = (latest:Artifact {name: "cyberpunk-city-v5"})
  -[:ITERATION_OF*]->(first:Artifact)
RETURN path

-- Was hat sich zwischen Versionen geändert?
MATCH (v3:Artifact)-[:ITERATION_OF]->(v2:Artifact)
MATCH (v3)-[:GENERATED_BY]->(p3:Prompt)
MATCH (v2)-[:GENERATED_BY]->(p2:Prompt)
WHERE v3.name = "cyberpunk-city-v3"
RETURN p2.data.text AS previous_prompt,
       p3.data.text AS current_prompt,
       v2.data.metrics AS previous_metrics,
       v3.data.metrics AS current_metrics
```

### 3.5 Zeitreise-Queries

```cypher
-- "Was wussten wir am 01.05.2026 über FP8?"
MATCH (l:Learning)-[:TAGGED]->(t:Tag {name: "fp8"})
WHERE l.data.created_at < "2026-05-01T00:00:00Z"
  AND l.data.status != "archived"
RETURN l.name, l.data.claim, l.data.category
ORDER BY l.data.created_at DESC

-- "Welche Entscheidungen haben wir seitdem revidiert?"
MATCH (new:Decision)-[:SUPERSEDES]->(old:Decision)
WHERE old.data.created_at < "2026-05-01"
  AND new.data.created_at >= "2026-05-01"
RETURN old.name AS original, new.name AS revision,
       new.data.rationale
```

---

## 4. Integration-Pattern

### 4.1 Architektur-Überblick

```
┌─────────────────────────────────────────────────────────────┐
│                      context.db (SQLite)                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  │
│  │  Nodes   │  │  Edges   │  │ HNSW Index │  │ KV-Store │  │
│  └──────────┘  └──────────┘  └────────────┘  └──────────┘  │
└────────────┬────────────────────────┬───────────────────────┘
             │ CLI                    │ CLI
     ┌───────┴────────┐      ┌───────┴────────┐
     │  ctx-wrapper    │      │  ctx-wrapper    │
     │  (fish/bash)    │      │  (fish/bash)    │
     └───────┬────────┘      └───────┬────────┘
             │                       │
     ┌───────┴────────┐      ┌───────┴────────┐
     │   OpenCode     │      │  flux-gen /     │
     │   (Coding-     │      │  udio-gen /     │
     │    Agent)      │      │  runway-gen     │
     └───────┬────────┘      └────────────────┘
             │ OpenAI API
     ┌───────┴────────┐
     │  VulkanForge   │
     │  (LLM Backend) │
     └────────────────┘
```

**Kein Server-Prozess.** SQLiteGraph im CLI-Modus öffnet die DB, führt den Query aus, schließt die DB. WAL-Mode erlaubt concurrent reads. Das passt zu Agents die Shell-Commands ausführen.

### 4.2 CLI-Wrapper-Suite (`ctx-*`)

Ein Set von fish-Shell-Funktionen die als Thin-Wrapper um `sqlitegraph` dienen:

```fish
# ~/.config/fish/functions/ctx-init.fish
function ctx-init
  set -gx CTX_DB ~/context.db
  
  # Indizes erstellen falls nicht vorhanden
  sqlitegraph --db $CTX_DB query '
    CREATE INDEX IF NOT EXISTS idx_node_kind ON nodes(kind);
    CREATE INDEX IF NOT EXISTS idx_node_status ON nodes(data->"status");
    CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(type);
  '
end
```

```fish
# ctx-load: Context für eine neue Session laden
function ctx-load
  set project $argv[1]
  set task_description $argv[2]
  
  echo "=== PROJECT STATE ==="
  sqlitegraph --db $CTX_DB query "
    MATCH (p:Project {name: '$project'})
    RETURN p.data
  "
  
  echo "=== RECENT SESSIONS ==="
  sqlitegraph --db $CTX_DB query "
    MATCH (:Project {name: '$project'})-[:CONTAINS]->(:Sprint)-[:CONTAINS]->(s:Session)
    RETURN s.name, s.data.summary, s.data.created_at
    ORDER BY s.data.created_at DESC LIMIT 5
  "
  
  echo "=== RELEVANT KNOWLEDGE ==="
  sqlitegraph --db $CTX_DB vector-search \
    --index knowledge_idx \
    --query-text "$task_description" \
    --top-k 8 \
    --metric cosine
  
  echo "=== OPEN BUGS ==="
  sqlitegraph --db $CTX_DB query "
    MATCH (b:Bug)
    WHERE b.data.status = 'open'
    RETURN b.name, b.data.summary
  "
  
  echo "=== PINNED LEARNINGS ==="
  sqlitegraph --db $CTX_DB query "
    MATCH (l:Learning)
    WHERE l.data.status = 'pinned'
    RETURN l.name, l.data.claim
  "
end
```

```fish
# ctx-commit: Session-Ergebnisse speichern
function ctx-commit
  set session_name $argv[1]
  set summary $argv[2]
  set sprint $argv[3]
  
  sqlitegraph --db $CTX_DB query "
    MATCH (sp:Sprint {name: '$sprint'})
    CREATE (s:Session {
      name: '$session_name',
      data: {
        \"summary\": \"$summary\",
        \"created_at\": \"$(date -Iseconds)\",
        \"status\": \"active\"
      }
    })
    CREATE (sp)-[:CONTAINS]->(s)
  "
  echo "Session '$session_name' committed."
end
```

```fish
# ctx-decide: Entscheidung dokumentieren
function ctx-decide
  set name $argv[1]
  set choice $argv[2]
  set rationale $argv[3]
  
  sqlitegraph --db $CTX_DB query "
    CREATE (d:Decision {
      name: '$name',
      data: {
        \"choice\": \"$choice\",
        \"rationale\": \"$rationale\",
        \"created_at\": \"$(date -Iseconds)\",
        \"status\": \"active\",
        \"impact\": \"medium\"
      }
    })
  "
end
```

```fish
# ctx-bench: Benchmark speichern
function ctx-bench
  set name $argv[1]
  set metric $argv[2]
  set value $argv[3]
  set model $argv[4]
  
  sqlitegraph --db $CTX_DB query "
    CREATE (b:Benchmark {
      name: '$name',
      data: {
        \"target\": \"$model\",
        \"metric\": \"$metric\",
        \"value\": $value,
        \"created_at\": \"$(date -Iseconds)\",
        \"environment\": {
          \"gpu\": \"RX 9070 XT\",
          \"driver\": \"RADV Mesa 26.1\"
        }
      }
    })
  "
end
```

### 4.3 OpenCode-Integration

OpenCode als Coding-Agent bekommt den Context-Store über sein System-Prompt:

```
[System-Prompt Suffix für OpenCode]

Du hast Zugriff auf einen persistenten Wissens-Graph via Shell.
Verfügbare Befehle:
  ctx-load <project> <task>     # Kontext laden (IMMER am Session-Start)
  ctx-commit <name> <summary> <sprint>  # Session speichern (IMMER am Ende)
  ctx-learn <claim> <category>  # Erkenntnis speichern
  ctx-decide <name> <choice> <rationale>  # Entscheidung dokumentieren
  ctx-bench <name> <metric> <value> <model>  # Benchmark speichern
  ctx-bug <name> <description>  # Bug erfassen
  ctx-search <query>            # Semantische Suche im Wissens-Graph

Regeln:
1. IMMER ctx-load am Session-Start aufrufen.
2. IMMER ctx-commit am Session-Ende aufrufen.
3. Jede nicht-triviale Erkenntnis mit ctx-learn speichern.
4. Jede Architektur-Entscheidung mit ctx-decide speichern.
5. Jede Performance-Messung mit ctx-bench speichern.
```

### 4.4 Automatisierung via Git-Hooks

```bash
#!/bin/bash
# .git/hooks/post-commit
# Automatisch Commits im Context-Store tracken

HASH=$(git rev-parse HEAD)
MSG=$(git log -1 --format='%s')
FILES=$(git diff-tree --no-commit-id --name-only -r HEAD | tr '\n' ',')

sqlitegraph --db ~/context.db query "
  CREATE (a:Artifact {
    name: '$HASH',
    data: {
      \"artifact_type\": \"commit\",
      \"message\": \"$MSG\",
      \"files\": \"$FILES\",
      \"created_at\": \"$(date -Iseconds)\"
    }
  })
"
```

---

## 5. Skalierung

### 5.1 Erwartete Volumina

```
Zeithorizont    Sessions    Nodes       Edges       DB-Größe
──────────────────────────────────────────────────────────────
3 Monate        ~200        ~2.000      ~5.000      ~10 MB
1 Jahr          ~800        ~8.000      ~20.000     ~50 MB
3 Jahre         ~2.500      ~25.000     ~60.000     ~200 MB
```

Bei diesen Volumina ist SQLite komfortabel — die Grenze liegt erfahrungsgemäß bei Millionen von Rows, nicht bei Zehntausenden.

### 5.2 Indizes

Zwingend von Anfang an:

```sql
-- Node-Lookup by kind (jeder Context-Load filtert nach kind)
CREATE INDEX idx_node_kind ON nodes(kind);

-- Temporal Queries (Aging, Zeitreise, "letzte N Sessions")
CREATE INDEX idx_node_created ON nodes(json_extract(data, '$.created_at'));

-- Status-Filter (active/stale/archived/pinned)
CREATE INDEX idx_node_status ON nodes(json_extract(data, '$.status'));

-- Edge-Typ (CONTAINS, PRODUCED, etc.)
CREATE INDEX idx_edge_type ON edges(type);

-- Compound: kind + status (häufigstes Pattern)
CREATE INDEX idx_kind_status ON nodes(kind, json_extract(data, '$.status'));
```

### 5.3 HNSW-Vektor-Indizes

Separate Indizes pro semantischer Domäne, weil die Embedding-Qualität besser ist wenn die Vektoren domänenkohärent sind:

```
knowledge_idx    Learning + Decision + Concept     ~5.000 Vektoren
artifact_idx     Artifact + Prompt                  ~10.000 Vektoren
interaction_idx  Interaction                        ~2.000 Vektoren
```

HNSW-Parameter für diese Größenordnung: `M=16, ef_construction=200, ef_search=64`. Bei <50k Vektoren ist der Recall bei diesen Settings >99%.

### 5.4 Archivierung und Pruning

**Automatisches Archivieren** (monatlicher Cron):

```bash
# Alte stale Nodes → archived
sqlitegraph --db ~/context.db query '
  MATCH (n)
  WHERE n.data.status = "stale"
    AND n.data.updated_at < datetime("now", "-180 days")
  SET n.data.status = "archived"
'

# Archived Nodes aus HNSW-Indizes entfernen (spart RAM)
sqlitegraph --db ~/context.db vector-remove \
  --index knowledge_idx \
  --filter 'status = "archived"'
```

**Snapshot vor Pruning:**

```bash
# Wöchentliches Backup
sqlitegraph --db ~/context.db snapshot ~/backups/context-$(date +%Y%m%d).db
```

**Hard-Delete:** Nur manuell, nie automatisch. Archivierte Nodes bleiben für Zeitreise-Queries erreichbar, kosten aber keinen HNSW-Suchaufwand.

### 5.5 Denormalisierung für Performance

Bei >10.000 Nodes werden Multi-Hop-Traversals spürbar. Lösung: Denormalisierte Aggregate als KV-Store-Einträge (SQLiteGraph KV-Store mit TTL):

```bash
# Tägliches Aggregate-Update
sqlitegraph --db ~/context.db kv-set \
  "project:vulkanforge:stats" \
  '{
    "total_sessions": 342,
    "total_learnings": 89,
    "total_bugs_open": 3,
    "decode_tok_s_best": 121,
    "last_session": "2026-05-19",
    "active_sprint": "sprint-62"
  }' \
  --ttl 86400  # 24h, wird täglich refresht
```

Damit ist der Projekt-Snapshot ein einziger KV-Lookup statt 4 Graph-Queries.

---

## 6. Semantische Suche

### 6.1 Embedding-Strategie

**Modell-Wahl:** Das Embedding-Modell sollte lokal laufen (Offline-Anforderung). Optionen, in absteigender Präferenz:

1. **VulkanForge selbst** — Qwen3-8B kann Embeddings extrahieren (letzte Hidden-State, mean-pooled). Vorteil: Keine zusätzliche Dependency. Nachteil: 8B ist für Embeddings oversized, ~50ms pro Embedding.

2. **Dediziertes kleines Modell** — Ein GGUF-quantisiertes Embedding-Modell (z.B. `nomic-embed-text-v1.5` Q8_0, ~260MB). Über VulkanForge als Backend oder über llama.cpp. Vorteil: Schneller (~5ms), bessere Embedding-Qualität da speziell trainiert. Nachteil: Zusätzliches Modell.

3. **CPU-only** — `all-MiniLM-L6-v2` via Python `sentence-transformers` oder Rust `candle`. ~384 Dimensionen, ~80MB, ~2ms auf Zen4. Vorteil: Kein GPU nötig, winzig. Nachteil: Englisch-only, kleinere Dimensionalität.

**Empfehlung:** Option 3 für den Start (CPU, MiniLM). Option 2 als Upgrade wenn die Embedding-Qualität zum Bottleneck wird. VulkanForge als Embedding-Backend ergibt sich organisch wenn der API-Server steht.

### 6.2 Was bekommt ein Embedding?

Nicht jeder Node braucht eins. Die Regel: **Wenn der Node textuellen Inhalt hat der semantisch gesucht werden soll, bekommt er ein Embedding.**

```
Node-Kind       Embedding-Quelle                        Index
─────────────────────────────────────────────────────────────
Learning        claim + evidence                        knowledge_idx
Decision        choice + rationale                      knowledge_idx
Concept         name + summary                          knowledge_idx
Prompt          text (gekürzt auf 512 Tokens)           artifact_idx
Artifact        summary + file_path                     artifact_idx
Bug             name + summary                          knowledge_idx
Interaction     summary                                 interaction_idx
Benchmark       target + metric + summary               knowledge_idx
──
Project         —  (Graph-Traversal reicht)
Sprint          —
Session         —
Agent           —
Model           —
Tag             —  (exakte Match reicht)
Config          —
Document        —  (zu lang für ein Embedding, Chunking
                    nötig → eigenes Sub-System)
```

### 6.3 Embedding-Pipeline

```bash
# embed-node: Embedding für einen Node berechnen und speichern
function embed-node
  set node_id $argv[1]
  
  # 1. Text extrahieren
  set text (sqlitegraph --db $CTX_DB query "
    MATCH (n) WHERE id(n) = $node_id
    RETURN n.data.claim, n.data.summary, n.name
  " | jq -r 'map(values) | flatten | join(" ")')
  
  # 2. Embedding berechnen (MiniLM via Python)
  set embedding (python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
e = m.encode('$text')
print(','.join(map(str, e)))
  ")
  
  # 3. In HNSW-Index speichern
  sqlitegraph --db $CTX_DB vector-add \
    --index knowledge_idx \
    --id $node_id \
    --vector "$embedding"
end
```

**Batch-Embedding (für Import):**

```python
# embed_batch.py — Alle un-embedded Nodes embedden
import subprocess, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Alle Nodes ohne Embedding holen
result = subprocess.run([
    'sqlitegraph', '--db', 'context.db', 'query',
    'MATCH (n) WHERE n.kind IN ["Learning","Decision","Bug","Concept"] '
    'RETURN id(n), n.name, n.data.summary, n.data.claim'
], capture_output=True, text=True)

nodes = json.loads(result.stdout)
texts = [f"{n.get('name','')} {n.get('summary','')} {n.get('claim','')}" for n in nodes]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

for node, emb in zip(nodes, embeddings):
    vec_str = ','.join(map(str, emb.tolist()))
    subprocess.run([
        'sqlitegraph', '--db', 'context.db', 'vector-add',
        '--index', 'knowledge_idx',
        '--id', str(node['id']),
        '--vector', vec_str
    ])
```

### 6.4 Wann Vektor-Suche vs. Graph-Query

Die Entscheidung ist nicht entweder/oder — oft kombiniert man beides:

```
Szenario                              Tool            Warum
────────────────────────────────────────────────────────────────
"Letzte 5 Sessions"                   Graph           Temporal, exakt
"Alle offenen Bugs"                   Graph           Status-Filter, exakt
"Was wissen wir über FP8 WMMA?"       Vektor          Semantisch offen
"Entscheidungen zu MoE Dispatch"      Vektor+Graph    Vektor findet Relevanz,
                                                      Graph findet Kausalketten
"Erkenntnisse aus Musik für Video"    Vektor          Cross-Domain, semantisch
"Bug #42 Kontext"                     Graph           Exakter Lookup + Traversal
"Ähnliche Prompts wie dieser"         Vektor          Nearest-Neighbor auf Prompts
```

**Hybrid-Pattern (häufigstes Muster):**

```bash
# 1. Vektor-Suche: Semantisch relevante Nodes finden
RELEVANT=$(sqlitegraph --db $CTX_DB vector-search \
  --index knowledge_idx \
  --query-text "cooperative matrix prefill optimization" \
  --top-k 5)

# 2. Graph-Traversal: Kontext dieser Nodes anreichern
for id in $(echo $RELEVANT | jq -r '.[].id'); do
  sqlitegraph --db $CTX_DB query "
    MATCH (n)-[e]-(neighbor)
    WHERE id(n) = $id
    RETURN n.name, type(e), neighbor.name, neighbor.kind
  "
done
```

Das ergibt: Vektor-Suche als "Einstiegspunkt", Graph-Traversal als "Kontextanreicherung". Die Vektor-Suche sagt "diese 5 Nodes sind relevant", der Graph sagt "und hier sind ihre Entscheidungen, Bugs, und Sessions".

---

## 7. Workflow-spezifische Patterns

### 7.1 Software-Entwicklung (VulkanForge)

```
Project("vulkanforge")
  └─ Sprint("sprint-61") ── status: "active"
       └─ Session("sprint-61g") ── agent: "opencode", model: "qwen3-8b"
            ├─ PRODUCED → Decision("expert-grouped-dispatch")
            ├─ PRODUCED → Learning("fma-race-barrier-fix")
            ├─ PRODUCED → Benchmark("26b-decode-27tps")
            ├─ PRODUCED → Artifact("executor/moe.rs")
            └─ PRODUCED → Bug("fma-race") ── status: "fixed"
                              └─ FIXED_IN → Artifact("executor/moe.rs")
```

**Sprint-Report-Query:**
```cypher
MATCH (sp:Sprint {name: "sprint-61"})-[:CONTAINS]->(s:Session)
MATCH (s)-[:PRODUCED]->(output)
RETURN s.name, collect(output.kind) AS types,
       collect(output.name) AS items
ORDER BY s.data.created_at
```

### 7.2 Bild-Generierung

```
Project("cyberpunk-series")
  └─ Sprint("batch-2026-05")
       └─ Session("flux-session-42")
            ├─ USED_MODEL → Model("flux-dev")
            ├─ PRODUCED → Artifact("city-v1.png")
            │    └─ GENERATED_BY → Prompt("cyberpunk city golden hour...")
            │         └─ USED_CONFIG → Config("flux-default-30steps")
            └─ PRODUCED → Artifact("city-v2.png")
                 ├─ ITERATION_OF → Artifact("city-v1.png")
                 └─ GENERATED_BY → Prompt("cyberpunk city, more neon...")
```

**Style-Evolution-Query:**
```cypher
MATCH (a:Artifact)-[:GENERATED_BY]->(p:Prompt)-[:USED_CONFIG]->(c:Config)
WHERE a.data.artifact_type = "image"
  AND a.data.result_quality = "good"
RETURN p.data.text, c.data, a.data.created_at
ORDER BY a.data.created_at
-- → Zeigt wie sich erfolgreiche Prompts über Zeit entwickeln
```

### 7.3 Musik-Produktion

```
Project("synthwave-album")
  └─ Sprint("track-3-production")
       └─ Session("udio-session-7")
            ├─ USED_MODEL → Model("udio-v3")
            ├─ PRODUCED → Artifact("track3-v1.mp3")
            │    ├─ GENERATED_BY → Prompt("synthwave, 128bpm, melancholic...")
            │    └─ TAGGED → Tag("synthwave"), Tag("melancholic")
            └─ PRODUCED → Artifact("track3-v2.mp3")
                 └─ ITERATION_OF → Artifact("track3-v1.mp3")
```

### 7.4 Cross-Pollination Beispiel

Eine Erkenntnis aus der Bild-Generierung könnte den Software-Workflow beeinflussen:

```cypher
-- Bild-Workflow-Erkenntnis: "Iteration nach 3 Versionen liefert
-- diminishing returns — besser neuer Ansatz als v4+"
MATCH (l:Learning {name: "iteration-diminishing-returns"})

-- Verbindung zu Software-Entscheidung:
-- "Nach 3 Optimierungsversuchen an einem Kernel → neuer Algorithmus"
MATCH (d:Decision {name: "kernel-rewrite-after-3-attempts"})
CREATE (l)-[:CROSS_POLLINATED {
  data: {
    "context": "Gleiche Beobachtung: nach 3 Iterationen sinkt ROI",
    "created_at": "2026-05-19"
  }
}]->(d)
```

---

## 8. Bootstrap-Sequenz

Schrittweiser Aufbau, kein Big-Bang:

### Phase 1: Fundament (Tag 1)

```bash
# 1. DB erstellen
sqlitegraph --db ~/context.db init

# 2. Basis-Indizes
sqlitegraph --db ~/context.db query '
  CREATE INDEX idx_node_kind ON nodes(kind);
  CREATE INDEX idx_node_status ON nodes(json_extract(data, "$.status"));
  CREATE INDEX idx_edge_type ON edges(type);
'

# 3. HNSW-Index
sqlitegraph --db ~/context.db vector-create \
  --index knowledge_idx \
  --dimensions 384 \
  --metric cosine \
  --m 16 \
  --ef-construction 200

# 4. Projekte anlegen
sqlitegraph --db ~/context.db query '
  CREATE (p:Project {
    name: "vulkanforge",
    data: {
      "domain": "software",
      "repo": "https://github.com/oldnordic/vulkanforge",
      "current_version": "v0.4.4",
      "tech_stack": ["rust", "vulkan", "glsl"],
      "created_at": "2026-04-25",
      "status": "active"
    }
  })
'

# 5. fish-Functions installieren
cp ctx-*.fish ~/.config/fish/functions/
```

### Phase 2: Backfill (Tag 2-3)

Existierende Erkenntnisse aus Memory/Sprint-Reports importieren:

```bash
# Pinned Learnings aus den bisherigen 60+ Sprints
ctx-learn "Dispatch-Overhead ist NICHT der Bottleneck auf RDNA4" "positive" --pin
ctx-learn "IMMER messen statt schätzen, Cost-Modelle 30× daneben" "positive" --pin
ctx-learn "Wave-Parallelism versteckt Serial-Bottlenecks" "positive" --pin
ctx-learn "FP8 auf Q4_K bringt nichts (Dequant-Overhead)" "negative" --pin
ctx-learn "GEMV×GEMV Fusion killt BW, Elementwise-Epilog frei" "negative" --pin
# ... (die ~30 wichtigsten aus dem Memory-System extrahieren)

# Benchmark-Baselines
ctx-bench "v044-qwen3-decode" "decode_tok_s" 121 "Qwen3-8B Q4_K_M" --pin
ctx-bench "v044-llama-decode" "decode_tok_s" 114 "llama.cpp Vulkan"  --pin
```

### Phase 3: Live-Integration (Tag 4+)

OpenCode System-Prompt erweitern, Git-Hooks installieren, Aging-Cron einrichten.

---

## 9. Offene Entscheidungen

1. **Embedding-Modell:** MiniLM (CPU, sofort) vs. Nomic-Embed (GPU, besser) vs. VulkanForge-Embedding (kein Extra-Modell, aber 50ms). Empfehlung: MiniLM jetzt, Upgrade später.

2. **Cypher-Subset-Abdeckung:** Die SQLiteGraph Cypher-Implementierung deckt möglicherweise nicht alle oben gezeigten Patterns ab (z.B. variable-length Pfade `[:ITERATION_OF*]`, `collect()`, `datetime()`-Funktionen). Vor der Implementierung gegen die tatsächliche Query-Engine testen.

3. **Document-Chunking:** Architektur-Dokumente (2700+ Zeilen) passen nicht in ein einziges Embedding. Optionen: Chunk by Section (H2-Ebene), oder Zusammenfassung pro Dokument als Proxy. Die Section-Ebene ist präziser.

4. **Context-Budget-Optimierung:** Der dreistufige Context-Load gibt maximal ~3000 Tokens aus. Reicht das? Iterativ anpassen — der KV-Store-Cache kann Aggregate vorhalten die den Load beschleunigen.

5. **VulkanForge v0.5 Kopplung:** SQLiteGraph als Dependency in VulkanForge selbst einbinden (für Model Introspection Daten, Benchmark-History, Bandit-Kalibrierungen) oder als externes Tool belassen? Empfehlung: Extern starten, bei Bedarf als Library einbinden.
