# vf-clide

Ein schlanker Kommandozeilen-Client für VulkanForge. vf-clide spricht **ausschließlich über die OpenAI-kompatible
API** mit dem laufenden VF-Server (keine internen VF-Abhängigkeiten). Stand v0.2.0 (ausgeliefert mit VF **v0.9.0**)
ist es ein **Chat- *und* agentischer Coding-Client**: im `--agent`-Modus nutzt es Tools (lesen, schreiben, suchen,
shell) in einem Loop, mit einem **gestuften Permission-Modell** und **Workspace-Confinement**.

Du brauchst **zwei Terminals**: eins für den Server (läuft die ganze Zeit), eins für vf-clide.

> **fish-Stolperstein:** Umgebungsvariablen setzt man in fish **nicht** wie in bash. `VAR=1 befehl` funktioniert
> **nicht** — es muss `env VAR=1 befehl` heißen. Betrifft unten das `VULKANFORGE_KV_FP8=1` für gemma.

---

## 1. Installation (einmalig)

```fish
# Engine + vf-clide bauen
cargo build --release                                              # → ./target/release/vulkanforge
cargo build --release --manifest-path vf-clide/Cargo.toml          # → ./vf-clide/target/release/vf-clide

# als globale Befehle verfügbar machen (Symlinks in ~/.local/bin)
ln -s /home/maeddes/projects/vulkanforge/target/release/vulkanforge        ~/.local/bin/vulkanforge
ln -s /home/maeddes/projects/vulkanforge/vf-clide/target/release/vf-clide  ~/.local/bin/vf-clide
```

Danach sind `vulkanforge` und `vf-clide` von überall aufrufbar; die Symlinks überleben Rebuilds. Welche Modelle
du hast: `ls ~/models/`.

---

## 2. Terminal 1 — den Server starten

Der Server lädt **ein** Modell und bleibt im Vordergrund laufen. **Dieses Terminal offen lassen.**

**Variante A — Chat/Coding-Modell (empfohlen für `--agent`):**
```fish
vulkanforge serve --model ~/models/Qwen_Qwen3-14B-Q4_K_M.gguf --port 8080 --cors
```

**Variante B — gemma-QAT (Qualitätsmodell, braucht KV-FP8):**
```fish
env VULKANFORGE_KV_FP8=1 vulkanforge serve \
  --model ~/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --port 8080 --cors
```

> **Kontextgröße ist automatisch (ab v0.8.0):** Der Server errechnet sie selbst aus freiem VRAM + Modell und
> druckt beim Start, was er gewählt hat. Du musst **nichts** setzen. Für einen festen Wert geht `--ctx-size <N>`
> als Override. Harte Grenze auf dieser GPU: **16384** Tokens — die gilt **auch für `--ctx-size`**: höhere
> Werte werden *nicht* still gekappt, sondern führen beim Start zum Abbruch (die Pipeline-Erstellung lehnt das
> LDS-Limit ab). Auto-ctx bleibt von sich aus darunter.

**Erst bei der `listening`-Zeile** ist er bereit — vorher: „Connection refused".

> `--cors` braucht nur ein Browser-Client (Open WebUI); für vf-clide/curl egal. Alle Flags: `vulkanforge serve --help`.

---

## 3. Terminal 2 — vf-clide starten

### Chat (Default)

**Interaktiv (REPL):**
```fish
vf-clide --url http://localhost:8080
```
Eingabe tippen, Enter, die Antwort streamt live. REPL-Befehle: `/clear`, `/model <name>`, `/max-tokens <N>`,
`/think` · `/no-think`, `/quit` (`/q`, `/exit`).

**Einmalige Frage (headless):**
```fish
vf-clide --url http://localhost:8080 -p "Hauptstadt von Japan? Ein Wort."
```

### Agent (`--agent`) — Tool-gestütztes Coden

Im Agent-Modus darf das Modell **Tools** aufrufen; vf-clide führt den Roundtrip (Modell → Permission → Tool →
Ergebnis zurück → weiter) bis zu **8 Iterationen**. Default-Coder = **Qwen3-14B-Q4** (JSON-Tool-Argumente).

```fish
# headless, nur lesende Tools auto-erlaubt:
vf-clide --url http://localhost:8080 --agent --yes --workspace ~/code/myproj \
  -p "Suche nach 'TODO' und fasse die offenen Punkte zusammen."

# REPL im Agent-Modus (fragt pro Tool-Call interaktiv y/N):
vf-clide --url http://localhost:8080 --agent --workspace ~/code/myproj
```

**Die 4 Tools:**

| Tool | Tut | Risiko-Stufe | Confined? |
|---|---|---|---|
| `read_file` | Datei lesen (256 KB Cap) | ReadOnly | ja (Workspace) |
| `search` | Substring-Suche, `file:line`-Treffer (Cap 100/64 KB) | ReadOnly | ja (Workspace) |
| `write_file` | Datei anlegen/überschreiben (Parent-Dirs in-root) | Mutating | ja (Workspace) |
| `shell` | Shell-Kommando (cwd = Workspace, Output-Cap 256 KB, Timeout 30 s) | Exec | **nein** (s. u.) |

**Permission-Modell (3 Stufen, kumulativ):** Jedes Tool hat eine Risiko-Stufe. Headless steigt die Auto-Freigabe
**opt-in** und **kumulativ** — eine höhere Stufe schließt die niedrigeren ein:

| Flag | gibt auto frei |
|---|---|
| (kein Flag) | nichts — jeder Tool-Call wird abgelehnt |
| `--yes` | **ReadOnly** (`read_file`, `search`) |
| `--allow-mutating` | ReadOnly **+ Mutating** (`write_file`) — impliziert `--yes` |
| `--allow-shell` | ReadOnly + Mutating **+ Exec** (`shell`) — impliziert `--allow-mutating` |

`--yes` allein gibt also **nie** `write_file` oder `shell` frei. Im **REPL** wird stattdessen **pro Call
interaktiv** mit `y/N` bestätigt (mutierende/ausführende Tools mit deutlichem Warnhinweis) — die Flags brauchst du
dort nicht.

**Workspace & Konstitution:**

| Flag | Default | Zweck |
|---|---|---|
| `--agent` | aus | Agent-Loop aktivieren (sonst reiner Chat) |
| `--workspace <path>` | aktuelles Verzeichnis | Wurzel für die Datei-Tools; einmal kanonisiert |
| `--yes` / `--allow-mutating` / `--allow-shell` | aus | Auto-Freigabe-Stufen (s. o.) |
| `--system <file>` | — | Ersetzt den eingebauten System-Prompt (Konstitution) ganz |
| `--no-system` | aus | Keinen System-Prompt senden |

Ohne `--system`/`--no-system` bekommt der Agent einen knappen eingebauten **System-Prompt** (Rolle, Tool-Nutzung,
Permission-Respekt). Liegt eine **`AGENTS.md`** in der Workspace-Wurzel, wird sie **angehängt** (projektspezifische
Anweisungen) — sie wird **confined** gelesen (eine `AGENTS.md`, die per Symlink aus dem Workspace zeigt, wird
ignoriert).

**Gemeinsame Flags (Chat + Agent):**

| Flag | Default | Zweck |
|---|---|---|
| `-p`, `--prompt` | — | Einmalige Frage (headless statt REPL) |
| `--url` | `http://localhost:8080` | Server-Adresse |
| `--model` | `Qwen3-14B-Q4_K_M` | nur Label im Request (s. Einschränkungen) |
| `--max-tokens` | `6144` | Token-Budget; großzügig für Thinking-Modelle |
| `--no-think` | aus | hängt `/no_think` an → Antwort ohne Denk-Block |
| `--no-stream` | aus | komplette Antwort statt Streaming (nur Chat-Headless) |
| `--temperature` | `0.0` | Sampling-Temperatur |

---

## 4. Was geht
- **Chat** mit/ohne Streaming, **Multi-Turn** mit Verlauf innerhalb der Session.
- **Agent-Loop** (`--agent`): die 4 Tools im Roundtrip, gegated, mit Loop-Cap 8.
- **Workspace-Confinement** der Datei-Tools: `read_file`/`write_file`/`search` dürfen nicht aus der Wurzel
  hinaus (`../` und rausführende Symlinks werden abgewiesen — strukturierter Fehler ans Modell, kein Crash).
- **Sichtbare Marker statt stummer Fehler:** Token-Limit → `[truncated …]`; Thinking-Modell nur Denk-Block →
  `[empty answer …]`. Beide auf stderr, stdout bleibt sauber. Permission-Entscheidungen werden auf stderr geloggt.
- Validiert über gemma (QAT/Q3, @KV-FP8), Qwen3 (14B/8B), Llama-3.1-8B, Mistral-7B, DeepSeek-R1-Distill (Chat);
  Agent-Pfade @Qwen3-14B-Q4.

---

## 5. Einschränkungen (bitte lesen)

**Agent / Tools:**
- **`shell` ist NICHT confined.** cwd ist die Workspace-Wurzel, aber ein Kommando kann den Workspace verlassen
  (`cat ~/.ssh/id_rsa` ignoriert cwd). Der Schutz für `shell` ist **nicht** das Confinement, sondern die
  **Exec-Stufe**: `--allow-shell` ist die bewusste, laut benannte Opt-in-Stufe (bzw. interaktives `y` im REPL).
  Bewusst einsetzen.
- **Keine Persistenz / kein Gedächtnis.** Der Agent-Loop und der Verlauf leben nur in der Session; nichts wird auf
  Platte geschrieben. (Der Memory-Seam ist vorbereitet, aber leer.)
- **`search` ist substring-basiert** (kein Regex); `.git`/`target`/`node_modules`/… werden übersprungen; Treffer
  auf 100 / 64 KB gekappt.
- **gemma:** Tool-Calling ist für einfache Argumente validiert; code-tragende Argumente folgen. Default-Coder ist
  **Qwen3-14B-Q4**.

**Modell & Kontext:**
- **Ein Modell pro Server.** `--model` ist nur ein Label — *welches* Modell antwortet, bestimmt allein das, was
  der Server geladen hat. Modellwechsel = Server neu starten.
- **Kontext-Decke 16384 Tokens** (Hardware-Grenze der GPU, serverseitig). Lange Sessions mit vielen
  Dateien/Verlauf stoßen daran.
- **gemma-QAT ist VRAM-eng:** auto-ctx gibt ihm nur ~2,5k Kontext im 16-GB-Budget — gut für kurze Turns, zu knapp
  für große Coding-Kontexte. Dafür ist **Qwen3-14B-Q4** (bis 16384) die bessere Wahl.

**Thinking-Modelle (Qwen3, gemma):**
- Bei langen Aufgaben kann der `<think>`-Block das Budget aufbrauchen → leere sichtbare Antwort (der Empty-Marker
  weist darauf hin). Abhilfe: `--max-tokens` höher oder `--no-think`. Default 6144 reicht für die meisten Turns.

**Bedienung:**
- Der **REPL braucht ein echtes Terminal** (TTY) — nicht per Pipe/Skript fütterbar. Fürs Skripten den
  Headless-Modus (`-p`) nutzen.

---

## 6. Troubleshooting

| Symptom | Ursache / Fix |
|---|---|
| `Connection refused` | Server lädt noch oder läuft nicht. Auf die `listening`-Zeile in Terminal 1 warten. |
| vf-clide erreicht den Server nicht | Port-Mismatch: `--port` (Terminal 1) = `--url` (Terminal 2). |
| **Tool-Call wird abgelehnt** (`[agent] DENIED …`) | Die nötige Stufe fehlt: `write_file` braucht `--allow-mutating`, `shell` braucht `--allow-shell` (headless). Im REPL pro Call `y` bestätigen. |
| **Tool-Ergebnis „outside the workspace"** | Pfad zeigt aus der `--workspace`-Wurzel hinaus (`../` oder Symlink). Im Workspace bleiben oder `--workspace` passend setzen. |
| **`shell` „TIMED OUT"** | Kommando lief > 30 s und wurde gekillt. Kürzeres/nicht-blockierendes Kommando. |
| **Antwort bricht mitten im Satz ab** | Token-Budget zu klein: `--max-tokens` höher (Default 6144). Die Kontextgröße wählt der Server selbst. |
| **Leere Antwort bei Thinking-Modell** | `<think>`-Block hat das Budget verbraucht. `--max-tokens` höher oder `--no-think`. vf-clide zeigt dazu einen Marker. |
| **Server bricht beim Laden ab mit FP8-Hinweis** | gemma-26B ohne KV-FP8: der Guard stoppt mit „…only FP8 (E4M3) KV is correct. Set `VULKANFORGE_KV_FP8=1`…". → mit `env VULKANFORGE_KV_FP8=1` neu starten. (Notüberbrückung: `VULKANFORGE_ALLOW_BROKEN_KV=1`.) |
| **Pipeline-/Shader-Fehler beim Start mit `--ctx-size`** | Override **> 16384** überschreitet das LDS-Limit der GPU → Abbruch. Wert ≤ 16384 oder `--ctx-size` weglassen (auto-ctx bleibt sicher darunter). |
| Ein Flag wird nicht erkannt | `vulkanforge serve --help` bzw. `vf-clide --help` prüfen. |
