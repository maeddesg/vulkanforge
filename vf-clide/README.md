# vf-clide

Ein schlanker Kommandozeilen-Client für VulkanForge. vf-clide spricht **ausschließlich über die OpenAI-kompatible
API** mit dem laufenden VF-Server (keine internen VF-Abhängigkeiten). Stand v0.1.0 (ausgeliefert mit VF **v0.8.0**)
ist es ein **Chat-Client**; der agentische Loop (Tools, Dateizugriff, Gedächtnis) folgt in Phase 2.

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

**Variante A — Chat-Modell (z. B. Qwen):**
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

**Was du sehen solltest** (Werte variieren je Modell):
```
VulkanForge: GPU = AMD Radeon RX 9070 XT (RADV GFX1201)
VulkanForge: auto ctx-size = 16384 (free … − weights … − reserve … ; bound: …)
Loading … GGUF tensors...
VulkanForge: VRAM budget … used / … total (… free)
VulkanForge API server listening on http://127.0.0.1:8080
```
**Erst bei der `listening`-Zeile** ist er bereit — vorher: „Connection refused".

> `--cors` braucht nur ein Browser-Client (Open WebUI); für vf-clide/curl egal. Alle Flags: `vulkanforge serve --help`.

**Schneller Lebt-er-Check** (zweites Terminal):
```fish
curl -sS http://localhost:8080/v1/models | jq
```

---

## 3. Terminal 2 — vf-clide starten

**Interaktiv (REPL):**
```fish
vf-clide --url http://localhost:8080
```
Eingabe tippen, Enter, die Antwort streamt live. REPL-Befehle:

| Befehl | Wirkung |
|---|---|
| `/clear` | Gesprächsverlauf leeren |
| `/model <name>` | Modell-Label wechseln |
| `/max-tokens <N>` | Token-Budget zur Laufzeit setzen |
| `/think` · `/no-think` | Denken (Thinking-Modelle) an/aus |
| `/quit`, `/q`, `/exit` | beenden |

**Einmalige Frage (headless):**
```fish
vf-clide --url http://localhost:8080 -p "Hauptstadt von Japan? Ein Wort."
```

**Flags:**

| Flag | Default | Zweck |
|---|---|---|
| `-p`, `--prompt` | — | Einmalige Frage stellen, Antwort ausgeben, beenden (headless statt REPL) |
| `--url` | `http://localhost:8080` | Server-Adresse |
| `--model` | `Qwen3-14B-Q4_K_M` | nur Label im Request (s. Einschränkungen) |
| `--max-tokens` | `6144` | Token-Budget; großzügig für Thinking-Modelle |
| `--no-think` | aus | hängt `/no_think` an → Antwort ohne Denk-Block |
| `--no-stream` | aus | komplette Antwort statt Streaming |
| `--temperature` | `0.0` | Sampling-Temperatur |
| `--project` | — | Projekt-Scope (Platzhalter, noch ohne Wirkung) |

---

## 4. Was geht (validiert über 7 Modelle)
- **Chat** mit und ohne Streaming, **Multi-Turn** mit Verlauf innerhalb der Session.
- **Sichtbare Marker statt stummer Fehler:** bricht eine Antwort am Token-Limit ab →
  `[truncated at the token limit (N) …]`; produziert ein Thinking-Modell nur einen Denk-Block ohne sichtbare
  Antwort → `[empty answer — the budget was likely consumed by the <think> block …]`. Beide auf stderr, stdout
  bleibt sauber.
- Funktioniert mit gemma (QAT/Q3, @KV-FP8), Qwen3 (14B/8B), Llama-3.1-8B, Mistral-7B, DeepSeek-R1-Distill.

---

## 5. Einschränkungen (bitte lesen)

**Funktionsumfang (v0.1.0):**
- **Nur Chat.** Kein Agent-Loop, keine Tools, kein Datei-/Shell-Zugriff — das ist Phase 2.
- **Gedächtnis ist ein No-op.** Der Seam ist da, aber leer; `--project` wird geparst, tut aber noch nichts.
- **Keine Persistenz.** Der Verlauf lebt nur in der Session; nichts wird auf Platte geschrieben.

**Modell & Kontext:**
- **Ein Modell pro Server.** `--model` ist nur ein Label — *welches* Modell antwortet, bestimmt allein das, was
  der Server geladen hat. Modellwechsel = Server neu starten.
- **Kontext-Decke 16384 Tokens** (Hardware-Grenze der GPU, serverseitig). Lange Sessions mit vielen
  Dateien/Verlauf stoßen daran.
- **gemma-QAT ist VRAM-eng:** auto-ctx gibt ihm nur ~2,5k Kontext im 16-GB-Budget — gut für kurze Turns, zu knapp
  für große Coding-Kontexte. Dafür ist **Qwen3-14B-Q4** (bis 16384) die bessere Wahl.

**Thinking-Modelle (Qwen3, gemma):**
- Bei langen Aufgaben kann der `<think>`-Block das Budget aufbrauchen → leere sichtbare Antwort (der Empty-Marker
  weist darauf hin). Abhilfe: `--max-tokens` höher oder `--no-think`. Default 6144 reicht für die meisten Chats;
  sehr lange Coding-Antworten mit Denken brauchen mehr.

**Bedienung:**
- Der **REPL braucht ein echtes Terminal** (TTY) — nicht per Pipe/Skript fütterbar. Fürs Skripten den
  Headless-Modus (`-p`) nutzen.

---

## 6. Troubleshooting

| Symptom | Ursache / Fix |
|---|---|
| `Connection refused` | Server lädt noch oder läuft nicht. Auf die `listening`-Zeile in Terminal 1 warten. |
| vf-clide erreicht den Server nicht | Port-Mismatch: `--port` (Terminal 1) = `--url` (Terminal 2). |
| **Antwort bricht mitten im Satz ab** | Token-Budget zu klein: `--max-tokens` höher (Default 6144). Die Kontextgröße wählt der Server selbst. |
| **Leere Antwort bei Thinking-Modell** | `<think>`-Block hat das Budget verbraucht. `--max-tokens` höher oder `--no-think`. vf-clide zeigt dazu einen Marker. |
| **Server bricht beim Laden ab mit FP8-Hinweis** | gemma-26B ohne KV-FP8: der Guard stoppt mit „…only FP8 (E4M3) KV is correct. Set `VULKANFORGE_KV_FP8=1`…". → mit `env VULKANFORGE_KV_FP8=1` neu starten. (Erzwingt FP8 durch **Abbruch**, aktiviert es nicht automatisch; Notüberbrückung: `VULKANFORGE_ALLOW_BROKEN_KV=1`.) |
| **Pipeline-/Shader-Fehler beim Start mit `--ctx-size`** | Override **> 16384** überschreitet das LDS-Limit der GPU → Abbruch bei der Pipeline-Erstellung. Wert ≤ 16384 wählen oder `--ctx-size` weglassen (auto-ctx bleibt sicher darunter). |
| **„out of memory" beim Start** | Nur wenn du `--ctx-size` *manuell* überschreibst — Wert runter oder Flag weglassen (auto-ctx wählt sicher). |
| Ein Flag wird nicht erkannt | `vulkanforge serve --help` bzw. `vf-clide --help` prüfen. |
