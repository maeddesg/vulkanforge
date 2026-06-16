// SPDX-License-Identifier: GPL-3.0-only
//! Agent tools: `read_file` + `search` (read-only) and `write_file` +
//! `shell` (mutating).
//!
//! Each tool is a *definition* (the JSON-schema sent in the request's
//! `tools` array, carrying its [`ToolRisk`]) plus an *executor* that
//! returns a tool-result string and **never panics** — errors come back as
//! structured text the model can react to, not as a crash.
//!
//! ## Workspace confinement (Phase 2)
//! All **file** paths (read and write) go through [`confined_path`], which
//! resolves `..` lexically and resolves symlinks in any existing prefix
//! (via `canonicalize`), then rejects anything outside the canonical
//! workspace root. This retroactively confines the Slice-1 `read_file`.
//!
//! **Honest limit:** `shell` is *not* path-confinable — a command can
//! `cat ~/.ssh/id_rsa` regardless of `cwd`. Its protection is therefore
//! **not** the workspace root but the `Mutating` permission class (always
//! gated; never auto-approved by `--yes`). cwd is set to the root only so
//! relative commands behave intuitively, not as a security boundary.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{Tool, ToolRisk};

pub const READ_FILE: &str = "read_file";
pub const WRITE_FILE: &str = "write_file";
pub const SEARCH: &str = "search";
pub const SHELL: &str = "shell";
/// Memory tools (Stufe B-2). Dispatched on their own axis (async, via the
/// HTTP memory client), **before** the file/shell permission gate — they
/// never touch files or the shell, so they are visible but not ceiling-gated.
pub const RECALL: &str = "recall";
pub const REMEMBER: &str = "remember";

/// Cap on returned file / shell content (256 KB). Larger output is
/// truncated with a clear marker rather than flooding the context / OOM.
pub const READ_FILE_CAP: usize = 256 * 1024;
pub const SHELL_OUTPUT_CAP: usize = 256 * 1024;
/// Search result caps: stop after this many hits or this many output bytes.
pub const SEARCH_MAX_HITS: usize = 100;
pub const SEARCH_OUTPUT_CAP: usize = 64 * 1024;
/// Wall-clock cap on a `shell` command before it is killed (no hang).
pub const SHELL_TIMEOUT: Duration = Duration::from_secs(30);

/// All Slice-2 tool definitions (schema + risk), in dispatch order.
pub fn all_tools() -> Vec<Tool> {
    vec![read_file_tool(), write_file_tool(), search_tool(), shell_tool()]
}

// -----------------------------------------------------------------
// Tool definitions (schema + risk = the gating source of truth).

pub fn read_file_tool() -> Tool {
    Tool::function(
        READ_FILE,
        "Read a UTF-8 text file (within the workspace) and return its contents.",
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string", "description": "Path to the file to read" } },
            "required": ["path"]
        }),
        ToolRisk::ReadOnly,
    )
}

pub fn write_file_tool() -> Tool {
    Tool::function(
        WRITE_FILE,
        "Create or overwrite a UTF-8 text file (within the workspace). Missing parent \
         directories inside the workspace are created.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write" },
                "content": { "type": "string", "description": "Full file contents to write" }
            },
            "required": ["path", "content"]
        }),
        ToolRisk::Mutating,
    )
}

pub fn search_tool() -> Tool {
    Tool::function(
        SEARCH,
        "Search the workspace FILES/CODE for a substring; returns file:line matches (capped). \
         This searches files on disk, NOT the project's memory — use `recall` for remembered \
         notes, decisions, or past context.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Substring to search for in workspace files" },
                "path": { "type": "string", "description": "Optional subdirectory (within the workspace) to limit the search" }
            },
            "required": ["query"]
        }),
        ToolRisk::ReadOnly,
    )
}

pub fn shell_tool() -> Tool {
    Tool::function(
        SHELL,
        "Run a shell command in the workspace directory and return stdout/stderr/exit code. \
         NOT confined to the workspace (a command can read paths anywhere); gated as a mutating tool.",
        serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string", "description": "The shell command to run" } },
            "required": ["command"]
        }),
        // `Exec` is its own top tier above `Mutating`: shell is the only
        // tool the workspace confinement cannot fence in, so the gate is
        // its SOLE guard (needs the loud `--allow-shell`, not `--allow-mutating`).
        ToolRisk::Exec,
    )
}

// -----------------------------------------------------------------
// Memory tools (Stufe B-2) — definitions only. They are NOT executed here
// (execution is async, via the HTTP memory client in `agent.rs`) and are NOT
// gated by the file/shell ceiling (a separate axis — see `agent::dispatch_memory`).
// The `risk` field is a required placeholder (`ReadOnly`); it is never read,
// because memory calls are intercepted before the permission gate.

pub fn recall_tool() -> Tool {
    Tool::function(
        RECALL,
        "Search this project's stored long-term MEMORY (notes, decisions, learnings, bugs saved \
         in THIS project) for relevant past context. Use for questions about earlier decisions, \
         conventions, or context — e.g. \"what did we decide / what was our rule for …\". This is \
         NOT a file search: use `search` for file or code contents.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "What to look for in memory (natural language)" },
                "k": { "type": "integer", "description": "Max notes to return (default 5)" }
            },
            "required": ["query"]
        }),
        ToolRisk::ReadOnly,
    )
}

pub fn remember_tool() -> Tool {
    Tool::function(
        REMEMBER,
        "Store a durable decision, learning, or bug worth recalling in future sessions. Use \
         sparingly — only things genuinely worth keeping, not routine actions.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "kind": { "type": "string", "description": "Decision | Learning | Bug | Benchmark | Artifact | Concept | Note" },
                "text": { "type": "string", "description": "The content to remember" }
            },
            "required": ["kind", "text"]
        }),
        ToolRisk::ReadOnly,
    )
}

/// The two memory tool definitions, added to the agent's tool set **only**
/// when the server reports memory enabled (the startup probe in `agent::run`).
pub fn memory_tools() -> Vec<Tool> {
    vec![recall_tool(), remember_tool()]
}

// -----------------------------------------------------------------
// Workspace confinement (Phase 2).

/// Lexically normalize a path: resolve `.` / `..` without touching the FS.
fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Resolve `requested` against the canonical workspace `root` and confine
/// it: the result must lie inside `root`. Relative paths are taken
/// relative to `root`; `..` is resolved lexically (step 1) and symlinks in
/// any existing prefix are resolved via `canonicalize` (step 2) so a
/// symlink pointing out is caught. Returns the confined absolute path or a
/// structured error string. Never panics.
fn confined_path(requested: &str, root: &Path) -> Result<PathBuf, String> {
    if requested.is_empty() {
        return Err("error: empty path.".into());
    }
    let req = Path::new(requested);
    let joined = if req.is_absolute() { req.to_path_buf() } else { root.join(req) };
    let lex = lexical_normalize(&joined);
    // 1) lexical containment — catches `../` escapes and absolute foreign paths.
    if !lex.starts_with(root) {
        return Err(format!(
            "error: \"{requested}\" resolves outside the workspace root ({})",
            root.display()
        ));
    }
    // 2) symlink containment — canonicalize the deepest existing ancestor
    //    (resolves every symlink in the existing chain); if it escapes the
    //    root, reject. The non-existing tail has no `..` (step 1 removed
    //    them) and no symlinks (doesn't exist yet).
    let mut anc: &Path = lex.as_path();
    loop {
        match anc.canonicalize() {
            Ok(c) => {
                if !c.starts_with(root) {
                    return Err(format!(
                        "error: \"{requested}\" escapes the workspace via a symlink"
                    ));
                }
                break;
            }
            Err(_) => match anc.parent() {
                Some(parent) => anc = parent,
                None => break, // nothing existed; lexical check already passed
            },
        }
    }
    Ok(lex)
}

// -----------------------------------------------------------------
// Arg parsing helpers (structured errors, never panic).

fn parse_json(arguments: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str(arguments).map_err(|e| format!("error: arguments were not valid JSON ({e})."))
}

fn get_str(v: &serde_json::Value, key: &str) -> Result<String, String> {
    match v.get(key).and_then(|x| x.as_str()) {
        Some(s) if !s.is_empty() => Ok(s.to_string()),
        _ => Err(format!("error: missing required string argument \"{key}\".")),
    }
}

// -----------------------------------------------------------------
// read_file (now confined).

pub fn execute_read_file(arguments: &str, workspace: &Path) -> String {
    let v = match parse_json(arguments) {
        Ok(v) => v,
        Err(e) => return format!("read_file {e}"),
    };
    let path = match get_str(&v, "path") {
        Ok(p) => p,
        Err(e) => return format!("read_file {e}"),
    };
    let target = match confined_path(&path, workspace) {
        Ok(t) => t,
        Err(e) => return format!("read_file {e}"),
    };
    read_file_at(&target)
}

fn read_file_at(path: &Path) -> String {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => return format!("read_file error: {}: {e}", path.display()),
    };
    if meta.is_dir() {
        return format!("read_file error: {}: is a directory, not a file.", path.display());
    }
    match std::fs::read(path) {
        Ok(bytes) => {
            let total = bytes.len();
            let capped = total > READ_FILE_CAP;
            let slice = if capped { &bytes[..READ_FILE_CAP] } else { &bytes[..] };
            let mut s = String::from_utf8_lossy(slice).into_owned();
            if capped {
                s.push_str(&format!(
                    "\n\n[... truncated: file is {total} bytes; showing the first {READ_FILE_CAP} bytes ...]"
                ));
            }
            s
        }
        Err(e) => format!("read_file error: {}: {e}", path.display()),
    }
}

/// Read the workspace `AGENTS.md` (project constitution override), confined
/// to the root exactly like the file tools (`confined_path`). Returns
/// `None` if it is missing, a directory, escapes the root (e.g. a symlink
/// out), or isn't valid UTF-8 — the caller treats absence as "no override".
pub fn read_agents_md(workspace: &Path) -> Option<String> {
    let target = confined_path("AGENTS.md", workspace).ok()?;
    if target.is_dir() {
        return None;
    }
    std::fs::read_to_string(&target).ok()
}

// -----------------------------------------------------------------
// write_file (mutating, confined).

pub fn execute_write_file(arguments: &str, workspace: &Path) -> String {
    let v = match parse_json(arguments) {
        Ok(v) => v,
        Err(e) => return format!("write_file {e}"),
    };
    let path = match get_str(&v, "path") {
        Ok(p) => p,
        Err(e) => return format!("write_file {e}"),
    };
    // `content` may legitimately be empty → don't use get_str (rejects "").
    let content = match v.get("content").and_then(|x| x.as_str()) {
        Some(c) => c.to_string(),
        None => return "write_file error: missing required string argument \"content\".".to_string(),
    };
    let target = match confined_path(&path, workspace) {
        Ok(t) => t,
        Err(e) => return format!("write_file {e}"),
    };
    if target.is_dir() {
        return format!("write_file error: {}: is a directory.", target.display());
    }
    let existed = target.exists();
    // Parent dirs are inside the root (target is confined) → safe to create.
    if let Some(parent) = target.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return format!("write_file error: could not create parent dir {}: {e}", parent.display());
            }
        }
    }
    match std::fs::write(&target, content.as_bytes()) {
        Ok(()) => format!(
            "write_file: {} {} ({} bytes).",
            if existed { "overwrote" } else { "created" },
            target.display(),
            content.len()
        ),
        Err(e) => format!("write_file error: {}: {e}", target.display()),
    }
}

// -----------------------------------------------------------------
// search (read-only, confined, capped).

pub fn execute_search(arguments: &str, workspace: &Path) -> String {
    let v = match parse_json(arguments) {
        Ok(v) => v,
        Err(e) => return format!("search {e}"),
    };
    let query = match get_str(&v, "query") {
        Ok(q) => q,
        Err(e) => return format!("search {e}"),
    };
    // Optional subdirectory, confined to the workspace.
    let base = match v.get("path").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
        Some(sub) => match confined_path(sub, workspace) {
            Ok(t) => t,
            Err(e) => return format!("search {e}"),
        },
        None => workspace.to_path_buf(),
    };

    let mut hits: Vec<String> = Vec::new();
    let mut bytes = 0usize;
    let mut capped = false;
    walk_search(&base, workspace, &query, &mut hits, &mut bytes, &mut capped);

    if hits.is_empty() {
        return format!("search: no matches for {query:?} under {}.", display_rel(&base, workspace));
    }
    let mut out = format!("search: {} match(es) for {query:?}:\n", hits.len());
    out.push_str(&hits.join("\n"));
    if capped {
        out.push_str(&format!(
            "\n[... results capped at {SEARCH_MAX_HITS} hits / {SEARCH_OUTPUT_CAP} bytes — narrow the query ...]"
        ));
    }
    out
}

/// Directories skipped while searching (VCS / build noise).
const SEARCH_SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__"];

fn walk_search(
    dir: &Path,
    root: &Path,
    query: &str,
    hits: &mut Vec<String>,
    bytes: &mut usize,
    capped: &mut bool,
) {
    if *capped {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    // Deterministic order.
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if *capped {
            return;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Use the entry's OWN type via `symlink_metadata` (which does NOT
        // follow the final component) — `Path::is_dir`/`is_file` follow
        // symlinks, so a symlink pointing OUT of the workspace (e.g.
        // `escape -> /etc`) would be seen as a directory/file and
        // recursed/read, leaking files outside the root. A code search has
        // no reason to follow symlinks out of the workspace, so skip them
        // entirely (closes the confinement hole + avoids symlink cycles).
        let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
        let ft = meta.file_type();
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            if SEARCH_SKIP_DIRS.contains(&name) {
                continue;
            }
            walk_search(&path, root, query, hits, bytes, capped);
        } else if ft.is_file() {
            // Read as text; skip binary / unreadable files silently.
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let rel = display_rel(&path, root);
            for (i, line) in text.lines().enumerate() {
                if line.contains(query) {
                    let trimmed: String = line.chars().take(200).collect();
                    let hit = format!("{rel}:{}: {}", i + 1, trimmed.trim());
                    *bytes += hit.len() + 1;
                    hits.push(hit);
                    if hits.len() >= SEARCH_MAX_HITS || *bytes >= SEARCH_OUTPUT_CAP {
                        *capped = true;
                        return;
                    }
                }
            }
        }
    }
}

fn display_rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).display().to_string()
}

// -----------------------------------------------------------------
// shell (mutating; cwd = workspace; output cap + timeout; never hangs).

pub fn execute_shell(arguments: &str, workspace: &Path) -> String {
    let v = match parse_json(arguments) {
        Ok(v) => v,
        Err(e) => return format!("shell {e}"),
    };
    let command = match get_str(&v, "command") {
        Ok(c) => c,
        Err(e) => return format!("shell {e}"),
    };
    run_shell(&command, workspace, SHELL_TIMEOUT)
}

/// Run `command` in `cwd`, capturing stdout/stderr/exit, capped + killed
/// after `timeout`. `timeout` is a parameter (injectable) so the kill /
/// no-hang path is unit-testable without waiting the production 30 s.
fn run_shell(command: &str, cwd: &Path, timeout: Duration) -> String {
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return format!("shell error: failed to spawn command: {e}"),
    };

    // Drain stdout/stderr to EOF on separate threads (prevents pipe-fill
    // deadlock + lets the child finish) while retaining only the first
    // CAP bytes (bounded memory). Returns (kept, total).
    let mut so = child.stdout.take().expect("piped stdout");
    let mut se = child.stderr.take().expect("piped stderr");
    let h_o = thread::spawn(move || drain_capped(&mut so));
    let h_e = thread::spawn(move || drain_capped(&mut se));

    let start = Instant::now();
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(s)) => break (Some(s), false),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break (None, true);
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return format!("shell error: waiting on command failed: {e}"),
        }
    };
    let (out, out_total) = h_o.join().unwrap_or_default();
    let (err, err_total) = h_e.join().unwrap_or_default();

    let render = |label: &str, kept: Vec<u8>, total: usize| -> String {
        let mut s = String::from_utf8_lossy(&kept).into_owned();
        if total > kept.len() {
            s.push_str(&format!("\n[... {label} truncated: {total} bytes total, kept {} ...]", kept.len()));
        }
        s
    };
    let out = render("stdout", out, out_total);
    let err = render("stderr", err, err_total);

    if timed_out {
        return format!(
            "shell: TIMED OUT after {}s and was killed.\n--- stdout ---\n{out}\n--- stderr ---\n{err}",
            timeout.as_secs()
        );
    }
    let code = match status.and_then(|s| s.code()) {
        Some(c) => c.to_string(),
        None => "killed-by-signal".to_string(),
    };
    format!("shell: exit_code={code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}")
}

/// Read a pipe to EOF, retaining at most `SHELL_OUTPUT_CAP` bytes.
/// Returns `(kept, total_read)`. Reads past the cap (discarding) so the
/// child can finish without blocking on a full pipe.
fn drain_capped(r: &mut impl Read) -> (Vec<u8>, usize) {
    let mut kept = Vec::new();
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                if kept.len() < SHELL_OUTPUT_CAP {
                    let take = (SHELL_OUTPUT_CAP - kept.len()).min(n);
                    kept.extend_from_slice(&buf[..take]);
                }
            }
            Err(_) => break,
        }
    }
    (kept, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, canonical temp workspace dir for one test.
    fn workspace(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vf_clide_ws_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p.canonicalize().unwrap()
    }

    fn args(json: serde_json::Value) -> String {
        serde_json::to_string(&json).unwrap()
    }

    // ---- definitions / risk ----

    #[test]
    fn definitions_carry_risk() {
        assert_eq!(read_file_tool().risk, ToolRisk::ReadOnly);
        assert_eq!(search_tool().risk, ToolRisk::ReadOnly);
        assert_eq!(write_file_tool().risk, ToolRisk::Mutating);
        assert_eq!(shell_tool().risk, ToolRisk::Exec); // own top tier (Slice 3)
        assert_eq!(all_tools().len(), 4);
    }

    #[test]
    fn recall_and_search_descriptions_are_disambiguated() {
        // B-2 finding #1: the agent used `search` (files) for a memory question.
        // Each tool's description must point AT the other so they can't be confused.
        let recall = recall_tool().function.description.to_lowercase();
        let search = search_tool().function.description.to_lowercase();
        assert!(recall.contains("memory"), "recall must say memory: {recall}");
        assert!(recall.contains("search") && recall.contains("file"),
            "recall must point at `search` for files: {recall}");
        assert!(search.contains("file"), "search must say files: {search}");
        assert!(search.contains("recall") && search.contains("memory"),
            "search must point at `recall` for memory: {search}");
    }

    // ---- AGENTS.md (constitution override), confined ----

    #[test]
    fn agents_md_present_is_read() {
        let ws = workspace("agents_present");
        std::fs::write(ws.join("AGENTS.md"), b"project rules").unwrap();
        assert_eq!(read_agents_md(&ws).as_deref(), Some("project rules"));
    }

    #[test]
    fn agents_md_missing_is_none() {
        let ws = workspace("agents_missing");
        assert!(read_agents_md(&ws).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn agents_md_escaping_symlink_is_ignored() {
        let ws = workspace("agents_symlink");
        // AGENTS.md as a symlink pointing OUT of the workspace → confined
        // read rejects it → treated as absent (no crash, no leak).
        std::os::unix::fs::symlink("/etc/hostname", ws.join("AGENTS.md")).unwrap();
        assert!(read_agents_md(&ws).is_none(), "escaping AGENTS.md symlink must be ignored");
    }

    #[test]
    fn write_schema_requires_path_and_content() {
        let t = write_file_tool();
        let req = &t.function.parameters["required"];
        assert!(req.as_array().unwrap().iter().any(|x| x == "path"));
        assert!(req.as_array().unwrap().iter().any(|x| x == "content"));
    }

    // ---- confinement (applies to read_file + write_file + search) ----

    #[test]
    fn read_inside_workspace_ok() {
        let ws = workspace("read_in");
        std::fs::write(ws.join("hello.txt"), b"hi there\n").unwrap();
        let out = execute_read_file(&args(serde_json::json!({"path": "hello.txt"})), &ws);
        assert_eq!(out, "hi there\n");
    }

    #[test]
    fn read_outside_via_parent_is_denied() {
        let ws = workspace("read_parent");
        let out = execute_read_file(&args(serde_json::json!({"path": "../../../etc/passwd"})), &ws);
        assert!(out.contains("outside the workspace"), "got: {out}");
    }

    #[test]
    fn read_absolute_foreign_path_is_denied() {
        let ws = workspace("read_abs");
        let out = execute_read_file(&args(serde_json::json!({"path": "/etc/passwd"})), &ws);
        assert!(out.contains("outside the workspace"), "got: {out}");
    }

    #[cfg(unix)]
    #[test]
    fn read_symlink_escaping_workspace_is_denied() {
        let ws = workspace("read_symlink");
        // A symlink inside the workspace that points OUT (to /etc).
        let link = ws.join("escape");
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        let out = execute_read_file(&args(serde_json::json!({"path": "escape/passwd"})), &ws);
        assert!(
            out.contains("symlink") || out.contains("outside the workspace"),
            "symlink escape not caught: {out}"
        );
        // sanity: the target really is reachable outside the guard.
        assert!(Path::new("/etc/passwd").exists());
    }

    // ---- write_file ----

    #[test]
    fn write_creates_then_overwrites_within_workspace() {
        let ws = workspace("write_ok");
        let out = execute_write_file(&args(serde_json::json!({"path": "out.txt", "content": "v1"})), &ws);
        assert!(out.contains("created"), "got: {out}");
        assert_eq!(std::fs::read_to_string(ws.join("out.txt")).unwrap(), "v1");
        let out2 = execute_write_file(&args(serde_json::json!({"path": "out.txt", "content": "v2"})), &ws);
        assert!(out2.contains("overwrote"), "got: {out2}");
        assert_eq!(std::fs::read_to_string(ws.join("out.txt")).unwrap(), "v2");
    }

    #[test]
    fn write_creates_parent_dirs_within_workspace() {
        let ws = workspace("write_parents");
        let out = execute_write_file(
            &args(serde_json::json!({"path": "a/b/c.txt", "content": "deep"})),
            &ws,
        );
        assert!(out.contains("created"), "got: {out}");
        assert_eq!(std::fs::read_to_string(ws.join("a/b/c.txt")).unwrap(), "deep");
    }

    #[test]
    fn write_outside_workspace_is_denied_and_not_executed() {
        let ws = workspace("write_out");
        let target = std::env::temp_dir().join("vf_clide_escaped_write.txt");
        let _ = std::fs::remove_file(&target);
        let out = execute_write_file(
            &args(serde_json::json!({"path": "../vf_clide_escaped_write.txt", "content": "X"})),
            &ws,
        );
        assert!(out.contains("outside the workspace"), "got: {out}");
        assert!(!target.exists(), "write escaped the workspace");
    }

    // ---- search ----

    #[test]
    fn search_finds_within_and_is_relative() {
        let ws = workspace("search_ok");
        std::fs::write(ws.join("a.txt"), b"alpha\nNEEDLE here\nbeta\n").unwrap();
        std::fs::create_dir_all(ws.join("sub")).unwrap();
        std::fs::write(ws.join("sub/b.txt"), b"nothing\nanother NEEDLE\n").unwrap();
        let out = execute_search(&args(serde_json::json!({"query": "NEEDLE"})), &ws);
        assert!(out.contains("a.txt:2:"), "got: {out}");
        assert!(out.contains("sub/b.txt:2:"), "got: {out}");
    }

    #[test]
    fn search_cap_applies() {
        let ws = workspace("search_cap");
        let mut body = String::new();
        for _ in 0..(SEARCH_MAX_HITS + 50) {
            body.push_str("MATCH\n");
        }
        std::fs::write(ws.join("big.txt"), body.as_bytes()).unwrap();
        let out = execute_search(&args(serde_json::json!({"query": "MATCH"})), &ws);
        assert!(out.contains("results capped"), "cap marker missing: {}", &out[..out.len().min(200)]);
    }

    #[test]
    fn search_path_outside_workspace_is_denied() {
        let ws = workspace("search_out");
        let out = execute_search(&args(serde_json::json!({"query": "x", "path": "/etc"})), &ws);
        assert!(out.contains("outside the workspace"), "got: {out}");
    }

    #[cfg(unix)]
    #[test]
    fn search_does_not_follow_escaping_symlink() {
        // v0.9.1 regression: the recursive walk used Path::is_dir/is_file,
        // which FOLLOW symlinks — a symlink out of the workspace was
        // recursed and its files read. The walk must skip symlinks.
        let ws = workspace("search_symlink");
        std::fs::write(ws.join("inside.txt"), b"NEEDLE inside the workspace\n").unwrap();
        // A directory OUTSIDE the workspace, with a matching file.
        let outside = workspace("search_symlink_outside");
        std::fs::write(outside.join("secret.txt"), b"NEEDLE outside the workspace\n").unwrap();
        // A symlink inside the workspace pointing at that outside dir.
        std::os::unix::fs::symlink(&outside, ws.join("escape")).unwrap();

        let out = execute_search(&args(serde_json::json!({"query": "NEEDLE"})), &ws);
        assert!(out.contains("inside.txt"), "must find the in-workspace file: {out}");
        assert!(
            !out.contains("secret.txt") && !out.contains("escape/"),
            "search followed an escaping symlink and read outside the workspace: {out}"
        );
    }

    // ---- shell ----

    #[test]
    fn shell_captures_stdout_and_exit() {
        let ws = workspace("shell_ok");
        let out = execute_shell(&args(serde_json::json!({"command": "echo hello-shell"})), &ws);
        assert!(out.contains("exit_code=0"), "got: {out}");
        assert!(out.contains("hello-shell"), "got: {out}");
    }

    #[test]
    fn shell_runs_in_workspace_cwd() {
        let ws = workspace("shell_cwd");
        std::fs::write(ws.join("marker.txt"), b"x").unwrap();
        let out = execute_shell(&args(serde_json::json!({"command": "ls"})), &ws);
        assert!(out.contains("marker.txt"), "cwd not the workspace: {out}");
    }

    #[test]
    fn shell_nonzero_exit_captured() {
        let ws = workspace("shell_exit");
        let out = execute_shell(&args(serde_json::json!({"command": "exit 7"})), &ws);
        assert!(out.contains("exit_code=7"), "got: {out}");
    }

    #[test]
    fn shell_output_cap_applies() {
        let ws = workspace("shell_cap");
        // Emit > SHELL_OUTPUT_CAP bytes; must be truncated with a marker.
        let cmd = format!("yes A | head -c {}", SHELL_OUTPUT_CAP + 5000);
        let out = execute_shell(&args(serde_json::json!({"command": cmd})), &ws);
        assert!(out.contains("truncated"), "output cap marker missing");
    }

    #[test]
    fn shell_injected_timeout_kills_and_does_not_hang() {
        // Inject a 1 s timeout against a 30 s sleep → the kill / no-hang
        // path runs for real (no need to wait the production 30 s): a
        // structured TIMED OUT result comes back in ~1 s, not ~30 s.
        let ws = workspace("shell_timeout");
        let start = Instant::now();
        let out = run_shell("sleep 30", &ws, Duration::from_secs(1));
        let elapsed = start.elapsed();
        assert!(out.contains("TIMED OUT"), "expected a timeout result, got: {out}");
        assert!(elapsed < Duration::from_secs(10), "did not honor the timeout (took {elapsed:?})");
    }

    // ---- arg errors ----

    #[test]
    fn bad_args_are_structured_not_panics() {
        let ws = workspace("badargs");
        assert!(execute_read_file("not json", &ws).contains("not valid JSON"));
        assert!(execute_read_file(&args(serde_json::json!({"nope": 1})), &ws).contains("missing required"));
        assert!(execute_write_file(&args(serde_json::json!({"path": "x"})), &ws).contains("content"));
        assert!(execute_search("nope", &ws).contains("not valid JSON"));
        assert!(execute_shell(&args(serde_json::json!({})), &ws).contains("missing required"));
    }
}
