// SPDX-License-Identifier: GPL-3.0-only
//! Slice-1 agent tools. Only `read_file` (read-only).
//!
//! Each tool is a *definition* (the JSON-schema sent in the request's
//! `tools` array) plus an *executor* that returns a tool-result string and
//! **never panics** — errors come back as structured text the model can
//! react to, not as a crash.

use crate::types::Tool;

pub const READ_FILE: &str = "read_file";

/// Cap on returned file content (256 KB). Larger files are truncated with
/// a clear marker rather than flooding the context window / risking OOM.
pub const READ_FILE_CAP: usize = 256 * 1024;

/// The `read_file` tool definition (schema for the request `tools` array).
pub fn read_file_tool() -> Tool {
    Tool::function(
        READ_FILE,
        "Read a UTF-8 text file from disk and return its contents.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read" }
            },
            "required": ["path"]
        }),
    )
}

/// Execute `read_file` from its JSON-string arguments. Returns the
/// tool-result content (file body, possibly truncated) or a structured
/// error string. Never panics.
pub fn execute_read_file(arguments: &str) -> String {
    let path = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v) => match v.get("path").and_then(|p| p.as_str()) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => return "read_file error: missing required string argument \"path\".".to_string(),
        },
        Err(e) => return format!("read_file error: arguments were not valid JSON ({e})."),
    };
    read_path(&path)
}

fn read_path(path: &str) -> String {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => return format!("read_file error: {path}: {e}"),
    };
    if meta.is_dir() {
        return format!("read_file error: {path}: is a directory, not a file.");
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
        Err(e) => format!("read_file error: {path}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vf_clide_tools_{name}"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn tool_schema_has_required_path() {
        let t = read_file_tool();
        assert_eq!(t.function.name, "read_file");
        assert_eq!(t.function.parameters["required"][0], "path");
    }

    #[test]
    fn reads_an_existing_file() {
        let p = tmp("read", b"hello agent\n");
        let out = execute_read_file(&format!("{{\"path\":\"{}\"}}", p.display()));
        assert_eq!(out, "hello agent\n");
    }

    #[test]
    fn missing_file_returns_structured_error_not_panic() {
        let out = execute_read_file(r#"{"path":"/nonexistent/vf_clide_nope.txt"}"#);
        assert!(out.starts_with("read_file error:"), "got: {out}");
        assert!(out.contains("/nonexistent/vf_clide_nope.txt"));
    }

    #[test]
    fn directory_is_an_error() {
        let dir = std::env::temp_dir();
        let out = execute_read_file(&format!("{{\"path\":\"{}\"}}", dir.display()));
        assert!(out.contains("is a directory"), "got: {out}");
    }

    #[test]
    fn missing_or_bad_arguments() {
        assert!(execute_read_file(r#"{"nope":"x"}"#).contains("missing required"));
        assert!(execute_read_file("not json").contains("not valid JSON"));
        assert!(execute_read_file(r#"{"path":""}"#).contains("missing required"));
    }

    #[test]
    fn over_cap_is_truncated_with_marker() {
        let big = vec![b'a'; READ_FILE_CAP + 1000];
        let p = tmp("big", &big);
        let out = execute_read_file(&format!("{{\"path\":\"{}\"}}", p.display()));
        assert!(out.contains("[... truncated"), "expected a truncation marker");
        assert!(out.contains(&format!("file is {} bytes", READ_FILE_CAP + 1000)));
        // body itself capped to READ_FILE_CAP 'a's (+ the marker text).
        assert!(out.matches('a').count() <= READ_FILE_CAP + 50);
    }
}
