// SPDX-License-Identifier: GPL-3.0-only
//! Phase-2 pinned status line (Option B — raw ANSI scroll region, no TUI
//! framework). The bottom terminal row stays fixed (a token meter + the
//! current action); everything above scrolls normally and the rustyline
//! prompt stays in the upper region.
//!
//! **No-op when stdout is not a TTY** (so `-p`/headless and piped output
//! stay byte-clean — no escapes, no meter). Interior mutability lets the
//! bar be shared as `&StatusBar` and updated from the REPL and the agent
//! loop without threading `&mut` everywhere.

use std::cell::RefCell;
use std::io::Write;

use crate::types::{SessionUsage, Usage};

pub struct StatusBar {
    enabled: bool,
    inner: RefCell<Inner>,
}

struct Inner {
    action: String,
    session: SessionUsage,
    turn: Usage,
}

impl StatusBar {
    /// `enabled` should be "stdout is a TTY". When false every method is a
    /// silent no-op (nothing written), keeping piped output byte-clean.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            inner: RefCell::new(Inner {
                action: "idle".into(),
                session: SessionUsage::default(),
                turn: Usage::default(),
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Reserve the bottom row: set the scroll region to rows `1..H-1` and
    /// drop the cursor inside it. Call once at REPL start.
    pub fn enter(&self) {
        if !self.enabled {
            return;
        }
        if let Some(h) = term_height() {
            let top = h.saturating_sub(1).max(1);
            let mut o = std::io::stdout();
            let _ = write!(o, "\x1b[1;{top}r\x1b[{top};1H");
            let _ = o.flush();
        }
        self.render();
    }

    /// Reset the scroll region, clear the status row, and park the cursor
    /// at the bottom. Call on every exit path (/quit, Ctrl-D, Ctrl-C) and
    /// before anything that needs the whole screen. Safe to call twice.
    pub fn leave(&self) {
        if !self.enabled {
            return;
        }
        if let Some(h) = term_height() {
            let mut o = std::io::stdout();
            let _ = write!(o, "\x1b[r\x1b[{h};1H\x1b[2K");
            let _ = o.flush();
        }
    }

    /// Set the right-hand "action" (e.g. `idle`, `generating…`,
    /// `running search(…)`) and repaint.
    pub fn set_action(&self, action: impl Into<String>) {
        if !self.enabled {
            return;
        }
        self.inner.borrow_mut().action = action.into();
        self.render();
    }

    /// Fold a completed turn's usage into the session total, show it as the
    /// current turn, and repaint.
    pub fn record_turn(&self, turn: Usage) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.session.add(&turn);
            inner.turn = turn;
        }
        if self.enabled {
            self.render();
        }
    }

    /// Repaint the status row: save cursor → bottom row → clear → write →
    /// restore cursor. Re-queries the terminal size each time (so a resize
    /// is picked up on the next event without a SIGWINCH handler).
    pub fn render(&self) {
        if !self.enabled {
            return;
        }
        let Some((w, h)) = term_size() else { return };
        let line = {
            let inner = self.inner.borrow();
            meter_line(&inner, w as usize)
        };
        let mut o = std::io::stdout();
        let _ = write!(o, "\x1b[s\x1b[{h};1H\x1b[2K\x1b[2m{line}\x1b[0m\x1b[u");
        let _ = o.flush();
    }
}

fn meter_line(inner: &Inner, width: usize) -> String {
    let up = inner.turn.prompt_tokens.unwrap_or(0);
    let down = inner.turn.completion_tokens.unwrap_or(0);
    let tot = inner.turn.total_tokens.unwrap_or(up + down);
    // `~` only if a turn's tokens were estimated (not currently — VF reports
    // real usage on both paths — but kept honest for a future server/path).
    let est = if inner.session.estimated { "~" } else { "" };
    let s = format!(
        "tokens: {up}↑ {down}↓ ({tot}) · session {est}{} · {}",
        human(inner.session.total),
        inner.action
    );
    truncate_cols(&s, width)
}

fn human(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Truncate to at most `cols` display columns (char-count ≈ columns: the
/// meter is ASCII plus the `↑`/`↓`/`·`/`…` glyphs, each one column).
fn truncate_cols(s: &str, cols: usize) -> String {
    if cols == 0 {
        return String::new();
    }
    if s.chars().count() <= cols {
        s.to_string()
    } else {
        s.chars().take(cols.saturating_sub(1)).collect::<String>() + "…"
    }
}

fn term_size() -> Option<(u16, u16)> {
    terminal_size::terminal_size().map(|(w, h)| (w.0, h.0))
}

fn term_height() -> Option<u16> {
    term_size().map(|(_, h)| h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_bar_is_silent_noop() {
        // When stdout isn't a TTY the bar must write nothing and never panic.
        let bar = StatusBar::new(false);
        bar.enter();
        bar.set_action("generating…");
        bar.record_turn(Usage { prompt_tokens: Some(5), completion_tokens: Some(3), total_tokens: Some(8) });
        bar.render();
        bar.leave();
        assert!(!bar.is_enabled());
    }

    #[test]
    fn meter_line_formats_and_truncates() {
        let inner = Inner {
            action: "running search(…)".into(),
            session: SessionUsage { prompt: 1000, completion: 1340, total: 2340, turns: 3, estimated: false },
            turn: Usage { prompt_tokens: Some(1234), completion_tokens: Some(56), total_tokens: Some(1290) },
        };
        let full = meter_line(&inner, 200);
        assert!(full.contains("1234↑"), "{full}");
        assert!(full.contains("56↓"), "{full}");
        assert!(full.contains("(1290)"), "{full}");
        assert!(full.contains("session 2.3k"), "{full}");
        assert!(full.contains("running search(…)"), "{full}");
        // narrow width truncates with an ellipsis, never exceeding the cap
        let narrow = meter_line(&inner, 12);
        assert!(narrow.chars().count() <= 12, "got {} cols: {narrow:?}", narrow.chars().count());
        assert!(narrow.ends_with('…'));
    }

    #[test]
    fn human_readable_thousands() {
        assert_eq!(human(0), "0");
        assert_eq!(human(999), "999");
        assert_eq!(human(2340), "2.3k");
    }
}
