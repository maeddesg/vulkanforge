//! Pin the /v1/completions byte-identity gate at the TOKEN level (no GPU):
//! does `encode_with_special(render_full_history-as-string)` reproduce the
//! exact token vector `render_full_history` itself produces?
//!
//!   cargo run --release --example tok_diff -- ~/models/Qwen3-8B-Q4_K_M.gguf

use vulkanforge::backend::vulkan::chat_template::{ChatTemplate, HistoryRole, RenderMessage};
use vulkanforge::backend::vulkan::gguf::GgufFile;
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: tok_diff <gguf>")?;
    let gguf = GgufFile::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let tpl = ChatTemplate::detect(&gguf, &tok);
    println!("template = {tpl:?}");

    let msgs = [RenderMessage { role: HistoryRole::User, content: "The capital of France is" }];
    let t_chat = tpl.render_full_history(&tok, &msgs);
    let rendered = tok.decode(&t_chat);
    let t_comp = tok.encode_with_special(&rendered);

    println!("rendered string = {rendered:?}");
    println!("chat tokens ({}): {:?}", t_chat.len(), t_chat);
    println!("comp tokens ({}): {:?}", t_comp.len(), t_comp);

    if t_chat == t_comp {
        println!("RESULT: ✓ byte-identical token vectors — round-trip holds");
    } else {
        let n = t_chat.len().min(t_comp.len());
        let first = (0..n).find(|&i| t_chat[i] != t_comp[i]).unwrap_or(n);
        println!("RESULT: ✗ DIVERGE at index {first}");
        let lo = first.saturating_sub(3);
        for i in lo..(first + 4).min(t_chat.len().max(t_comp.len())) {
            let a = t_chat.get(i).map(|&x| format!("{x} {:?}", tok.decode(&[x]))).unwrap_or("—".into());
            let b = t_comp.get(i).map(|&x| format!("{x} {:?}", tok.decode(&[x]))).unwrap_or("—".into());
            let mark = if t_chat.get(i) != t_comp.get(i) { " <<<" } else { "" };
            println!("  [{i}] chat={a:<22} comp={b}{mark}");
        }
    }
    Ok(())
}
