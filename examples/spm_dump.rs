use std::path::PathBuf;
use vulkanforge::backend::vulkan::gguf::{GgufFile, MetadataValue};
use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        format!("{}/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", std::env::var("HOME").unwrap())
    }));
    let gguf = GgufFile::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    println!("vocab_size = {}", tok.vocab_size());
    println!("bos_id     = {:?} ({:?})", tok.bos_id, tok.bos_id.and_then(|i| tok.token_str(i)));
    println!("eos_id     = {} ({:?})", tok.eos_id, tok.token_str(tok.eos_id));
    for n in ["[INST]", "[/INST]", "<s>", "</s>", "<unk>"] {
        println!("special    {:8} = {:?}", n, tok.special_id(n).map(|i| (i, tok.token_str(i))));
    }
    let template = ChatTemplate::detect(&gguf, &tok);
    println!("template   = {:?}", template);
    let prompt = template.render_first_turn(&tok, "", "What is 2 + 2?");
    println!("prompt ids = {:?}", prompt);
    print!("prompt str: ");
    for &id in &prompt {
        print!("[{:5}={:?}] ", id, tok.token_str(id));
    }
    println!();
    println!("decoded    = {:?}", tok.decode(&prompt));

    for s in ["▁", "▁What", "▁is", "What", "is", "2", "?", "▁2", "▁?",
              "▁the", "▁of", "▁a", "▁and", "Sure", "▁Sure", "cret", "▁cret",
              "▁secret", "▁Translate", "Translate", "▁sentence"] {
        println!("lookup     {:12} = {:?}", s, tok.special_id(s));
    }
    println!("\n-- first 16 vocab entries --");
    for i in 0..16 { println!("  {:4} = {:?}", i, tok.token_str(i)); }
    println!("\n-- vocab 28..40 (looking for ▁) --");
    for i in 28..40 { println!("  {:4} = {:?}", i, tok.token_str(i)); }
    println!("\n-- vocab 256..272 --");
    for i in 256..272 { println!("  {:4} = {:?}", i, tok.token_str(i)); }

    // Score / type lookup for the IDs we just queried.
    let scores = gguf.metadata.get("tokenizer.ggml.scores").and_then(|v| v.as_array()).unwrap();
    let types = gguf.metadata.get("tokenizer.ggml.token_type").and_then(|v| v.as_array()).unwrap();
    let probe_ids = [
        ("<unk>", 0u32),
        ("<s>", 1),
        ("</s>", 2),
        ("[INST]", 3),
        ("[/INST]", 4),
        ("▁", 29473),
        ("▁What", 2592),
        ("▁Wh", 1711),
        ("at", 1038),
        ("What", 3963),
        ("2", 29518),
        ("?", 29572),
        ("cret", 4695),
        ("<0xE2>", 997),
    ];
    println!("\n-- score / type for probe ids --");
    for (label, id) in probe_ids {
        let s = scores[id as usize].as_f32().unwrap_or(0.0);
        let t = types[id as usize].as_i32().unwrap_or(-1);
        println!("  {:8} id={:5} score={:>10.4} type={}", label, id, s, t);
    }
    // What is the actual encoded sequence for "What"?
    let what = tok.encode("What");
    println!("\nencode(\"What\") = {:?} → {:?}", what, what.iter().map(|&i| tok.token_str(i)).collect::<Vec<_>>());
    let space_what = tok.encode_no_prefix(" What");
    println!("encode_no_prefix(\" What\") = {:?} → {:?}", space_what, space_what.iter().map(|&i| tok.token_str(i)).collect::<Vec<_>>());
    let mutex = tok.encode_no_prefix(" Explain what a mutex is in one paragraph.");
    println!("\nencode_no_prefix(\" Explain ...\"):");
    for &id in &mutex {
        print!("  [{:5}={:?}]", id, tok.token_str(id));
    }
    println!();
    let trans = tok.encode_no_prefix(" Translate to German: Hello.");
    println!("\nencode_no_prefix(\" Translate ...\"):");
    for &id in &trans {
        print!("  [{:5}={:?}]", id, tok.token_str(id));
    }
    println!();
    let _ = MetadataValue::U8(0);  // suppress unused-import warning
    Ok(())
}
