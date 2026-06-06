# greentic-llm

Provider-agnostic multi-LLM abstraction for the Greentic platform.

Extracted from greentic-designer's post-rig-migration LLM layer. One trait,
nine providers (openai, anthropic, deepseek, gemini, cohere, ollama, groq,
perplexity, xai) via [rig-core](https://crates.io/crates/rig-core).

## Usage

```rust
use greentic_llm::{
    ChatMessage, ChatRequest, CredentialSource, EnvCredentialSource,
    LlmProvider, MessageRole, ProviderKind, RigBackend,
};

let credential = EnvCredentialSource
    .get_credential(ProviderKind::Openai)
    .await?; // reads GREENTIC_LLM_PROVIDER / GREENTIC_LLM_API_KEY / GREENTIC_LLM_BASE_URL
let backend = RigBackend::new(ProviderKind::Openai, "gpt-4o", &credential)?;
let response = backend
    .chat(ChatRequest {
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: "hello".into(),
            images: vec![],
        }],
        tools: vec![],
        tool_choice: None,
        max_tokens: None,
        temperature: None,
    })
    .await?;
```

## Features

- `clap` — `clap::ValueEnum` on `ProviderKind` for CLI flags.
- `test-mock` — `mock::TestLlmProviderBuilder`, a scriptable test double.

## Consumers

- `greentic-designer` (chat/agent routes, DW composer)
- `greentic-operala` (LLM-backed prompting)

## License

MIT
