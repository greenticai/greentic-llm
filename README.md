# greentic-llm

Provider-agnostic multi-LLM abstraction for the Greentic platform.

Extracted from greentic-designer's post-rig-migration LLM layer. One trait,
23 providers via [rig-core](https://crates.io/crates/rig-core) (plus
[rig-bedrock](https://crates.io/crates/rig-bedrock) behind the `bedrock`
feature):

| Group | Providers |
|---|---|
| Major clouds | `openai`, `anthropic`, `azure`, `bedrock`, `gemini` |
| Hosted APIs | `deepseek`, `cohere`, `groq`, `perplexity`, `xai`, `mistral`, `moonshot`, `minimax`, `zai`, `xiaomimimo` |
| Aggregators / routers | `openrouter`, `huggingface`, `together`, `hyperbolic`, `galadriel`, `mira` |
| Local daemons (keyless) | `ollama`, `llamafile` |

Provider-specific notes:

- **azure** — `base_url` is required (the resource endpoint,
  `https://{resource}.openai.azure.com`); the key is sent as the `api-key`
  header. Optional `api_version` overrides the GA default. The model name is
  the deployment id.
- **bedrock** — compile with the `bedrock` cargo feature. Authenticates via
  the standard AWS credential chain (env vars, `~/.aws`, instance roles);
  `api_key` is ignored and an optional `aws_profile` selects a named profile.
  The feature is off by default because it pulls in the AWS SDK.
- **ollama / llamafile** — keyless; `base_url` defaults to
  `http://localhost:11434` / `http://localhost:8080`.

## Usage

```rust
use greentic_llm::{
    ChatMessage, ChatRequest, CredentialSource, EnvCredentialSource,
    LlmProvider, MessageRole, ProviderKind, RigBackend,
};

let credential = EnvCredentialSource
    .get_credential(ProviderKind::Openai)
    .await?; // reads GREENTIC_LLM_PROVIDER / GREENTIC_LLM_API_KEY / GREENTIC_LLM_BASE_URL
             // (+ optional GREENTIC_LLM_API_VERSION / GREENTIC_LLM_AWS_PROFILE)
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

- `bedrock` — AWS Bedrock backend via `rig-bedrock` (off by default; pulls in
  the AWS SDK).
- `clap` — `clap::ValueEnum` on `ProviderKind` for CLI flags.
- `test-mock` — `mock::TestLlmProviderBuilder`, a scriptable test double.

## Consumers

- `greentic-designer` (chat/agent routes, DW composer)
- `greentic-operala` (LLM-backed prompting)

## License

MIT
