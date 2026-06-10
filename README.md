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
    LlmProvider, ProviderKind, RigBackend,
};

let credential = EnvCredentialSource
    .get_credential(ProviderKind::Openai)
    .await?; // reads GREENTIC_LLM_PROVIDER / GREENTIC_LLM_API_KEY / GREENTIC_LLM_BASE_URL
             // (+ optional GREENTIC_LLM_API_VERSION / GREENTIC_LLM_AWS_PROFILE)
let backend = RigBackend::new(ProviderKind::Openai, "gpt-4o", &credential)?;
let response = backend
    .chat(ChatRequest {
        messages: vec![ChatMessage::user("hello")],
        tools: vec![],
        tool_choice: None,
        max_tokens: None,
        temperature: None,
    })
    .await?;
```

## Tool calling & vision

`chat()` executes tool-calling and vision requests through rig's low-level
completion API. What a given backend accepts is governed by its
`Capabilities` matrix (`backend.capabilities()`): sending `tools` to a
provider with `tools: false`, or images to one with `vision: false`, returns
`LlmError::UnsupportedCapability("tools")` / `("vision")`. Streaming
(`chat_stream()`) is not implemented yet and always returns
`UnsupportedCapability("streaming")`.

One `chat()` call is exactly one completion — **the caller drives the tool
loop**:

```rust
use greentic_llm::{ChatMessage, ChatRequest, FinishReason, ToolDef};

let mut messages = vec![ChatMessage::user("what is the weather in Ghent?")];
let tools = vec![ToolDef {
    name: "get_weather".into(),
    description: "Resolve current weather for a city".into(),
    schema: serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
}];

loop {
    let response = backend
        .chat(ChatRequest {
            messages: messages.clone(),
            tools: tools.clone(),
            tool_choice: Some("auto".into()), // "auto" | "none" | "required" | a tool name
            max_tokens: None,
            temperature: None,
        })
        .await?;

    if response.finish_reason != FinishReason::ToolCalls {
        break; // final assistant text in response.content
    }

    // Replay the assistant turn, then answer every call by id.
    messages.push(ChatMessage::assistant_with_tool_calls(
        response.content.clone(),
        response.tool_calls.clone(),
    ));
    for call in &response.tool_calls {
        let result = dispatch_tool(&call.name, &call.arguments)?; // your dispatcher
        messages.push(ChatMessage::tool_result(call.id.clone(), result));
    }
}
```

The `ToolCall.id` surfaced in `ChatResponse` is the provider's correlation id
(rig's `call_id` when present, else `id`); echo it back unchanged via
`ChatMessage::tool_result` so providers can pair results with calls.

Vision: attach images to a user message via `ChatMessage::images`
(`ChatImage { data_base64, media_type }`, e.g. `image/png`); providers with
`vision: true` receive them as base64 image content parts.

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
