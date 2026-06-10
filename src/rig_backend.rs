//! `RigBackend` — dispatches the [`LlmProvider`] trait to rig-core 0.38
//! provider clients (plus `rig-bedrock` behind the `bedrock` feature).
//!
//! Current scope: text-only chat via `Agent<M>::prompt`. Tool calling and
//! vision return `LlmError::UnsupportedCapability`; streaming is not yet
//! implemented and also returns `UnsupportedCapability("streaming")`.
//!
//! Architectural note — tools are dynamic
//! ---------------------------------------
//! Our `LlmProvider::chat()` accepts `req.tools: Vec<ToolDef>` per call —
//! tools are runtime-discovered from WASM extensions and may differ between
//! requests. `rig_core::agent::AgentBuilder::tool(...)` consumes `self`, so we
//! cannot pre-build an `Agent<M>` and add tools later. `RigBackend` therefore
//! stores the rig `Client` (provider connection) plus the model name only, and
//! builds a fresh agent inside `chat()` / `chat_stream()` from the per-request
//! tools. The underlying HTTP connection in `Client` is reused across calls.
//!
//! OpenAI completions API choice
//! -----------------------------
//! rig's default `openai::Client` posts to `/responses` (Responses API).
//! Our wiremock test speaks Chat Completions, so we explicitly use
//! `openai::CompletionsClient` here. Work that needs Responses API features
//! (built-in tools, web search) can opt in via a new `ProviderKind` variant.
//!
//! Provider-specific construction
//! ------------------------------
//! - Azure requires `cred.base_url` (the resource endpoint) and accepts an
//!   optional `cred.api_version`; the key is sent as the `api-key` header.
//! - Bedrock (feature `bedrock`) authenticates through the AWS credential
//!   chain — `cred.api_key` is ignored, `cred.aws_profile` selects a named
//!   profile, and `cred.base_url` is rejected because the AWS SDK derives the
//!   endpoint from the region.
//! - Ollama and Llamafile are keyless local daemons (rig's `Nothing` marker);
//!   `base_url` defaults to `http://localhost:11434` / `http://localhost:8080`.
//! - Hugging Face uses the default `SubProvider::HFInference` router; other
//!   sub-providers can be reached via `base_url` for now.

use async_trait::async_trait;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

use super::capabilities::{Capabilities, ProviderKind};
use super::credentials::Credential;
use super::provider::{
    ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError, LlmProvider, MessageRole,
};

/// Backend that dispatches `LlmProvider` calls to rig provider clients.
pub struct RigBackend {
    kind: ProviderKind,
    model: String,
    inner: Inner,
}

impl RigBackend {
    /// Doc-hidden accessor used by greentic-designer's `rig_agent` module
    /// to drive per-provider `AgentBuilder` construction.
    ///
    /// Exposed (doc-hidden) for greentic-designer's `rig_agent`, which builds
    /// per-provider `AgentBuilder`s on top of this backend. Not a stable API.
    #[doc(hidden)]
    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

/// One variant per supported provider. Each variant holds the rig provider
/// `Client` (HTTP connection + auth headers); the `Agent<M>` is built fresh
/// per `chat()` call because `AgentBuilder::tool` consumes `self`.
///
/// Exposed (doc-hidden) for greentic-designer's `rig_agent`, which builds
/// per-provider `AgentBuilder`s on top of this backend. Not a stable API.
#[doc(hidden)]
pub enum Inner {
    Openai(rig_core::providers::openai::CompletionsClient),
    Anthropic(rig_core::providers::anthropic::Client),
    Deepseek(rig_core::providers::deepseek::Client),
    Gemini(rig_core::providers::gemini::Client),
    Cohere(rig_core::providers::cohere::Client),
    Ollama(rig_core::providers::ollama::Client),
    Groq(rig_core::providers::groq::Client),
    Perplexity(rig_core::providers::perplexity::Client),
    Xai(rig_core::providers::xai::Client),
    Azure(rig_core::providers::azure::Client),
    Mistral(rig_core::providers::mistral::Client),
    Openrouter(rig_core::providers::openrouter::Client),
    Huggingface(rig_core::providers::huggingface::Client),
    Together(rig_core::providers::together::Client),
    Moonshot(rig_core::providers::moonshot::Client),
    Minimax(rig_core::providers::minimax::Client),
    Hyperbolic(rig_core::providers::hyperbolic::Client),
    Galadriel(rig_core::providers::galadriel::Client),
    Mira(rig_core::providers::mira::Client),
    Zai(rig_core::providers::zai::Client),
    Xiaomimimo(rig_core::providers::xiaomimimo::Client),
    Llamafile(rig_core::providers::llamafile::Client),
    #[cfg(feature = "bedrock")]
    Bedrock(rig_bedrock::client::Client),
}

impl RigBackend {
    /// Construct a backend for the given provider with the supplied model
    /// name and credentials.
    ///
    /// `cred.base_url` overrides the provider's default endpoint where the
    /// underlying rig client supports `ClientBuilder::base_url(...)` (every
    /// keyed provider does). For Ollama and Llamafile the credential's
    /// `api_key` is ignored (rig's transport uses the `Nothing` API-key
    /// marker) and `base_url` defaults to the local daemon. See the module
    /// docs for Azure and Bedrock specifics.
    pub fn new(kind: ProviderKind, model: &str, cred: &Credential) -> Result<Self, LlmError> {
        // Build a client with the bearer-style `Client::builder().api_key(..)
        // .base_url(..).build()` pattern shared by every keyed provider.
        // `$client_ty` differs per provider, so a `macro_rules!` keeps the
        // boilerplate compact without erasing types.
        macro_rules! build_keyed {
            ($client_ty:ty, $variant:ident, $label:literal) => {{
                let mut builder = <$client_ty>::builder().api_key(&cred.api_key);
                if let Some(base) = &cred.base_url {
                    builder = builder.base_url(base);
                }
                let client = builder
                    .build()
                    .map_err(|e| LlmError::Transport(format!("{} client: {e}", $label)))?;
                Inner::$variant(client)
            }};
        }

        // Keyless local daemons (Ollama, Llamafile): rig's builder takes the
        // `Nothing` marker instead of an API key and falls back to the
        // daemon's default localhost base URL.
        macro_rules! build_keyless {
            ($client_ty:ty, $variant:ident, $label:literal) => {{
                let mut builder = <$client_ty>::builder().api_key(rig_core::client::Nothing);
                if let Some(base) = &cred.base_url {
                    builder = builder.base_url(base);
                }
                let client = builder
                    .build()
                    .map_err(|e| LlmError::Transport(format!("{} client: {e}", $label)))?;
                Inner::$variant(client)
            }};
        }

        let inner = match kind {
            ProviderKind::Openai => {
                build_keyed!(
                    rig_core::providers::openai::CompletionsClient,
                    Openai,
                    "openai"
                )
            }
            ProviderKind::Anthropic => {
                build_keyed!(
                    rig_core::providers::anthropic::Client,
                    Anthropic,
                    "anthropic"
                )
            }
            ProviderKind::Deepseek => {
                build_keyed!(rig_core::providers::deepseek::Client, Deepseek, "deepseek")
            }
            ProviderKind::Gemini => {
                build_keyed!(rig_core::providers::gemini::Client, Gemini, "gemini")
            }
            ProviderKind::Cohere => {
                build_keyed!(rig_core::providers::cohere::Client, Cohere, "cohere")
            }
            ProviderKind::Groq => {
                build_keyed!(rig_core::providers::groq::Client, Groq, "groq")
            }
            ProviderKind::Perplexity => {
                build_keyed!(
                    rig_core::providers::perplexity::Client,
                    Perplexity,
                    "perplexity"
                )
            }
            ProviderKind::Xai => {
                build_keyed!(rig_core::providers::xai::Client, Xai, "xai")
            }
            ProviderKind::Mistral => {
                build_keyed!(rig_core::providers::mistral::Client, Mistral, "mistral")
            }
            ProviderKind::Openrouter => {
                build_keyed!(
                    rig_core::providers::openrouter::Client,
                    Openrouter,
                    "openrouter"
                )
            }
            ProviderKind::Huggingface => {
                build_keyed!(
                    rig_core::providers::huggingface::Client,
                    Huggingface,
                    "huggingface"
                )
            }
            ProviderKind::Together => {
                build_keyed!(rig_core::providers::together::Client, Together, "together")
            }
            ProviderKind::Moonshot => {
                build_keyed!(rig_core::providers::moonshot::Client, Moonshot, "moonshot")
            }
            ProviderKind::Minimax => {
                build_keyed!(rig_core::providers::minimax::Client, Minimax, "minimax")
            }
            ProviderKind::Hyperbolic => {
                build_keyed!(
                    rig_core::providers::hyperbolic::Client,
                    Hyperbolic,
                    "hyperbolic"
                )
            }
            ProviderKind::Galadriel => {
                build_keyed!(
                    rig_core::providers::galadriel::Client,
                    Galadriel,
                    "galadriel"
                )
            }
            ProviderKind::Mira => {
                build_keyed!(rig_core::providers::mira::Client, Mira, "mira")
            }
            ProviderKind::Zai => {
                build_keyed!(rig_core::providers::zai::Client, Zai, "zai")
            }
            ProviderKind::Xiaomimimo => {
                build_keyed!(
                    rig_core::providers::xiaomimimo::Client,
                    Xiaomimimo,
                    "xiaomimimo"
                )
            }
            ProviderKind::Ollama => {
                build_keyless!(rig_core::providers::ollama::Client, Ollama, "ollama")
            }
            ProviderKind::Llamafile => {
                build_keyless!(
                    rig_core::providers::llamafile::Client,
                    Llamafile,
                    "llamafile"
                )
            }
            ProviderKind::Azure => {
                // Azure's endpoint is per-resource, so there is no usable
                // default: require it up front instead of failing on the
                // first request. The key goes out as the `api-key` header
                // (classic Azure OpenAI resource key); Entra bearer tokens
                // are not supported through this constructor yet.
                let endpoint = cred.base_url.clone().ok_or_else(|| {
                    LlmError::Config(
                        "azure requires base_url, e.g. https://{resource}.openai.azure.com"
                            .to_string(),
                    )
                })?;
                let mut builder = rig_core::providers::azure::Client::builder()
                    .api_key(rig_core::providers::azure::AzureOpenAIAuth::ApiKey(
                        cred.api_key.clone(),
                    ))
                    .azure_endpoint(endpoint);
                if let Some(version) = &cred.api_version {
                    builder = builder.api_version(version);
                }
                let client = builder
                    .build()
                    .map_err(|e| LlmError::Transport(format!("azure client: {e}")))?;
                Inner::Azure(client)
            }
            #[cfg(feature = "bedrock")]
            ProviderKind::Bedrock => {
                // Bedrock authenticates through the AWS credential chain;
                // the AWS SDK derives the endpoint from the region, so a
                // base_url override would be silently ignored — reject it.
                if cred.base_url.is_some() {
                    return Err(LlmError::Config(
                        "bedrock does not support base_url; set the region via AWS env vars \
                         or aws_profile"
                            .to_string(),
                    ));
                }
                let client = match &cred.aws_profile {
                    Some(profile) => rig_bedrock::client::Client::with_profile_name(profile),
                    None => {
                        use rig_core::client::ProviderClient;
                        rig_bedrock::client::Client::from_env()
                            .map_err(|e| LlmError::Config(format!("bedrock client: {e}")))?
                    }
                };
                Inner::Bedrock(client)
            }
            #[cfg(not(feature = "bedrock"))]
            ProviderKind::Bedrock => {
                return Err(LlmError::Config(
                    "greentic-llm was built without the `bedrock` cargo feature".to_string(),
                ));
            }
        };

        Ok(RigBackend {
            kind,
            model: model.to_string(),
            inner,
        })
    }
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Concatenate every `System` message in the request into a single preamble
/// string. Returns `None` if no system message is present (rig's
/// `AgentBuilder::preamble` is optional).
fn build_preamble(messages: &[super::provider::ChatMessage]) -> Option<String> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| matches!(m.role, MessageRole::System))
        .map(|m| m.content.clone())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Format the conversation history into a single prompt body suitable for
/// `Agent::prompt()`.
///
/// rig's `Prompt::prompt(prompt)` takes a single prompt string (or
/// `Message`), so we serialise the non-system turns into one body: every turn
/// before the final user message becomes a `Role: text` line in the history
/// header, and the final user message is the tail prompt. This mirrors the
/// existing legacy chat loop's flat-text format.
fn build_prompt_body(messages: &[super::provider::ChatMessage]) -> String {
    let last_user_idx = messages
        .iter()
        .rposition(|m| matches!(m.role, MessageRole::User))
        .unwrap_or(messages.len().saturating_sub(1));

    let mut history = String::new();
    for (idx, msg) in messages.iter().enumerate() {
        if matches!(msg.role, MessageRole::System) {
            // already lifted into the preamble
            continue;
        }
        if idx == last_user_idx {
            // tail handled separately
            continue;
        }
        let role_label = match msg.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::Tool => "Tool",
            MessageRole::System => unreachable!("system messages filtered above"),
        };
        history.push_str(&format!("{role_label}: {}\n", msg.content));
    }

    let tail = messages
        .get(last_user_idx)
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if history.is_empty() {
        tail
    } else {
        format!("{history}\n{tail}")
    }
}

/// Default `max_tokens` for Anthropic, which rejects requests without one.
// wired into chat() in the follow-up commit
#[allow(dead_code)]
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 4096;

/// Convert a greentic [`ChatRequest`] into rig's provider-agnostic
/// `CompletionRequest`.
///
/// System messages are folded into the preamble; user/assistant/tool turns
/// become rig chat history. Tool results are encoded as rig `User` messages
/// carrying `UserContent::ToolResult` keyed by `tool_call_id` (the same id
/// surfaced by [`map_choice`], so caller-driven tool loops round-trip).
// wired into chat() in the follow-up commit
#[allow(dead_code)]
fn build_completion_request(
    req: &ChatRequest,
    kind: ProviderKind,
) -> Result<rig_core::completion::CompletionRequest, LlmError> {
    use rig_core::message::{
        AssistantContent, ImageMediaType, Message, MimeType, ToolResultContent, UserContent,
    };

    let preamble = build_preamble(&req.messages);
    let mut history: Vec<Message> = Vec::new();
    for m in &req.messages {
        match m.role {
            MessageRole::System => {} // folded into the preamble
            MessageRole::User => {
                let mut content: Vec<UserContent> = Vec::new();
                if !m.content.is_empty() {
                    content.push(UserContent::text(m.content.clone()));
                }
                for img in &m.images {
                    content.push(UserContent::image_base64(
                        img.data_base64.clone(),
                        ImageMediaType::from_mime_type(&img.media_type),
                        None,
                    ));
                }
                if content.is_empty() {
                    content.push(UserContent::text(String::new()));
                }
                history.push(Message::User {
                    content: rig_core::OneOrMany::many(content)
                        .map_err(|_| LlmError::Parse("empty user content".into()))?,
                });
            }
            MessageRole::Assistant => {
                let mut content: Vec<AssistantContent> = Vec::new();
                if !m.content.is_empty() {
                    content.push(AssistantContent::text(m.content.clone()));
                }
                for tc in &m.tool_calls {
                    content.push(AssistantContent::tool_call(
                        tc.id.clone(),
                        tc.name.clone(),
                        tc.arguments.clone(),
                    ));
                }
                if content.is_empty() {
                    // Skip empty assistant turns rather than erroring; some
                    // callers store placeholder assistant rows.
                    continue;
                }
                history.push(Message::Assistant {
                    id: None,
                    content: rig_core::OneOrMany::many(content)
                        .map_err(|_| LlmError::Parse("empty assistant content".into()))?,
                });
            }
            MessageRole::Tool => {
                let id = m.tool_call_id.clone().unwrap_or_default();
                history.push(Message::User {
                    content: rig_core::OneOrMany::one(UserContent::tool_result(
                        id,
                        rig_core::OneOrMany::one(ToolResultContent::text(m.content.clone())),
                    )),
                });
            }
        }
    }
    let chat_history = rig_core::OneOrMany::many(history)
        .map_err(|_| LlmError::Parse("request contained no user/assistant/tool messages".into()))?;

    // Anthropic's API requires max_tokens; default it when the caller did not
    // set one so requests do not fail provider-side.
    let max_tokens = req
        .max_tokens
        .map(u64::from)
        .or_else(|| (kind == ProviderKind::Anthropic).then_some(ANTHROPIC_DEFAULT_MAX_TOKENS));

    Ok(rig_core::completion::CompletionRequest {
        model: None,
        preamble,
        chat_history,
        documents: vec![],
        tools: req
            .tools
            .iter()
            .map(|t| rig_core::completion::ToolDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.schema.clone(),
            })
            .collect(),
        temperature: req.temperature.map(f64::from),
        max_tokens,
        tool_choice: map_tool_choice(req.tool_choice.as_deref()),
        additional_params: None,
        output_schema: None,
    })
}

/// Map the greentic `tool_choice` string convention (`"auto"` / `"none"` /
/// `"required"` / a specific tool name) onto rig's `ToolChoice`.
// wired into chat() in the follow-up commit
#[allow(dead_code)]
fn map_tool_choice(choice: Option<&str>) -> Option<rig_core::message::ToolChoice> {
    match choice {
        None => None,
        Some("auto") => Some(rig_core::message::ToolChoice::Auto),
        Some("none") => Some(rig_core::message::ToolChoice::None),
        Some("required") => Some(rig_core::message::ToolChoice::Required),
        Some(name) => Some(rig_core::message::ToolChoice::Specific {
            function_names: vec![name.to_string()],
        }),
    }
}

/// Map rig's response choice back onto greentic's [`ChatResponse`].
///
/// Text parts are concatenated (newline-joined); tool calls surface the
/// provider correlation id (`call_id` when present, else `id`) so the caller
/// can echo it back via [`super::provider::ChatMessage::tool_result`].
/// Reasoning and image parts are not surfaced through `ChatResponse`.
// wired into chat() in the follow-up commit
#[allow(dead_code)]
fn map_choice(choice: rig_core::OneOrMany<rig_core::message::AssistantContent>) -> ChatResponse {
    use rig_core::message::AssistantContent;
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for part in choice {
        match part {
            AssistantContent::Text(t) => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&t.text);
            }
            AssistantContent::ToolCall(tc) => tool_calls.push(super::provider::ToolCall {
                id: tc.call_id.clone().unwrap_or(tc.id),
                name: tc.function.name,
                arguments: tc.function.arguments,
            }),
            // Reasoning / Image parts are not surfaced through ChatResponse.
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
        }
    }
    let finish_reason = if tool_calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
    };
    ChatResponse {
        content,
        tool_calls,
        finish_reason,
    }
}

// ============================================================================
// LlmProvider impl
// ============================================================================

#[async_trait]
impl LlmProvider for RigBackend {
    fn capabilities(&self) -> Capabilities {
        self.kind.into()
    }

    fn provider_name(&self) -> &'static str {
        self.kind.as_str()
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Text-only chat. Tool calls and vision return UnsupportedCapability
        // so callers receive a clear error; both surfaces are not yet wired.
        if !req.tools.is_empty() {
            return Err(LlmError::UnsupportedCapability("tool_calling_in_chat"));
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) {
            return Err(LlmError::UnsupportedCapability("vision_in_chat"));
        }

        let preamble = build_preamble(&req.messages);
        let prompt_body = build_prompt_body(&req.messages);

        // Each provider's `Client::agent(model)` returns an `AgentBuilder<M>`
        // with a different `M` generic, so the dispatch can't be DRY'd into a
        // helper function (the return type would need to be erased). The
        // macro below expands one identical block per provider — rebuilding a
        // fresh agent per call because `AgentBuilder::tool` consumes `self`.
        // The HTTP connection in the underlying `Client` is reused across calls.
        macro_rules! run_provider {
            ($client:expr) => {{
                let mut builder = $client.agent(&self.model);
                if let Some(preamble_text) = preamble.as_deref() {
                    builder = builder.preamble(preamble_text);
                }
                if let Some(temp) = req.temperature {
                    builder = builder.temperature(temp as f64);
                }
                if let Some(max) = req.max_tokens {
                    builder = builder.max_tokens(max as u64);
                }
                builder
                    .build()
                    .prompt(prompt_body.as_str())
                    .await
                    .map_err(|e| LlmError::Transport(e.to_string()))?
            }};
        }

        let text = match &self.inner {
            Inner::Openai(client) => run_provider!(client),
            Inner::Anthropic(client) => run_provider!(client),
            Inner::Deepseek(client) => run_provider!(client),
            Inner::Gemini(client) => run_provider!(client),
            Inner::Cohere(client) => run_provider!(client),
            Inner::Ollama(client) => run_provider!(client),
            Inner::Groq(client) => run_provider!(client),
            Inner::Perplexity(client) => run_provider!(client),
            Inner::Xai(client) => run_provider!(client),
            Inner::Azure(client) => run_provider!(client),
            Inner::Mistral(client) => run_provider!(client),
            Inner::Openrouter(client) => run_provider!(client),
            Inner::Huggingface(client) => run_provider!(client),
            Inner::Together(client) => run_provider!(client),
            Inner::Moonshot(client) => run_provider!(client),
            Inner::Minimax(client) => run_provider!(client),
            Inner::Hyperbolic(client) => run_provider!(client),
            Inner::Galadriel(client) => run_provider!(client),
            Inner::Mira(client) => run_provider!(client),
            Inner::Zai(client) => run_provider!(client),
            Inner::Xiaomimimo(client) => run_provider!(client),
            Inner::Llamafile(client) => run_provider!(client),
            #[cfg(feature = "bedrock")]
            Inner::Bedrock(client) => run_provider!(client),
        };

        Ok(ChatResponse {
            content: text,
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
        })
    }

    /// Streaming is not yet implemented by this backend.
    /// Returns `LlmError::UnsupportedCapability("streaming")` unconditionally.
    /// The canonical capability string checked by callers is `"streaming"`.
    async fn chat_stream(&self, _req: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedCapability("streaming"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatImage, ChatMessage, ToolCall, ToolDef};

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }

    fn tool_request() -> ChatRequest {
        ChatRequest {
            messages: vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
                ChatMessage::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        arguments: serde_json::json!({"q": "x"}),
                    }],
                ),
                ChatMessage::tool_result("call_1", "{\"answer\":42}"),
            ],
            tools: vec![ToolDef {
                name: "lookup".into(),
                description: "d".into(),
                schema: serde_json::json!({"type": "object"}),
            }],
            tool_choice: Some("auto".into()),
            max_tokens: None,
            temperature: Some(0.2),
        }
    }

    #[test]
    fn builds_completion_request_with_tools_and_history() {
        let req = tool_request();
        let r = build_completion_request(&req, ProviderKind::Anthropic).expect("convert");
        assert_eq!(r.preamble.as_deref(), Some("you are helpful"));
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.tools[0].name, "lookup");
        assert_eq!(r.max_tokens, Some(4096)); // anthropic default
        assert_eq!(r.temperature, Some(0.2f32 as f64));
        assert_eq!(r.chat_history.len(), 3); // user, assistant(tool_call), tool-result
        assert!(matches!(
            r.tool_choice,
            Some(rig_core::message::ToolChoice::Auto)
        ));
    }

    #[test]
    fn non_anthropic_max_tokens_stays_unset() {
        let req = tool_request();
        let r = build_completion_request(&req, ProviderKind::Openai).expect("convert");
        assert_eq!(r.max_tokens, None);
    }

    #[test]
    fn tool_call_history_round_trips_through_rig_messages() {
        // rig response with a Completions-API style call (call_id = None,
        // id = "call_9") must surface id "call_9"; replaying that id must
        // land on both the assistant tool_call and the tool_result.
        let req = tool_request();
        let r = build_completion_request(&req, ProviderKind::Openai).expect("convert");
        let history: Vec<_> = r.chat_history.into_iter().collect();
        match &history[1] {
            rig_core::message::Message::Assistant { content, .. } => match content.first() {
                rig_core::message::AssistantContent::ToolCall(tc) => {
                    assert_eq!(tc.id, "call_1");
                    assert_eq!(tc.function.name, "lookup");
                }
                other => panic!("expected tool call, got {other:?}"),
            },
            other => panic!("expected assistant message, got {other:?}"),
        }
        match &history[2] {
            rig_core::message::Message::User { content } => match content.first() {
                rig_core::message::UserContent::ToolResult(tr) => {
                    assert_eq!(tr.id, "call_1");
                }
                other => panic!("expected tool result, got {other:?}"),
            },
            other => panic!("expected user(tool result) message, got {other:?}"),
        }
    }

    #[test]
    fn maps_choice_with_tool_calls() {
        let choice = rig_core::OneOrMany::many(vec![
            rig_core::message::AssistantContent::text("thinking"),
            rig_core::message::AssistantContent::tool_call(
                "call_9",
                "lookup",
                serde_json::json!({"q": "y"}),
            ),
        ])
        .expect("non-empty");
        let resp = map_choice(choice);
        assert_eq!(resp.content, "thinking");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_9");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn maps_choice_prefers_call_id_when_present() {
        let choice = rig_core::OneOrMany::one(rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::new(
                "fc_123".into(),
                rig_core::message::ToolFunction {
                    name: "lookup".into(),
                    arguments: serde_json::json!({}),
                },
            )
            .with_call_id("call_abc".into()),
        ));
        let resp = map_choice(choice);
        assert_eq!(resp.tool_calls[0].id, "call_abc");
    }

    #[test]
    fn maps_text_only_choice_to_stop() {
        let choice = rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("hello"));
        let resp = map_choice(choice);
        assert_eq!(resp.content, "hello");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn user_message_with_image_becomes_image_content() {
        let mut msg = ChatMessage::user("look at this");
        msg.images.push(ChatImage {
            data_base64: "aGVsbG8=".into(),
            media_type: "image/png".into(),
        });
        let req = ChatRequest {
            messages: vec![msg],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let r = build_completion_request(&req, ProviderKind::Openai).expect("convert");
        let history: Vec<_> = r.chat_history.into_iter().collect();
        match &history[0] {
            rig_core::message::Message::User { content } => {
                let parts: Vec<_> = content.iter().collect();
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[0], rig_core::message::UserContent::Text(_)));
                match parts[1] {
                    rig_core::message::UserContent::Image(img) => {
                        assert_eq!(img.media_type, Some(rig_core::message::ImageMediaType::PNG));
                    }
                    other => panic!("expected image content, got {other:?}"),
                }
            }
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn all_system_messages_is_an_error_not_panic() {
        let req = ChatRequest {
            messages: vec![ChatMessage::system("only system")],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let err = build_completion_request(&req, ProviderKind::Openai)
            .expect_err("no chat turns must be an error");
        assert!(matches!(err, LlmError::Parse(_)));
    }

    #[test]
    fn maps_tool_choice_strings() {
        assert!(map_tool_choice(None).is_none());
        assert!(matches!(
            map_tool_choice(Some("auto")),
            Some(rig_core::message::ToolChoice::Auto)
        ));
        assert!(matches!(
            map_tool_choice(Some("none")),
            Some(rig_core::message::ToolChoice::None)
        ));
        assert!(matches!(
            map_tool_choice(Some("required")),
            Some(rig_core::message::ToolChoice::Required)
        ));
        match map_tool_choice(Some("lookup")) {
            Some(rig_core::message::ToolChoice::Specific { function_names }) => {
                assert_eq!(function_names, vec!["lookup".to_string()]);
            }
            other => panic!("expected Specific, got {other:?}"),
        }
    }

    fn system_msg(text: &str) -> ChatMessage {
        ChatMessage::system(text)
    }

    fn dummy_cred() -> Credential {
        // `Credential` implements `Drop` (zeroize), so struct-update syntax
        // is unavailable; spell out every field.
        Credential {
            api_key: "test-key".to_string(),
            base_url: None,
            expires_at: None,
            api_version: None,
            aws_profile: None,
        }
    }

    fn expect_config_err(result: Result<RigBackend, LlmError>, context: &str) {
        match result {
            Ok(_) => panic!("{context}: expected a Config error, got Ok"),
            Err(LlmError::Config(_)) => {}
            Err(other) => panic!("{context}: expected Config error, got: {other:?}"),
        }
    }

    #[test]
    fn preamble_lifts_only_system_messages() {
        let messages = vec![
            system_msg("be concise"),
            user_msg("hello"),
            system_msg("answer in english"),
        ];
        let preamble = build_preamble(&messages).expect("preamble");
        assert_eq!(preamble, "be concise\n\nanswer in english");
    }

    #[test]
    fn preamble_returns_none_without_system() {
        let messages = vec![user_msg("hello")];
        assert!(build_preamble(&messages).is_none());
    }

    #[test]
    fn prompt_body_uses_last_user_as_tail() {
        let messages = vec![
            user_msg("first turn"),
            ChatMessage::assistant("first reply"),
            user_msg("second turn"),
        ];
        let body = build_prompt_body(&messages);
        assert!(body.contains("User: first turn"));
        assert!(body.contains("Assistant: first reply"));
        assert!(body.ends_with("second turn"));
    }

    #[test]
    fn prompt_body_returns_single_user_when_no_history() {
        let messages = vec![user_msg("only message")];
        let body = build_prompt_body(&messages);
        assert_eq!(body, "only message");
    }

    #[test]
    fn every_provider_constructs_offline() {
        // Construction must never hit the network. Azure additionally needs
        // an endpoint; Bedrock is exercised separately because it depends on
        // the `bedrock` feature.
        for kind in ProviderKind::all() {
            if *kind == ProviderKind::Bedrock {
                continue;
            }
            let mut cred = dummy_cred();
            if *kind == ProviderKind::Azure {
                cred.base_url = Some("https://example.openai.azure.com".to_string());
            }
            let backend = RigBackend::new(*kind, "test-model", &cred)
                .unwrap_or_else(|e| panic!("{} backend should build: {e}", kind.as_str()));
            assert_eq!(backend.provider_name(), kind.as_str());
            assert_eq!(backend.model(), "test-model");
        }
    }

    #[test]
    fn azure_without_endpoint_is_a_config_error() {
        expect_config_err(
            RigBackend::new(ProviderKind::Azure, "gpt-4o", &dummy_cred()),
            "azure without base_url",
        );
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn bedrock_constructs_offline_with_and_without_profile() {
        // The AWS SDK config is loaded lazily on first request, so plain
        // construction must succeed without AWS credentials present.
        let backend = RigBackend::new(
            ProviderKind::Bedrock,
            "amazon.nova-lite-v1:0",
            &dummy_cred(),
        )
        .expect("bedrock from_env constructs");
        assert_eq!(backend.provider_name(), "bedrock");

        let mut cred = dummy_cred();
        cred.aws_profile = Some("greentic-test".to_string());
        RigBackend::new(ProviderKind::Bedrock, "amazon.nova-lite-v1:0", &cred)
            .expect("bedrock with profile constructs");
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn bedrock_rejects_base_url_override() {
        let mut cred = dummy_cred();
        cred.base_url = Some("https://example.com".to_string());
        expect_config_err(
            RigBackend::new(ProviderKind::Bedrock, "amazon.nova-lite-v1:0", &cred),
            "bedrock with base_url",
        );
    }

    #[cfg(not(feature = "bedrock"))]
    #[test]
    fn bedrock_without_feature_is_a_config_error() {
        expect_config_err(
            RigBackend::new(
                ProviderKind::Bedrock,
                "amazon.nova-lite-v1:0",
                &dummy_cred(),
            ),
            "bedrock without the feature",
        );
    }
}
