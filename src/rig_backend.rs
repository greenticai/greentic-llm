//! `RigBackend` — dispatches the [`LlmProvider`] trait to rig-core 0.38
//! provider clients (plus `rig-bedrock` behind the `bedrock` feature).
//!
//! Scope: one-shot chat via rig's low-level completion API, including tool
//! calling and vision wherever the provider's [`Capabilities`] allow them
//! (requests that exceed the capability matrix are rejected with
//! `LlmError::UnsupportedCapability("tools")` / `("vision")`). `chat()`
//! executes exactly one completion per call — the **caller** drives the tool
//! loop: dispatch the returned `ChatResponse::tool_calls`, append the
//! assistant turn via `ChatMessage::assistant_with_tool_calls` plus one
//! `ChatMessage::tool_result` per call, and invoke `chat()` again. Streaming
//! is not yet implemented and returns `UnsupportedCapability("streaming")`.
//!
//! Architectural note — tools are dynamic
//! ---------------------------------------
//! Our `LlmProvider::chat()` accepts `req.tools: Vec<ToolDef>` per call —
//! tools are runtime-discovered from WASM extensions and may differ between
//! requests, and tool dispatch happens in the caller. That rules out rig's
//! `Agent` abstraction (tools are baked in at agent build time and rig would
//! drive the tool loop itself). Instead, `chat()` converts the request once
//! into rig's provider-agnostic `CompletionRequest` and dispatches it via
//! `CompletionClient::completion_model()` + `CompletionModel::completion()`.
//! `RigBackend` stores the rig `Client` (provider connection) plus the model
//! name; the underlying HTTP connection is reused across calls.
//!
//! OpenAI completions API choice
//! -----------------------------
//! rig's default `openai::Client` posts to `/responses` (Responses API).
//! We explicitly use `openai::CompletionsClient` here so requests go to the
//! Chat Completions endpoint, which is what OpenAI-compatible gateways and
//! self-hosted shims speak. Work that needs Responses API features (built-in
//! tools, web search) can opt in via a new `ProviderKind` variant.
//!
//! Provider-specific construction
//! ------------------------------
//! - Azure requires `cred.base_url` (the resource endpoint) and accepts an
//!   optional `cred.api_version`; the key is sent as the `api-key` header.
//! - Azure AI Foundry (serverless "Models as a Service") rides Foundry's
//!   OpenAI-compatible v1 surface: `cred.base_url` is the Foundry resource
//!   endpoint (`https://{resource}.services.ai.azure.com`), normalised to
//!   `{endpoint}/openai/v1`, with the API key sent as a bearer token. The
//!   model name travels in the request body, so any catalog model id works;
//!   `cred.api_version` is ignored (the v1 surface is unversioned).
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
use rig_core::completion::CompletionModel;

use super::capabilities::{Capabilities, ProviderKind};
use super::credentials::Credential;
use super::provider::{
    ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError, LlmProvider, MessageRole, Usage,
};

/// Backend that dispatches `LlmProvider` calls to rig provider clients.
pub struct RigBackend {
    kind: ProviderKind,
    model: String,
    inner: Inner,
}

/// One variant per supported provider. Each variant holds the rig provider
/// `Client` (HTTP connection + auth headers); a `CompletionModel` handle is
/// created fresh per `chat()` call (cheap — it borrows the client connection).
pub(crate) enum Inner {
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
    AzureFoundry(rig_core::providers::openai::CompletionsClient),
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
            ProviderKind::AzureFoundry => {
                // Foundry's OpenAI-compatible v1 surface accepts the resource
                // API key as a bearer token and takes the model id in the
                // request body, so rig's OpenAI completions client does the
                // wire work; only the base URL is Foundry-specific.
                let endpoint = cred.base_url.as_deref().ok_or_else(|| {
                    LlmError::Config(
                        "azure-foundry requires base_url, e.g. \
                         https://{resource}.services.ai.azure.com"
                            .to_string(),
                    )
                })?;
                let client = rig_core::providers::openai::CompletionsClient::builder()
                    .api_key(&cred.api_key)
                    .base_url(foundry_openai_base(endpoint))
                    .build()
                    .map_err(|e| LlmError::Transport(format!("azure-foundry client: {e}")))?;
                Inner::AzureFoundry(client)
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

/// Normalise a Foundry resource endpoint to its OpenAI-compatible v1 base.
///
/// Users may supply the bare resource endpoint
/// (`https://{resource}.services.ai.azure.com`) or the full v1 path; both
/// resolve to `{endpoint}/openai/v1` so rig's OpenAI client posts to the
/// right route.
fn foundry_openai_base(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    if trimmed.ends_with("/openai/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/openai/v1")
    }
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Concatenate every `System` message in the request into a single preamble
/// string. Returns `None` if no system message is present (rig's
/// `CompletionRequest::preamble` is optional).
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

/// Default `max_tokens` for Anthropic, which rejects requests without one.
///
/// Deliberately generous: a small default (e.g. 4096) silently truncates rich
/// responses such as multi-card AdaptiveCard flows inlined in a single reply.
/// Callers that want a tighter budget can always pass an explicit
/// `max_tokens`; this constant only applies when none is provided.
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 32_768;

/// Convert a greentic [`ChatRequest`] into rig's provider-agnostic
/// `CompletionRequest`.
///
/// System messages are folded into the preamble; user/assistant/tool turns
/// become rig chat history. Tool results are encoded as rig `User` messages
/// carrying `UserContent::ToolResult` keyed by `tool_call_id` (the same id
/// surfaced by [`map_choice`], so caller-driven tool loops round-trip).
///
/// Consumes the request so message payloads (notably base64 image data, which
/// can be multiple megabytes) move into the rig request instead of being
/// cloned.
fn build_completion_request(
    req: ChatRequest,
    kind: ProviderKind,
) -> Result<rig_core::completion::CompletionRequest, LlmError> {
    use rig_core::message::{
        AssistantContent, ImageMediaType, Message, MimeType, ToolResultContent, UserContent,
    };

    // Build the preamble while the messages are still borrowable; the loop
    // below consumes them.
    let preamble = build_preamble(&req.messages);
    let mut history: Vec<Message> = Vec::new();
    for m in req.messages {
        match m.role {
            MessageRole::System => {} // folded into the preamble
            MessageRole::User => {
                let mut content: Vec<UserContent> = Vec::new();
                if !m.content.is_empty() {
                    content.push(UserContent::text(m.content));
                }
                for img in m.images {
                    let media_type = ImageMediaType::from_mime_type(&img.media_type);
                    content.push(UserContent::image_base64(img.data_base64, media_type, None));
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
                    content.push(AssistantContent::text(m.content));
                }
                for tc in m.tool_calls {
                    content.push(AssistantContent::tool_call(tc.id, tc.name, tc.arguments));
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
                // Providers key tool results to the originating call; an
                // unkeyed result would be silently misattributed, so fail
                // fast instead of sending an empty id.
                let id = m
                    .tool_call_id
                    .ok_or_else(|| LlmError::Parse("tool message missing tool_call_id".into()))?;
                history.push(Message::User {
                    content: rig_core::OneOrMany::one(UserContent::tool_result(
                        id,
                        rig_core::OneOrMany::one(ToolResultContent::text(m.content)),
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
            .into_iter()
            .map(|t| rig_core::completion::ToolDefinition {
                name: t.name,
                description: t.description,
                parameters: t.schema,
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
///
/// rig's generic `CompletionResponse` carries no finish reason, so truncation
/// is inferred from token usage: when the reported `output_tokens` reaches the
/// requested `max_tokens` cap, the response was cut off and the finish reason
/// is [`FinishReason::Length`]. Providers that report no usage
/// (`output_tokens == 0` per rig's `Usage` contract) never report `Length`.
///
/// `model` is the active model id; it is forwarded verbatim into
/// [`Usage::model`] so callers can attribute cost without keeping a separate
/// reference to the backend configuration.
fn map_choice(
    choice: rig_core::OneOrMany<rig_core::message::AssistantContent>,
    usage: rig_core::completion::Usage,
    requested_max_tokens: Option<u64>,
    model: &str,
) -> ChatResponse {
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
                id: tc.call_id.unwrap_or(tc.id),
                name: tc.function.name,
                arguments: tc.function.arguments,
            }),
            // Reasoning / Image parts are not surfaced through ChatResponse.
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
        }
    }
    let truncated = usage.output_tokens > 0
        && requested_max_tokens.is_some_and(|cap| usage.output_tokens >= cap);
    let finish_reason = if !tool_calls.is_empty() {
        FinishReason::ToolCalls
    } else if truncated {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    let token_usage = Usage {
        model: model.to_string(),
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
    };
    ChatResponse {
        content,
        tool_calls,
        finish_reason,
        usage: Some(token_usage),
    }
}

/// Map rig's `CompletionError` onto [`LlmError`], preserving HTTP status
/// codes where rig surfaces them.
fn map_completion_error(e: rig_core::completion::CompletionError) -> LlmError {
    use rig_core::completion::CompletionError;
    use rig_core::http_client;
    match e {
        CompletionError::HttpError(http_client::Error::InvalidStatusCode(status)) => {
            LlmError::Status {
                status: status.as_u16(),
                body: String::new(),
            }
        }
        CompletionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            status,
            body,
        )) => LlmError::Status {
            status: status.as_u16(),
            body,
        },
        CompletionError::HttpError(e) => LlmError::Transport(e.to_string()),
        CompletionError::JsonError(e) => LlmError::Parse(e.to_string()),
        CompletionError::UrlError(e) => LlmError::Config(e.to_string()),
        CompletionError::RequestError(e) => LlmError::Transport(e.to_string()),
        CompletionError::ResponseError(s) => LlmError::Parse(s),
        CompletionError::ProviderError(s) => LlmError::Transport(s),
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

    /// Execute a single completion (text, tool calling, vision) against the
    /// configured provider. Requests carrying tools or images are rejected
    /// up front with `UnsupportedCapability("tools")` / `("vision")` when the
    /// provider's capability matrix does not advertise the feature.
    ///
    /// One call = one completion: when the response carries
    /// [`FinishReason::ToolCalls`], the caller dispatches the tools and
    /// replays the conversation (see the module docs for the loop pattern).
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        let caps = self.capabilities();
        if !req.tools.is_empty() && !caps.tools {
            return Err(LlmError::UnsupportedCapability("tools"));
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) && !caps.vision {
            return Err(LlmError::UnsupportedCapability("vision"));
        }

        let request = build_completion_request(req, self.kind)?;
        // Captured before the request moves into the provider call; needed
        // afterwards to infer truncation from token usage.
        let requested_max_tokens = request.max_tokens;

        // Each provider's `completion_model(model)` returns a different
        // `CompletionModel` type, so the dispatch can't be DRY'd into a
        // helper function (the return type would need to be erased). The
        // macro expands one identical block per provider. NO wildcard arm:
        // a new `Inner` variant must fail to compile here rather than
        // silently miss tool support.
        macro_rules! complete {
            ($client:expr) => {{
                let rig_model = $client.completion_model(self.model.as_str());
                let response = rig_model
                    .completion(request)
                    .await
                    .map_err(map_completion_error)?;
                Ok(map_choice(
                    response.choice,
                    response.usage,
                    requested_max_tokens,
                    self.model.as_str(),
                ))
            }};
        }

        match &self.inner {
            Inner::Openai(client) => complete!(client),
            Inner::Anthropic(client) => complete!(client),
            Inner::Deepseek(client) => complete!(client),
            Inner::Gemini(client) => complete!(client),
            Inner::Cohere(client) => complete!(client),
            Inner::Ollama(client) => complete!(client),
            Inner::Groq(client) => complete!(client),
            Inner::Perplexity(client) => complete!(client),
            Inner::Xai(client) => complete!(client),
            Inner::Azure(client) => complete!(client),
            Inner::AzureFoundry(client) => complete!(client),
            Inner::Mistral(client) => complete!(client),
            Inner::Openrouter(client) => complete!(client),
            Inner::Huggingface(client) => complete!(client),
            Inner::Together(client) => complete!(client),
            Inner::Moonshot(client) => complete!(client),
            Inner::Minimax(client) => complete!(client),
            Inner::Hyperbolic(client) => complete!(client),
            Inner::Galadriel(client) => complete!(client),
            Inner::Mira(client) => complete!(client),
            Inner::Zai(client) => complete!(client),
            Inner::Xiaomimimo(client) => complete!(client),
            Inner::Llamafile(client) => complete!(client),
            #[cfg(feature = "bedrock")]
            Inner::Bedrock(client) => complete!(client),
        }
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
        let r = build_completion_request(tool_request(), ProviderKind::Anthropic).expect("convert");
        assert_eq!(r.preamble.as_deref(), Some("you are helpful"));
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.tools[0].name, "lookup");
        assert_eq!(r.max_tokens, Some(32_768)); // anthropic default
        assert_eq!(r.temperature, Some(0.2f32 as f64));
        assert_eq!(r.chat_history.len(), 3); // user, assistant(tool_call), tool-result
        assert!(matches!(
            r.tool_choice,
            Some(rig_core::message::ToolChoice::Auto)
        ));
    }

    #[test]
    fn non_anthropic_max_tokens_stays_unset() {
        let r = build_completion_request(tool_request(), ProviderKind::Openai).expect("convert");
        assert_eq!(r.max_tokens, None);
    }

    #[test]
    fn tool_call_history_round_trips_through_rig_messages() {
        // rig response with a Completions-API style call (call_id = None,
        // id = "call_9") must surface id "call_9"; replaying that id must
        // land on both the assistant tool_call and the tool_result.
        let r = build_completion_request(tool_request(), ProviderKind::Openai).expect("convert");
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
        let resp = map_choice(
            choice,
            rig_core::completion::Usage::new(),
            None,
            "test-model",
        );
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
        let resp = map_choice(
            choice,
            rig_core::completion::Usage::new(),
            None,
            "test-model",
        );
        assert_eq!(resp.tool_calls[0].id, "call_abc");
    }

    #[test]
    fn maps_text_only_choice_to_stop() {
        let choice = rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("hello"));
        let resp = map_choice(
            choice,
            rig_core::completion::Usage::new(),
            None,
            "test-model",
        );
        assert_eq!(resp.content, "hello");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn output_at_max_tokens_cap_maps_to_length() {
        let choice =
            rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("truncated…"));
        let mut usage = rig_core::completion::Usage::new();
        usage.output_tokens = 4096;
        let resp = map_choice(choice, usage, Some(4096), "test-model");
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    #[test]
    fn unreported_usage_never_maps_to_length() {
        // rig's Usage contract: output_tokens == 0 means the provider did not
        // report usage — never infer truncation from it, even with a cap set.
        let choice = rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("hello"));
        let resp = map_choice(
            choice,
            rig_core::completion::Usage::new(),
            Some(1),
            "test-model",
        );
        assert_eq!(resp.finish_reason, FinishReason::Stop);

        // Below-cap usage with a cap set is also a normal stop.
        let choice = rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("hello"));
        let mut usage = rig_core::completion::Usage::new();
        usage.output_tokens = 10;
        let resp = map_choice(choice, usage, Some(4096), "test-model");
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
        let r = build_completion_request(req, ProviderKind::Openai).expect("convert");
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
    fn unknown_image_mime_type_passes_through_as_none() {
        let mut msg = ChatMessage::user("look at this");
        msg.images.push(ChatImage {
            data_base64: "aGVsbG8=".into(),
            media_type: "image/x-unknown".into(),
        });
        let req = ChatRequest {
            messages: vec![msg],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let r = build_completion_request(req, ProviderKind::Openai).expect("convert");
        let history: Vec<_> = r.chat_history.into_iter().collect();
        match &history[0] {
            rig_core::message::Message::User { content } => {
                let parts: Vec<_> = content.iter().collect();
                match parts[1] {
                    rig_core::message::UserContent::Image(img) => {
                        assert_eq!(img.media_type, None);
                    }
                    other => panic!("expected image content, got {other:?}"),
                }
            }
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn multi_image_user_message_keeps_text_then_images_in_order() {
        let mut msg = ChatMessage::user("two pictures");
        msg.images.push(ChatImage {
            data_base64: "Zmlyc3Q=".into(),
            media_type: "image/png".into(),
        });
        msg.images.push(ChatImage {
            data_base64: "c2Vjb25k".into(),
            media_type: "image/jpeg".into(),
        });
        let req = ChatRequest {
            messages: vec![msg],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let r = build_completion_request(req, ProviderKind::Openai).expect("convert");
        let history: Vec<_> = r.chat_history.into_iter().collect();
        match &history[0] {
            rig_core::message::Message::User { content } => {
                let parts: Vec<_> = content.iter().collect();
                assert_eq!(parts.len(), 3);
                match parts[0] {
                    rig_core::message::UserContent::Text(t) => {
                        assert_eq!(t.text, "two pictures");
                    }
                    other => panic!("expected text content, got {other:?}"),
                }
                let expected = [
                    ("Zmlyc3Q=", rig_core::message::ImageMediaType::PNG),
                    ("c2Vjb25k", rig_core::message::ImageMediaType::JPEG),
                ];
                for (part, (data, media_type)) in parts[1..].iter().zip(expected) {
                    match part {
                        rig_core::message::UserContent::Image(img) => {
                            assert_eq!(img.media_type, Some(media_type));
                            match &img.data {
                                rig_core::message::DocumentSourceKind::Base64(b64) => {
                                    assert_eq!(b64, data);
                                }
                                other => panic!("expected base64 image data, got {other:?}"),
                            }
                        }
                        other => panic!("expected image content, got {other:?}"),
                    }
                }
            }
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn tool_message_without_tool_call_id_is_a_parse_error() {
        let unkeyed_tool_msg = ChatMessage {
            role: MessageRole::Tool,
            content: "{\"answer\":42}".into(),
            images: vec![],
            tool_calls: vec![],
            tool_call_id: None,
        };
        let req = ChatRequest {
            messages: vec![ChatMessage::user("hi"), unkeyed_tool_msg],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let err = build_completion_request(req, ProviderKind::Openai)
            .expect_err("unkeyed tool result must be rejected");
        match err {
            LlmError::Parse(msg) => assert!(msg.contains("tool_call_id"), "message: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
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
        let err = build_completion_request(req, ProviderKind::Openai)
            .expect_err("no chat turns must be an error");
        assert!(matches!(err, LlmError::Parse(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_tools_when_capability_says_no() {
        // Ollama advertises tools = false.
        let b = RigBackend::new(ProviderKind::Ollama, "llama3.2", &dummy_cred()).expect("build");
        let req = ChatRequest {
            messages: vec![ChatMessage::user("hi")],
            tools: vec![ToolDef {
                name: "t".into(),
                description: "d".into(),
                schema: serde_json::json!({}),
            }],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let err = b.chat(req).await.expect_err("must reject");
        assert!(matches!(err, LlmError::UnsupportedCapability("tools")));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_images_when_capability_says_no() {
        // Cohere advertises vision = false.
        let b = RigBackend::new(ProviderKind::Cohere, "command-r", &dummy_cred()).expect("build");
        let mut msg = ChatMessage::user("what is this");
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
        let err = b.chat(req).await.expect_err("must reject");
        assert!(matches!(err, LlmError::UnsupportedCapability("vision")));
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
            ChatMessage::system("be concise"),
            ChatMessage::user("hello"),
            ChatMessage::system("answer in english"),
        ];
        let preamble = build_preamble(&messages).expect("preamble");
        assert_eq!(preamble, "be concise\n\nanswer in english");
    }

    #[test]
    fn preamble_returns_none_without_system() {
        let messages = vec![ChatMessage::user("hello")];
        assert!(build_preamble(&messages).is_none());
    }

    #[test]
    fn multi_turn_history_maps_each_turn() {
        let req = ChatRequest {
            messages: vec![
                ChatMessage::user("first turn"),
                ChatMessage::assistant("first reply"),
                ChatMessage::user("second turn"),
            ],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let r = build_completion_request(req, ProviderKind::Openai).expect("convert");
        assert_eq!(r.chat_history.len(), 3);
        assert!(r.preamble.is_none());
    }

    #[test]
    fn maps_completion_error_variants() {
        use rig_core::completion::CompletionError;
        use rig_core::http_client;

        let status = http::StatusCode::TOO_MANY_REQUESTS;
        match map_completion_error(CompletionError::HttpError(
            http_client::Error::InvalidStatusCodeWithMessage(status, "slow down".into()),
        )) {
            LlmError::Status { status, body } => {
                assert_eq!(status, 429);
                assert_eq!(body, "slow down");
            }
            other => panic!("expected Status, got {other:?}"),
        }
        assert!(matches!(
            map_completion_error(CompletionError::ResponseError("bad json".into())),
            LlmError::Parse(_)
        ));
        assert!(matches!(
            map_completion_error(CompletionError::ProviderError("overloaded".into())),
            LlmError::Transport(_)
        ));
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
            if *kind == ProviderKind::AzureFoundry {
                cred.base_url = Some("https://example.services.ai.azure.com".to_string());
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

    #[test]
    fn azure_foundry_without_endpoint_is_a_config_error() {
        expect_config_err(
            RigBackend::new(ProviderKind::AzureFoundry, "deepseek-v3", &dummy_cred()),
            "azure-foundry without base_url",
        );
    }

    #[test]
    fn foundry_base_url_is_normalised_to_the_openai_v1_surface() {
        assert_eq!(
            foundry_openai_base("https://res.services.ai.azure.com"),
            "https://res.services.ai.azure.com/openai/v1"
        );
        assert_eq!(
            foundry_openai_base("https://res.services.ai.azure.com/"),
            "https://res.services.ai.azure.com/openai/v1"
        );
        assert_eq!(
            foundry_openai_base("https://res.services.ai.azure.com/openai/v1"),
            "https://res.services.ai.azure.com/openai/v1"
        );
        assert_eq!(
            foundry_openai_base("https://res.services.ai.azure.com/openai/v1/"),
            "https://res.services.ai.azure.com/openai/v1"
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
