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
//! `ChatMessage::tool_result` per call, and invoke `chat()` again.
//! `chat_stream()` drives rig's `CompletionModel::stream` over the same
//! converted request, adapting rig's chunks into `StreamEvent`s and closing
//! with `Usage` (only when the provider reported it) then `Done`.
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
    ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError, LlmProvider, MessageRole,
    StreamEvent, Usage,
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

    // Resolve tool_choice first: a choice the provider cannot express fails
    // here, before any message conversion or network call.
    let (tools, tool_choice) = resolve_tool_choice(kind, req.tools, req.tool_choice.as_deref())?;

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
        tools: tools
            .into_iter()
            .map(|t| rig_core::completion::ToolDefinition {
                name: t.name,
                description: t.description,
                parameters: t.schema,
            })
            .collect(),
        temperature: req.temperature.map(f64::from),
        max_tokens,
        tool_choice,
        additional_params: None,
        output_schema: None,
    })
}

/// Resolve the caller's `tool_choice` against what the provider's wire
/// protocol can express. Returns the tools to advertise and the rig
/// `ToolChoice` to forward.
///
/// Every provider except Ollama forwards the choice unchanged through
/// [`map_tool_choice`] and rig's provider-specific encoding.
///
/// Ollama has no `tool_choice` on the wire, so the forward path would drop it:
///
/// - rig 0.38.2's `ollama::OllamaCompletionRequest` has no such field. Its
///   `TryFrom` logs `` `tool_choice` not supported for Ollama `` and discards
///   the value (`providers/ollama.rs:447`). Any other `additional_params` key
///   is merged into `options`, not the top level, so it can't be smuggled
///   through there either.
/// - Ollama's own `api.ChatRequest` (native `/api/chat`, which rig targets)
///   and `openai.ChatCompletionRequest` (its `/v1` compatibility shim) have
///   no `tool_choice` field either. Checked at ollama `4f6f739`. Go's JSON
///   decoder drops unknown keys, so sending the field would not fail; it
///   would just be ignored.
///
/// Ollama always behaves like `auto`: when `tools` is present the model may
/// call one or answer in text. Given that, Ollama handles each choice as
/// follows:
///
/// | `tool_choice`      | Ollama behaviour                                  |
/// |--------------------|---------------------------------------------------|
/// | absent / `"auto"`  | tools sent unchanged. Ollama's only mode, so this matches exactly. |
/// | `"none"`           | tools are **not sent**, so the model cannot call one. |
/// | `"required"`       | refused: `UnsupportedCapability`                  |
/// | a function name    | refused: `UnsupportedCapability`                  |
///
/// The last two are refused rather than degraded. A caller asking to force a
/// call expects its next step to receive one; quietly sending a plain `auto`
/// request would let the model answer in text and break that step
/// downstream, far from the cause. To steer Ollama toward one function, send
/// only that function in `tools`.
///
/// For `"auto"` rig receives `None` rather than `ToolChoice::Auto`, so it no
/// longer logs a "not supported" warning for a choice that is honoured.
#[allow(clippy::type_complexity)]
fn resolve_tool_choice(
    kind: ProviderKind,
    tools: Vec<super::provider::ToolDef>,
    choice: Option<&str>,
) -> Result<
    (
        Vec<super::provider::ToolDef>,
        Option<rig_core::message::ToolChoice>,
    ),
    LlmError,
> {
    if kind != ProviderKind::Ollama {
        return Ok((tools, map_tool_choice(choice)));
    }
    match choice {
        None | Some("auto") => Ok((tools, None)),
        Some("none") => Ok((Vec::new(), None)),
        Some("required") => Err(LlmError::UnsupportedCapability("tool_choice=required")),
        Some(_) => Err(LlmError::UnsupportedCapability(
            "tool_choice=<named function>",
        )),
    }
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

/// Map rig's end-of-stream usage onto a [`StreamEvent::Usage`], or `None` when
/// the provider did not report it.
///
/// rig's shared openai-compatible driver ends with `unwrap_or_default()`, so a
/// non-reporting provider yields zeros rather than an absence. This applies the
/// same `output_tokens == 0` convention [`map_choice`] uses for truncation, so
/// the crate has ONE rule for "the provider said nothing", not two.
fn stream_usage_event(reported: rig_core::completion::Usage, model: &str) -> Option<StreamEvent> {
    if reported.output_tokens == 0 {
        return None;
    }
    Some(StreamEvent::Usage(Usage {
        model: model.to_string(),
        input_tokens: reported.input_tokens,
        output_tokens: reported.output_tokens,
    }))
}

/// Adapt a rig streaming response into our [`ChatStream`].
///
/// The terminator order is a contract: `Usage` (when the provider reported it)
/// is emitted BEFORE `Done`, so a consumer that stops reading at `Done` has
/// already seen the usage.
///
/// A tool call arrives from rig complete — `StreamedAssistantContent::ToolCall`
/// is only yielded once rig has assembled the whole call — so one call becomes
/// a `ToolCallStart` immediately followed by a `ToolCallEnd`, and no
/// `StreamEvent::ToolCallArgs` is ever emitted. The correlation id follows
/// [`map_choice`]: `call_id` when the provider supplied one, else `id`.
///
/// The trailing match arm covers rig content this crate deliberately does not
/// surface — `Final` (its usage is read from `rig_stream.response` below),
/// `ToolCallDelta` (superseded by the assembled `ToolCall`), `Reasoning` and
/// `ReasoningDelta` (`map_choice` drops reasoning on the non-streaming path
/// too). It is a wildcard rather than an exhaustive list on purpose: rig's
/// enum is not `#[non_exhaustive]` and this crate takes `rig-core = "0.38"`,
/// so an exhaustive match would turn any new upstream variant into a build
/// break for every consumer. The no-wildcard rule applies to our own `Inner`
/// enum, where a missed arm loses a whole provider silently.
fn map_rig_stream<R>(
    mut rig_stream: rig_core::streaming::StreamingCompletionResponse<R>,
    model: String,
) -> ChatStream
where
    R: Clone + Unpin + rig_core::completion::GetTokenUsage + Send + 'static,
{
    // `token_usage()` resolves through the `GetTokenUsage` bound on `R`; no
    // `use` of the trait is needed (and one would be an unused import).
    use futures_util::StreamExt;
    use rig_core::streaming::StreamedAssistantContent;

    Box::pin(async_stream::stream! {
        let mut finish_reason = FinishReason::Stop;
        while let Some(item) = rig_stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(t)) => {
                    yield Ok(StreamEvent::TextChunk(t.text));
                }
                Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                    let id = tool_call.call_id.unwrap_or(tool_call.id);
                    yield Ok(StreamEvent::ToolCallStart {
                        id: id.clone(),
                        name: tool_call.function.name,
                    });
                    yield Ok(StreamEvent::ToolCallEnd {
                        id,
                        args: tool_call.function.arguments,
                    });
                    finish_reason = FinishReason::ToolCalls;
                }
                Ok(_) => {}
                Err(e) => {
                    yield Err(map_completion_error(e));
                    return;
                }
            }
        }
        if let Some(reported) = rig_stream.response.as_ref().and_then(|r| r.token_usage())
            && let Some(event) = stream_usage_event(reported, &model)
        {
            yield Ok(event);
        }
        yield Ok(StreamEvent::Done { finish_reason });
    })
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

    /// Stream a single completion from the configured provider.
    ///
    /// Same capability gate as [`LlmProvider::chat`]: requests carrying tools
    /// or images are rejected up front when the provider's matrix does not
    /// advertise the feature. The returned stream terminates with
    /// [`StreamEvent::Usage`] (only when the provider reported usage) followed
    /// by [`StreamEvent::Done`]; a provider error mid-stream is yielded as a
    /// single `Err` item and ends the stream, so no `Done` follows it.
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError> {
        let caps = self.capabilities();
        if !req.tools.is_empty() && !caps.tools {
            return Err(LlmError::UnsupportedCapability("tools"));
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) && !caps.vision {
            return Err(LlmError::UnsupportedCapability("vision"));
        }

        let request = build_completion_request(req, self.kind)?;
        let model = self.model.clone();

        // Mirrors `complete!` in `chat`: each provider's `completion_model()`
        // returns a different `CompletionModel` type, so the dispatch expands
        // one identical block per provider. NO wildcard arm: a new `Inner`
        // variant must fail to compile here rather than silently lose
        // streaming.
        macro_rules! stream_with {
            ($client:expr) => {{
                let rig_model = $client.completion_model(self.model.as_str());
                let rig_stream = rig_model
                    .stream(request)
                    .await
                    .map_err(map_completion_error)?;
                Ok(map_rig_stream(rig_stream, model))
            }};
        }

        match &self.inner {
            Inner::Openai(client) => stream_with!(client),
            Inner::Anthropic(client) => stream_with!(client),
            Inner::Deepseek(client) => stream_with!(client),
            Inner::Gemini(client) => stream_with!(client),
            Inner::Cohere(client) => stream_with!(client),
            Inner::Ollama(client) => stream_with!(client),
            Inner::Groq(client) => stream_with!(client),
            Inner::Perplexity(client) => stream_with!(client),
            Inner::Xai(client) => stream_with!(client),
            Inner::Azure(client) => stream_with!(client),
            Inner::AzureFoundry(client) => stream_with!(client),
            Inner::Mistral(client) => stream_with!(client),
            Inner::Openrouter(client) => stream_with!(client),
            Inner::Huggingface(client) => stream_with!(client),
            Inner::Together(client) => stream_with!(client),
            Inner::Moonshot(client) => stream_with!(client),
            Inner::Minimax(client) => stream_with!(client),
            Inner::Hyperbolic(client) => stream_with!(client),
            Inner::Galadriel(client) => stream_with!(client),
            Inner::Mira(client) => stream_with!(client),
            Inner::Zai(client) => stream_with!(client),
            Inner::Xiaomimimo(client) => stream_with!(client),
            Inner::Llamafile(client) => stream_with!(client),
            #[cfg(feature = "bedrock")]
            Inner::Bedrock(client) => stream_with!(client),
        }
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
    fn a_zero_usage_reading_produces_no_usage_event() {
        // rig's contract: output_tokens == 0 means the provider did not report
        // usage. `map_choice` already refuses to infer truncation from it; the
        // stream path must refuse to bill from it for the same reason.
        let reported = rig_core::completion::Usage::new();
        assert!(stream_usage_event(reported, "test-model").is_none());
    }

    #[test]
    fn a_real_usage_reading_produces_a_usage_event() {
        let mut reported = rig_core::completion::Usage::new();
        reported.input_tokens = 12;
        reported.output_tokens = 5;
        let event = stream_usage_event(reported, "test-model").expect("usage reported");
        match event {
            StreamEvent::Usage(u) => {
                assert_eq!(u.model, "test-model");
                assert_eq!(u.input_tokens, 12);
                assert_eq!(u.output_tokens, 5);
            }
            other => panic!("expected a Usage event, got {other:?}"),
        }
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
        // Llamafile advertises tools = false. This was Ollama until Ollama's
        // declaration was corrected — the subject changed, the mechanism being
        // pinned did not: a request carrying tools must be refused HERE, on
        // the matrix, before any provider call is built.
        let b =
            RigBackend::new(ProviderKind::Llamafile, "any-model", &dummy_cred()).expect("build");
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

    /// End-to-end proof that lifting Ollama's `tools` flag exposes real tool
    /// calling rather than a differently-shaped failure — the flag alone only
    /// proves the guard stopped firing.
    ///
    /// `#[ignore]`d: it needs a local Ollama serving a tool-capable model. Run
    /// it with a daemon on 11434 and llama3.2 pulled:
    ///
    /// ```text
    /// cargo test --lib rig_backend::tests::ollama_really_calls_a_tool -- --ignored --nocapture
    /// ```
    ///
    /// Note the base URL carries no `/v1`: this backend appends its own path,
    /// and a doubled prefix 404s.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "needs a local Ollama daemon with a tool-capable model"]
    async fn ollama_really_calls_a_tool() {
        let cred = Credential {
            api_key: String::new(),
            base_url: Some("http://127.0.0.1:11434".to_string()),
            expires_at: None,
            api_version: None,
            aws_profile: None,
        };
        let b = RigBackend::new(ProviderKind::Ollama, "llama3.2:latest", &cred).expect("build");
        let req = ChatRequest {
            messages: vec![ChatMessage::user(
                "What is the weather in Jakarta? Use the tool.",
            )],
            tools: vec![ToolDef {
                name: "get_weather".into(),
                description: "Get the current weather for a city".into(),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            }],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        };
        let res = b.chat(req).await.expect("ollama answers");
        assert_eq!(res.finish_reason, FinishReason::ToolCalls, "{res:?}");
        assert_eq!(
            res.tool_calls.first().map(|c| c.name.as_str()),
            Some("get_weather")
        );
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

    fn tool_request_with_choice(choice: Option<&str>) -> ChatRequest {
        let mut req = tool_request();
        req.tool_choice = choice.map(str::to_string);
        req
    }

    #[test]
    fn ollama_auto_or_absent_keeps_tools_and_forwards_no_choice() {
        for choice in [None, Some("auto")] {
            let r =
                build_completion_request(tool_request_with_choice(choice), ProviderKind::Ollama)
                    .expect("convert");
            assert_eq!(r.tools.len(), 1, "{choice:?}: tools must be advertised");
            // `None`, not `ToolChoice::Auto`: rig would log a false
            // "not supported" warning for a choice that is honoured.
            assert!(r.tool_choice.is_none(), "{choice:?}");
        }
    }

    #[test]
    fn ollama_none_withholds_the_tools() {
        let r =
            build_completion_request(tool_request_with_choice(Some("none")), ProviderKind::Ollama)
                .expect("convert");
        assert!(r.tools.is_empty(), "a model with no tools cannot call one");
        assert!(r.tool_choice.is_none());
        // History, tool calls and tool results included, is untouched.
        assert_eq!(r.chat_history.len(), 3);
    }

    #[test]
    fn ollama_refuses_choices_it_cannot_express() {
        let cases = [
            ("required", "tool_choice=required"),
            ("lookup", "tool_choice=<named function>"),
        ];
        for (choice, capability) in cases {
            let err = build_completion_request(
                tool_request_with_choice(Some(choice)),
                ProviderKind::Ollama,
            )
            .expect_err("must refuse rather than send an un-forced request");
            match err {
                LlmError::UnsupportedCapability(c) => assert_eq!(c, capability, "{choice}"),
                other => panic!("{choice}: unexpected error {other:?}"),
            }
        }
    }

    #[test]
    fn non_ollama_providers_forward_tool_choice_unchanged() {
        // The Ollama handling must not leak into providers whose protocol
        // encodes the field.
        for kind in [ProviderKind::Openai, ProviderKind::Llamafile] {
            let r = build_completion_request(tool_request_with_choice(Some("none")), kind)
                .expect("convert");
            assert_eq!(r.tools.len(), 1, "{kind:?}");
            assert!(matches!(
                r.tool_choice,
                Some(rig_core::message::ToolChoice::None)
            ));
            let r = build_completion_request(tool_request_with_choice(Some("required")), kind)
                .expect("convert");
            assert!(matches!(
                r.tool_choice,
                Some(rig_core::message::ToolChoice::Required)
            ));
        }
    }

    /// Serve exactly one HTTP request on a loopback port with a canned Ollama
    /// `/api/chat` reply, and hand back the request path and JSON body.
    ///
    /// These assertions are about the body rig puts on the wire. The
    /// `CompletionRequest` is only an intermediate form, and rig's Ollama
    /// encoding is where the field used to disappear.
    fn one_shot_ollama() -> (String, std::thread::JoinHandle<(String, serde_json::Value)>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut chunk).expect("read");
                assert!(n > 0, "connection closed before headers completed");
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let content_length = head
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse::<usize>().ok())
                        .flatten()
                })
                .expect("request carries a Content-Length");
            while buf.len() < header_end + content_length {
                let n = stream.read(&mut chunk).expect("read body");
                assert!(n > 0, "connection closed before body completed");
                buf.extend_from_slice(&chunk[..n]);
            }
            let path = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();
            let body: serde_json::Value =
                serde_json::from_slice(&buf[header_end..header_end + content_length])
                    .expect("request body is JSON");

            let reply = serde_json::json!({
                "model": "m",
                "created_at": "2026-09-18T00:00:00Z",
                "message": { "role": "assistant", "content": "ok" },
                "done": true,
                "done_reason": "stop"
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            );
            stream.write_all(response.as_bytes()).expect("write reply");
            (path, body)
        });
        (base, handle)
    }

    async fn ollama_wire_body(choice: Option<&str>) -> (String, serde_json::Value) {
        let (base, server) = one_shot_ollama();
        let cred = Credential {
            api_key: String::new(),
            base_url: Some(base),
            expires_at: None,
            api_version: None,
            aws_profile: None,
        };
        let b = RigBackend::new(ProviderKind::Ollama, "m", &cred).expect("build");
        let res = b
            .chat(tool_request_with_choice(choice))
            .await
            .expect("chat against the loopback server");
        assert_eq!(res.content, "ok");
        server.join().expect("server thread")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ollama_wire_auto_sends_tools_and_no_tool_choice_key() {
        let (path, body) = ollama_wire_body(Some("auto")).await;
        assert_eq!(path, "/api/chat", "rig targets Ollama's native endpoint");
        let tools = body["tools"].as_array().expect("tools on the wire");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "lookup");
        // Ollama has no such field; its absence is the correct encoding.
        assert!(body.get("tool_choice").is_none(), "{body}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ollama_wire_none_sends_no_tools() {
        let (path, body) = ollama_wire_body(Some("none")).await;
        assert_eq!(path, "/api/chat");
        assert!(
            body.get("tools").is_none(),
            "tools must be withheld: {body}"
        );
        assert!(body.get("tool_choice").is_none(), "{body}");
        // The conversation, earlier tool turns included, still goes out.
        assert!(
            body["messages"].as_array().is_some_and(|m| m.len() >= 3),
            "{body}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ollama_required_fails_before_any_request_is_sent() {
        // `dummy_cred()` leaves the default localhost:11434. Whether or not a
        // daemon listens there, reaching the transport would yield a
        // Transport/Status error or a response, never this refusal.
        let b = RigBackend::new(ProviderKind::Ollama, "m", &dummy_cred()).expect("build");
        let err = b
            .chat(tool_request_with_choice(Some("required")))
            .await
            .expect_err("must refuse");
        assert!(matches!(
            err,
            LlmError::UnsupportedCapability("tool_choice=required")
        ));
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
