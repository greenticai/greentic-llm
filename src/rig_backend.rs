//! `RigBackend` — dispatches the new [`LlmProvider`] trait to rig 0.35
//! provider clients.
//!
//! Phase 1 scope (Task 1.5): text-only chat via `Agent<M>::prompt`. Tool
//! calling and vision are gated as `LlmError::UnsupportedCapability` for
//! Phase 1 and will be enabled in Phase 3 when `rig::Agent` is wired into
//! `executor.rs`. `chat_stream()` is a deterministic stub returning
//! `UnsupportedCapability` until Phase 3.
//!
//! Architectural note — tools are dynamic
//! ---------------------------------------
//! Our `LlmProvider::chat()` accepts `req.tools: Vec<ToolDef>` per call —
//! tools are runtime-discovered from WASM extensions and may differ between
//! requests. `rig::agent::AgentBuilder::tool(...)` consumes `self`, so we
//! cannot pre-build an `Agent<M>` and add tools later. `RigBackend` therefore
//! stores the rig `Client` (provider connection) plus the model name only, and
//! builds a fresh agent inside `chat()` / `chat_stream()` from the per-request
//! tools. The underlying HTTP connection in `Client` is reused across calls.
//!
//! OpenAI completions API choice
//! -----------------------------
//! rig 0.35's default `openai::Client` posts to `/responses` (Responses API).
//! The legacy designer endpoints (and our wiremock test) speak Chat
//! Completions, so we explicitly use `openai::CompletionsClient` here. Future
//! Phase 3 work that needs Responses API features (built-in tools, web
//! search) can opt in via a new `ProviderKind` variant.

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::capabilities::{Capabilities, ProviderKind};
use super::credentials::Credential;
use super::provider::{
    ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError, LlmProvider, MessageRole,
};

/// Backend that dispatches `LlmProvider` calls to rig 0.35 provider clients.
pub struct RigBackend {
    kind: ProviderKind,
    model: String,
    inner: Inner,
}

impl RigBackend {
    /// Internal accessor used by the `rig_agent` sibling module to drive
    /// per-provider `AgentBuilder` construction without leaking the `Inner`
    /// enum publicly.
    pub(super) fn inner(&self) -> &Inner {
        &self.inner
    }
}

/// One variant per supported provider. Each variant holds the rig provider
/// `Client` (HTTP connection + auth headers); the `Agent<M>` is built fresh
/// per `chat()` call because `AgentBuilder::tool` consumes `self`.
pub(super) enum Inner {
    Openai(rig::providers::openai::CompletionsClient),
    Anthropic(rig::providers::anthropic::Client),
    Deepseek(rig::providers::deepseek::Client),
    Gemini(rig::providers::gemini::Client),
    Cohere(rig::providers::cohere::Client),
    Ollama(rig::providers::ollama::Client),
    Groq(rig::providers::groq::Client),
    Perplexity(rig::providers::perplexity::Client),
    Xai(rig::providers::xai::Client),
}

impl RigBackend {
    /// Construct a backend for the given provider with the supplied model
    /// name and credentials.
    ///
    /// `cred.base_url` overrides the provider's default endpoint where the
    /// underlying rig client supports `ClientBuilder::base_url(...)` (every
    /// supported provider does in rig 0.35). For Ollama the credential's
    /// `api_key` is ignored (rig's ollama transport uses the `Nothing`
    /// API-key marker) and `base_url` defaults to `http://localhost:11434`
    /// when not provided.
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

        let inner = match kind {
            ProviderKind::Openai => {
                build_keyed!(rig::providers::openai::CompletionsClient, Openai, "openai")
            }
            ProviderKind::Anthropic => {
                build_keyed!(rig::providers::anthropic::Client, Anthropic, "anthropic")
            }
            ProviderKind::Deepseek => {
                build_keyed!(rig::providers::deepseek::Client, Deepseek, "deepseek")
            }
            ProviderKind::Gemini => {
                build_keyed!(rig::providers::gemini::Client, Gemini, "gemini")
            }
            ProviderKind::Cohere => {
                build_keyed!(rig::providers::cohere::Client, Cohere, "cohere")
            }
            ProviderKind::Groq => {
                build_keyed!(rig::providers::groq::Client, Groq, "groq")
            }
            ProviderKind::Perplexity => {
                build_keyed!(rig::providers::perplexity::Client, Perplexity, "perplexity")
            }
            ProviderKind::Xai => {
                build_keyed!(rig::providers::xai::Client, Xai, "xai")
            }
            ProviderKind::Ollama => {
                // Ollama has no API key; the rig builder uses the `Nothing`
                // marker. Default base URL is the local daemon.
                let mut builder =
                    rig::providers::ollama::Client::builder().api_key(rig::client::Nothing);
                if let Some(base) = &cred.base_url {
                    builder = builder.base_url(base);
                }
                let client = builder
                    .build()
                    .map_err(|e| LlmError::Transport(format!("ollama client: {e}")))?;
                Inner::Ollama(client)
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
        // Phase 1 scope: text-only chat. Tool calls and vision return a
        // deterministic UnsupportedCapability so callers see a clear error
        // during the migration window. Phase 3 wires both surfaces through
        // rig::Agent in executor.rs.
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
        // macro below expands one identical block per provider — rebuild a
        // fresh agent per call so per-request `tools` (Phase 3) won't fight
        // `AgentBuilder::tool` consuming `self`. The HTTP connection in the
        // underlying `Client` is reused across calls.
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
        };

        Ok(ChatResponse {
            content: text,
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
        })
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError> {
        if !self.capabilities().streaming {
            return Err(LlmError::UnsupportedCapability("streaming"));
        }
        if !req.tools.is_empty() {
            return Err(LlmError::UnsupportedCapability("tool_calling_in_stream"));
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) {
            return Err(LlmError::UnsupportedCapability("vision_in_stream"));
        }

        // Phase 1: streaming is a deterministic stub. Phase 3 will replace
        // this with `Agent::stream_prompt` driving SSE through the existing
        // `AgentEvent` pipeline.
        Err(LlmError::UnsupportedCapability(
            "streaming_not_implemented_phase_1",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatMessage;

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: MessageRole::User,
            content: text.into(),
            images: vec![],
        }
    }

    fn system_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: MessageRole::System,
            content: text.into(),
            images: vec![],
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
            ChatMessage {
                role: MessageRole::Assistant,
                content: "first reply".into(),
                images: vec![],
            },
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
}
