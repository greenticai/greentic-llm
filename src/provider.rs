//! Provider-agnostic LLM trait and wire types for the crate.
//!
//! Defines [`LlmProvider`] — the single trait all provider backends implement —
//! together with the request/response/stream types that flow across it.
//! [`RigBackend`][crate::rig_backend::RigBackend] is the production
//! implementation; [`crate::mock::TestLlmProvider`] is the test double.
//!
//! Design notes:
//! - `ChatRequest` intentionally does NOT derive `Serialize`/`Deserialize`;
//!   only the inner message/tool types do. The provider impl is responsible
//!   for translating into the wire format (OpenAI / Anthropic / rig).
//! - Tool linkage is carried on `ChatMessage` itself: assistant turns replay
//!   their requested calls via `tool_calls`, and `MessageRole::Tool` messages
//!   reference the call they answer via `tool_call_id` (the result payload
//!   travels in `content`). Use [`ChatMessage::assistant_with_tool_calls`] and
//!   [`ChatMessage::tool_result`] to construct them.
//! - `ChatStream` is `BoxStream<'static, Result<StreamEvent, LlmError>>`.
//!   `LlmError` is `Send + Sync` because `anyhow::Error` is `Send + Sync`,
//!   so the stream is safe to ship across `.await` points and tasks.

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

use super::capabilities::Capabilities;

/// Role of a chat message in the conversation history.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a chat conversation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    /// Optional vision attachments. Empty `Vec` is canonical when no images
    /// are attached (avoids `Option<Vec<...>>` ambiguity).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ChatImage>,
    /// Tool calls made by this assistant turn. Populated when replaying a
    /// multi-round tool-calling conversation; empty for non-assistant roles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For [`MessageRole::Tool`] messages: id of the tool call this message
    /// answers. Providers that key results to calls require it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Build a plain text message with the given role (no images, no tool
    /// linkage).
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            images: vec![],
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    /// Build a [`MessageRole::System`] text message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(MessageRole::System, content)
    }

    /// Build a [`MessageRole::User`] text message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(MessageRole::User, content)
    }

    /// Build a [`MessageRole::Assistant`] text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(MessageRole::Assistant, content)
    }

    /// Build an assistant turn that requested tool calls. Used when replaying
    /// a multi-round tool-calling conversation back to the provider.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            images: vec![],
            tool_calls,
            tool_call_id: None,
        }
    }

    /// Build a [`MessageRole::Tool`] message carrying the result of the tool
    /// call identified by `tool_call_id`.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            images: vec![],
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// A vision attachment carried with a `ChatMessage`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatImage {
    /// Base64-encoded image bytes (no `data:` prefix).
    pub data_base64: String,
    /// IANA media type (e.g. `image/png`, `image/jpeg`).
    pub media_type: String,
}

/// Tool definition advertised to the provider for function calling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub schema: serde_json::Value,
}

/// One-shot chat request.
///
/// Not `Serialize`/`Deserialize` — providers translate into their own wire
/// formats. See [`crate::rig_backend::RigBackend`] for the canonical translation.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDef>,
    /// Optional tool selection hint. Conventionally `"auto"`, `"required"`,
    /// `"none"`, or a specific tool name; provider-specific semantics apply.
    pub tool_choice: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

/// A tool call requested by the assistant.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Token usage for one completion, normalized across providers.
///
/// Token counts default to `0` when a provider does not report usage; the
/// rig backend maps every provider into this shape. `model` is the active
/// model id that served the request.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// One-shot chat response.
#[derive(Clone, Debug)]
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    /// Token usage for this completion. `None` when the backend cannot report
    /// it (e.g. legacy/streaming paths); the rig backend populates it.
    pub usage: Option<Usage>,
}

/// Why the provider stopped generating tokens.
#[derive(Clone, Debug, PartialEq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Other(String),
}

/// Streaming event emitted from `chat_stream()`.
#[derive(Clone, Debug)]
pub enum StreamEvent {
    /// Incremental assistant text.
    TextChunk(String),
    /// Start marker for a tool call (id + name known, args incoming).
    ToolCallStart { id: String, name: String },
    /// Partial tool-call argument delta (provider-specific JSON fragment).
    ToolCallArgs { id: String, args_delta: String },
    /// Tool call complete with parsed arguments.
    ToolCallEnd { id: String, args: serde_json::Value },
    /// Stream terminated.
    Done { finish_reason: FinishReason },
}

/// Stream of `StreamEvent` values returned by `chat_stream`.
///
/// `'static` lifetime so the stream can outlive the request handler and be
/// passed into Axum SSE responses.
pub type ChatStream = BoxStream<'static, Result<StreamEvent, LlmError>>;

/// Errors returned by `LlmProvider` implementations.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("provider HTTP error: {0}")]
    Transport(String),
    #[error("provider returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("provider response could not be parsed: {0}")]
    Parse(String),
    #[error("capability '{0}' not supported by this provider")]
    UnsupportedCapability(&'static str),
    /// The provider was configured incorrectly (missing endpoint, unsupported
    /// option, or a backend compiled out via cargo features).
    #[error("invalid provider configuration: {0}")]
    Config(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Provider-agnostic LLM contract.
///
/// Implementors must be `Send + Sync` so the trait is usable behind
/// `Arc<dyn LlmProvider>` for shared application state.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Feature surface advertised by this provider — consulted by route
    /// guards (e.g. `/api/agent` requires `tools: true`).
    fn capabilities(&self) -> Capabilities;

    /// Stable provider identifier for telemetry and error messages.
    fn provider_name(&self) -> &'static str;

    /// Active model identifier.
    fn model(&self) -> &str;

    /// One-shot completion. Used by `/api/chat` and the planner phase of
    /// `/api/agent`.
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Streaming completion. Used by `/api/agent/stream`.
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError>;
}

#[cfg(test)]
mod usage_field_tests {
    use super::*;

    #[test]
    fn chat_response_holds_usage() {
        let r = ChatResponse {
            content: "hi".into(),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Some(Usage {
                model: "gpt-4o".into(),
                input_tokens: 12,
                output_tokens: 8,
            }),
        };
        let u = r.usage.as_ref().expect("usage present");
        assert_eq!(u.input_tokens, 12);
        assert_eq!(u.output_tokens, 8);
        assert_eq!(u.model, "gpt-4o");
    }

    #[test]
    fn usage_defaults_to_zero() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert!(u.model.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_tool_fields_default_and_skip() {
        let m: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "User", "content": "hi"
        }))
        .expect("old wire shape still deserializes");
        assert!(m.tool_calls.is_empty());
        assert!(m.tool_call_id.is_none());
        let v = serde_json::to_value(ChatMessage::user("hi")).expect("ser");
        assert!(v.get("tool_calls").is_none());
        assert!(v.get("tool_call_id").is_none());
    }

    #[test]
    fn tool_result_constructor_sets_role_and_id() {
        let m = ChatMessage::tool_result("call_1", "{\"ok\":true}");
        assert_eq!(m.role, MessageRole::Tool);
        assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn assistant_with_tool_calls_carries_calls() {
        let m = ChatMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_2".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
            }],
        );
        assert_eq!(m.role, MessageRole::Assistant);
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].name, "lookup");
        assert!(m.tool_call_id.is_none());
    }
}
