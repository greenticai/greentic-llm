//! Provider-agnostic LLM trait and request/response types.
//!
//! This is the new abstraction used by Phase 2 routes. It coexists with the
//! legacy `LlmProvider` trait in `super::mod` during the rig migration —
//! routes will switch to this trait in Phase 2, and the legacy trait is
//! deleted in Phase 4.
//!
//! Design notes:
//! - `ChatRequest` intentionally does NOT derive `Serialize`/`Deserialize`;
//!   only the inner message/tool types do. The provider impl is responsible
//!   for translating into the wire format (OpenAI / Anthropic / rig).
//! - `MessageRole::Tool` does not carry a `tool_call_id` field — tool
//!   results are encoded inside `ChatMessage::content` (typically as a JSON
//!   envelope) for now. A richer typed message variant model is out of
//!   scope for this trait.
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
/// formats. See `RigBackend` (Task 1.5) for the canonical translation.
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

/// One-shot chat response.
#[derive(Clone, Debug)]
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
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
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Provider-agnostic LLM contract used by Phase 2 routes.
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
