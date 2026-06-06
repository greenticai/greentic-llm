//! Provider-agnostic multi-LLM abstraction for the Greentic platform.
//!
//! Extracted from greentic-designer's `src/ui/llm/` (the post-rig-migration
//! abstraction). Answers exactly one question: "send this chat to provider X,
//! get a response/stream back." Knows nothing about tenants, roles, admin
//! backends, or tool dispatch.

pub mod capabilities;
pub mod credentials;
#[cfg(any(test, feature = "test-mock"))]
pub mod mock;
pub mod provider;
pub mod rig_backend;

pub use capabilities::{Capabilities, ProviderKind};
pub use credentials::{CredError, Credential, CredentialSource, EnvCredentialSource};
pub use provider::{
    ChatImage, ChatMessage, ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError,
    LlmProvider, MessageRole, StreamEvent, ToolCall, ToolDef,
};
pub use rig_backend::RigBackend;
