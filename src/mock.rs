//! Scriptable test double for the [`crate::provider::LlmProvider`] trait.
//!
//! Gated behind `#[cfg(any(test, feature = "test-mock"))]` so it is never
//! compiled into production binaries. Use the `test-mock` Cargo feature to
//! expose [`TestLlmProvider`] and [`TestLlmProviderBuilder`] in integration
//! tests or downstream crates that need a controllable LLM backend.
//!
//! The current surface supports capability override and a canned text
//! response; scripted-response queues can be layered on top as needed.

#![cfg(any(test, feature = "test-mock"))]

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};

use super::capabilities::Capabilities;
use super::provider::{
    ChatRequest, ChatResponse, ChatStream, FinishReason, LlmError, LlmProvider, StreamEvent,
};

/// Test double for the new `LlmProvider` trait. Builder-style construction.
pub struct TestLlmProvider {
    capabilities: Capabilities,
    provider_name: &'static str,
    model: String,
    response_text: String,
    scripted: Mutex<VecDeque<ChatResponse>>,
}

impl Default for TestLlmProvider {
    fn default() -> Self {
        Self {
            capabilities: Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            provider_name: "mock",
            model: "mock-model".into(),
            response_text: "mock response".into(),
            scripted: Mutex::new(VecDeque::new()),
        }
    }
}

pub struct TestLlmProviderBuilder {
    inner: TestLlmProvider,
}

impl TestLlmProviderBuilder {
    pub fn new() -> Self {
        Self {
            inner: TestLlmProvider::default(),
        }
    }

    pub fn capabilities(mut self, caps: Capabilities) -> Self {
        self.inner.capabilities = caps;
        self
    }

    pub fn provider_name(mut self, name: &'static str) -> Self {
        self.inner.provider_name = name;
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = model.into();
        self
    }

    pub fn response_text(mut self, text: impl Into<String>) -> Self {
        self.inner.response_text = text.into();
        self
    }

    pub fn script_response(mut self, response: ChatResponse) -> Self {
        self.inner
            .scripted
            .get_mut()
            .expect("mock script mutex poisoned")
            .push_back(response);
        self
    }

    pub fn build(self) -> TestLlmProvider {
        self.inner
    }
}

impl Default for TestLlmProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmProvider for TestLlmProvider {
    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn provider_name(&self) -> &'static str {
        self.provider_name
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, _req: ChatRequest) -> Result<ChatResponse, LlmError> {
        if let Some(scripted) = self
            .scripted
            .lock()
            .expect("mock script mutex poisoned")
            .pop_front()
        {
            return Ok(scripted);
        }
        Ok(ChatResponse {
            content: self.response_text.clone(),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
        })
    }

    async fn chat_stream(&self, _req: ChatRequest) -> Result<ChatStream, LlmError> {
        if !self.capabilities.streaming {
            return Err(LlmError::UnsupportedCapability("streaming"));
        }
        let text = self.response_text.clone();
        let events = vec![
            Ok(StreamEvent::TextChunk(text)),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
            }),
        ];
        Ok(stream::iter(events).boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;

    fn req() -> ChatRequest {
        ChatRequest {
            messages: vec![],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scripted_responses_are_returned_in_order_then_fall_back() {
        let provider = TestLlmProviderBuilder::new()
            .response_text("fallback")
            .script_response(ChatResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "emit_answers".into(),
                    arguments: serde_json::json!({"name": "first"}),
                }],
                finish_reason: FinishReason::ToolCalls,
            })
            .script_response(ChatResponse {
                content: "second".into(),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
            })
            .build();

        let first = provider.chat(req()).await.unwrap();
        assert_eq!(first.tool_calls[0].name, "emit_answers");

        let second = provider.chat(req()).await.unwrap();
        assert_eq!(second.content, "second");

        let third = provider.chat(req()).await.unwrap();
        assert_eq!(third.content, "fallback");
    }
}
