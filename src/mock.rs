//! Test mock implementing the new LlmProvider trait at `super::provider`.
//!
//! Lives behind `#[cfg(any(test, feature = "test-mock"))]` like the legacy
//! mock trait in `super::mod`. Phase 2's capability rejection tests
//! (`tests/llm_capability_rejection.rs`, Task 2.5) extend this with
//! scripted-response queues; for Phase 1 the surface is intentionally
//! minimal — capability override + canned text response.

#![cfg(any(test, feature = "test-mock"))]

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
