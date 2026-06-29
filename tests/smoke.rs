//! Crate-surface smoke test: the public API is sufficient to implement and
//! drive an LlmProvider without designer-internal types.
#![cfg(feature = "test-mock")]

use greentic_llm::mock::TestLlmProviderBuilder;
use greentic_llm::{ChatMessage, ChatRequest, FinishReason, LlmProvider};

#[tokio::test(flavor = "current_thread")]
async fn mock_provider_round_trip() {
    let provider = TestLlmProviderBuilder::new()
        .response_text("hello from mock")
        .build();
    let response = provider
        .chat(ChatRequest {
            messages: vec![ChatMessage::user("hi")],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
        })
        .await
        .expect("mock chat succeeds");
    assert_eq!(response.content, "hello from mock");
    assert_eq!(response.finish_reason, FinishReason::Stop);
}
