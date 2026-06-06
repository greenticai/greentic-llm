//! Provider-agnostic multi-LLM abstraction for the Greentic platform.
//!
//! Extracted from greentic-designer's `src/ui/llm/` (the post-rig-migration
//! abstraction). Answers exactly one question: "send this chat to provider X,
//! get a response/stream back." Knows nothing about tenants, roles, admin
//! backends, or tool dispatch.
