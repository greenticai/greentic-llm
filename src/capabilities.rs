//! Per-provider capability matrix.
//!
//! Each LLM provider exposes a different feature surface (tool calling, vision,
//! streaming, system prompts). Routes consult [`Capabilities`] before invoking
//! a provider — for example, `/api/agent` requires `tools: true`, while a
//! vision-aware chat endpoint requires `vision: true`.
//!
//! The matrix lives in code rather than configuration because it is a property
//! of the provider's API surface, not user-configurable. Adding a new provider
//! requires a new [`ProviderKind`] variant, an entry in [`ProviderKind::all`],
//! a string mapping, and a row in the [`From<ProviderKind>`] impl.

use std::str::FromStr;

/// Feature surface advertised by a provider.
///
/// All fields default to `false`; concrete providers opt in to features they
/// support via the `From<ProviderKind>` impl.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub chat: bool,
    pub tools: bool,
    pub streaming: bool,
    pub vision: bool,
    pub system_prompt: bool,
}

/// Identifier for an LLM provider backend.
///
/// The set is closed at compile time; adding a new provider means adding a new
/// variant here, an entry in [`ProviderKind::all`], a string mapping, and a
/// row in the [`From<ProviderKind>`] impl for [`Capabilities`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Deepseek,
    Gemini,
    Cohere,
    Ollama,
    Groq,
    Perplexity,
    Xai,
}

impl ProviderKind {
    /// Returns every supported provider in canonical order.
    ///
    /// Used by the snapshot test and by CLI/UI surfaces that enumerate the
    /// available backends.
    pub fn all() -> &'static [ProviderKind] {
        &[
            ProviderKind::Openai,
            ProviderKind::Anthropic,
            ProviderKind::Deepseek,
            ProviderKind::Gemini,
            ProviderKind::Cohere,
            ProviderKind::Ollama,
            ProviderKind::Groq,
            ProviderKind::Perplexity,
            ProviderKind::Xai,
        ]
    }

    /// Returns the lowercase canonical string identifier for this provider.
    ///
    /// This is the inverse of [`ProviderKind::from_str`] and matches the
    /// strings expected by CLI flags and configuration files.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::Openai => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Deepseek => "deepseek",
            ProviderKind::Gemini => "gemini",
            ProviderKind::Cohere => "cohere",
            ProviderKind::Ollama => "ollama",
            ProviderKind::Groq => "groq",
            ProviderKind::Perplexity => "perplexity",
            ProviderKind::Xai => "xai",
        }
    }
}

impl FromStr for ProviderKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Ok(ProviderKind::Openai),
            "anthropic" => Ok(ProviderKind::Anthropic),
            "deepseek" => Ok(ProviderKind::Deepseek),
            "gemini" => Ok(ProviderKind::Gemini),
            "cohere" => Ok(ProviderKind::Cohere),
            "ollama" => Ok(ProviderKind::Ollama),
            "groq" => Ok(ProviderKind::Groq),
            "perplexity" => Ok(ProviderKind::Perplexity),
            "xai" => Ok(ProviderKind::Xai),
            other => Err(format!("unknown provider: {other}")),
        }
    }
}

#[cfg(feature = "clap")]
impl clap::ValueEnum for ProviderKind {
    fn value_variants<'a>() -> &'a [Self] {
        Self::all()
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.as_str()))
    }
}

impl From<ProviderKind> for Capabilities {
    /// Maps each provider to its declared feature surface.
    ///
    /// Ollama and xAI use conservative defaults (`tools: false` / `vision:
    /// false`) because those features are model-dependent in practice;
    /// per-model capability detection is a follow-up tracked after the
    /// rig migration completes.
    fn from(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::Openai => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            ProviderKind::Anthropic => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            ProviderKind::Deepseek => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Gemini => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            ProviderKind::Cohere => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Ollama => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Groq => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Perplexity => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Xai => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
        }
    }
}
