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
    Azure,
    AzureFoundry,
    Bedrock,
    Mistral,
    Openrouter,
    Huggingface,
    Together,
    Moonshot,
    Minimax,
    Hyperbolic,
    Galadriel,
    Mira,
    Zai,
    Xiaomimimo,
    Llamafile,
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
            ProviderKind::Azure,
            ProviderKind::AzureFoundry,
            ProviderKind::Bedrock,
            ProviderKind::Mistral,
            ProviderKind::Openrouter,
            ProviderKind::Huggingface,
            ProviderKind::Together,
            ProviderKind::Moonshot,
            ProviderKind::Minimax,
            ProviderKind::Hyperbolic,
            ProviderKind::Galadriel,
            ProviderKind::Mira,
            ProviderKind::Zai,
            ProviderKind::Xiaomimimo,
            ProviderKind::Llamafile,
        ]
    }

    /// Returns `true` when the provider needs an API key to authenticate.
    ///
    /// Keyless providers: Ollama and Llamafile talk to a local daemon with no
    /// auth, and Bedrock authenticates through the standard AWS credential
    /// chain (env vars, `~/.aws` profiles, instance roles) rather than an
    /// API key. Credential sources may return an empty `api_key` for these.
    pub fn requires_api_key(&self) -> bool {
        !matches!(
            self,
            ProviderKind::Ollama | ProviderKind::Llamafile | ProviderKind::Bedrock
        )
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
            ProviderKind::Azure => "azure",
            ProviderKind::AzureFoundry => "azure-foundry",
            ProviderKind::Bedrock => "bedrock",
            ProviderKind::Mistral => "mistral",
            ProviderKind::Openrouter => "openrouter",
            ProviderKind::Huggingface => "huggingface",
            ProviderKind::Together => "together",
            ProviderKind::Moonshot => "moonshot",
            ProviderKind::Minimax => "minimax",
            ProviderKind::Hyperbolic => "hyperbolic",
            ProviderKind::Galadriel => "galadriel",
            ProviderKind::Mira => "mira",
            ProviderKind::Zai => "zai",
            ProviderKind::Xiaomimimo => "xiaomimimo",
            ProviderKind::Llamafile => "llamafile",
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
            "azure" => Ok(ProviderKind::Azure),
            "azure-foundry" => Ok(ProviderKind::AzureFoundry),
            "bedrock" => Ok(ProviderKind::Bedrock),
            "mistral" => Ok(ProviderKind::Mistral),
            "openrouter" => Ok(ProviderKind::Openrouter),
            "huggingface" => Ok(ProviderKind::Huggingface),
            "together" => Ok(ProviderKind::Together),
            "moonshot" => Ok(ProviderKind::Moonshot),
            "minimax" => Ok(ProviderKind::Minimax),
            "hyperbolic" => Ok(ProviderKind::Hyperbolic),
            "galadriel" => Ok(ProviderKind::Galadriel),
            "mira" => Ok(ProviderKind::Mira),
            "zai" => Ok(ProviderKind::Zai),
            "xiaomimimo" => Ok(ProviderKind::Xiaomimimo),
            "llamafile" => Ok(ProviderKind::Llamafile),
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
            // Azure OpenAI exposes the same feature surface as OpenAI; the
            // deployment behind the endpoint determines the actual model.
            ProviderKind::Azure => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            // Azure AI Foundry serverless models speak the OpenAI-compatible
            // v1 surface; tool calling is broadly available across the
            // catalog, vision is model-dependent.
            ProviderKind::AzureFoundry => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            // Bedrock's Converse API supports tool use, streaming, and
            // multimodal input across the Anthropic/Nova model families.
            ProviderKind::Bedrock => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: true,
                system_prompt: true,
            },
            ProviderKind::Mistral => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Openrouter => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            // Hugging Face routes to heterogeneous sub-providers; tool and
            // vision support is model-dependent, so default conservatively.
            ProviderKind::Huggingface => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Together => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Moonshot => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Minimax => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            // Hyperbolic, Galadriel, Mira, and Xiaomi MiMo are aggregators or
            // niche hosts where tool calling is model-dependent in practice;
            // conservative defaults until per-model detection lands.
            ProviderKind::Hyperbolic => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Galadriel => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Mira => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Zai => Capabilities {
                chat: true,
                tools: true,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            ProviderKind::Xiaomimimo => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
            // Llamafile is a local OpenAI-compatible server like Ollama;
            // same conservative defaults.
            ProviderKind::Llamafile => Capabilities {
                chat: true,
                tools: false,
                streaming: true,
                vision: false,
                system_prompt: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_round_trips_through_from_str_for_every_provider() {
        for kind in ProviderKind::all() {
            let parsed = kind
                .as_str()
                .parse::<ProviderKind>()
                .expect("canonical id parses back");
            assert_eq!(parsed, *kind);
        }
    }

    #[test]
    fn all_providers_are_chat_capable() {
        for kind in ProviderKind::all() {
            let caps = Capabilities::from(*kind);
            assert!(caps.chat, "{} must support chat", kind.as_str());
            assert!(
                caps.system_prompt,
                "{} must support system prompts",
                kind.as_str()
            );
        }
    }

    #[test]
    fn keyless_providers_are_exactly_local_daemons_and_bedrock() {
        let keyless: Vec<&str> = ProviderKind::all()
            .iter()
            .filter(|k| !k.requires_api_key())
            .map(|k| k.as_str())
            .collect();
        assert_eq!(keyless, ["ollama", "bedrock", "llamafile"]);
    }

    #[test]
    fn unknown_provider_string_is_rejected() {
        assert!("definitely-not-a-provider".parse::<ProviderKind>().is_err());
    }
}
