//! Environment-variable-backed [`CredentialSource`].
//!
//! Reads these variables:
//! - `GREENTIC_LLM_PROVIDER`    — the active provider id (e.g. `openai`).
//! - `GREENTIC_LLM_API_KEY`     — the API key for that provider. Optional for
//!   keyless providers (see [`ProviderKind::requires_api_key`]).
//! - `GREENTIC_LLM_BASE_URL`    — optional override for self-hosted gateways;
//!   required for `azure` (the resource endpoint).
//! - `GREENTIC_LLM_API_VERSION` — optional Azure OpenAI REST API version.
//! - `GREENTIC_LLM_AWS_PROFILE` — optional named AWS profile for `bedrock`.
//!
//! Any request for a [`ProviderKind`] that does not match
//! `GREENTIC_LLM_PROVIDER` returns [`CredError::Missing`] — the env source
//! intentionally exposes only one provider at a time.

use super::{CredError, Credential, CredentialSource};
use crate::capabilities::ProviderKind;

pub struct EnvCredentialSource;

#[async_trait::async_trait]
impl CredentialSource for EnvCredentialSource {
    async fn get_credential(&self, provider: ProviderKind) -> Result<Credential, CredError> {
        let active = std::env::var("GREENTIC_LLM_PROVIDER")
            .map_err(|_| CredError::Missing(provider))?
            .parse::<ProviderKind>()
            .map_err(|_| CredError::Missing(provider))?;

        if active != provider {
            return Err(CredError::Missing(provider));
        }

        let api_key = std::env::var("GREENTIC_LLM_API_KEY").unwrap_or_default();

        if api_key.is_empty() && provider.requires_api_key() {
            return Err(CredError::Missing(provider));
        }

        let base_url = std::env::var("GREENTIC_LLM_BASE_URL").ok();
        let api_version = std::env::var("GREENTIC_LLM_API_VERSION").ok();
        let aws_profile = std::env::var("GREENTIC_LLM_AWS_PROFILE").ok();

        Ok(Credential {
            api_key,
            base_url,
            expires_at: None,
            api_version,
            aws_profile,
        })
    }
}
