//! Pluggable credential source for LLM providers.
//!
//! The [`CredentialSource`] trait abstracts where API keys come from so the
//! runtime can swap implementations without touching call sites. Today we
//! ship [`EnvCredentialSource`] (reads `GREENTIC_LLM_*` env vars). Consumers with richer credential stores (e.g. the designer's admin-managed tenants) implement their own [`CredentialSource`].
//!
//! [`Credential`] deliberately does **not** implement `Serialize` so it
//! cannot be accidentally persisted to disk or sent over the wire. The
//! `api_key` field is zeroized on drop and redacted in `Debug` output.

mod env_source;

pub use env_source::EnvCredentialSource;

use crate::capabilities::ProviderKind;
use chrono::{DateTime, Utc};
use zeroize::ZeroizeOnDrop;

/// Resolved credential for a single provider.
///
/// `api_key` is held as a `String` and zeroized on drop. `base_url` and
/// `expires_at` are skipped from zeroization because they contain no
/// secret material.
#[derive(Clone, ZeroizeOnDrop)]
pub struct Credential {
    pub api_key: String,
    #[zeroize(skip)]
    pub base_url: Option<String>,
    #[zeroize(skip)]
    pub expires_at: Option<DateTime<Utc>>,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credential")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Errors returned by every [`CredentialSource`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum CredError {
    #[error("missing credential for provider {0:?}")]
    Missing(ProviderKind),
    /// Reserved for credential sources with expiring tokens (none ship today).
    #[error("credential expired for provider {0:?}")]
    Expired(ProviderKind),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Pluggable source of provider credentials.
///
/// Implementations resolve a [`ProviderKind`] to a [`Credential`]. The
/// optional `invalidate` hook lets sources backed by a remote store
/// (e.g. a future token-cache implementation) drop cached entries when the runtime
/// observes a 401 from the upstream provider.
#[async_trait::async_trait]
pub trait CredentialSource: Send + Sync {
    async fn get_credential(&self, provider: ProviderKind) -> Result<Credential, CredError>;

    async fn invalidate(&self, _provider: ProviderKind) -> Result<(), CredError> {
        Ok(())
    }
}
