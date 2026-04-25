//! Real-to-real DCT/DST transform metadata.

use crate::domain::contracts::error::{DctDstError, DctDstResult};
use serde::{Deserialize, Serialize};

/// Supported real-to-real transform families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealTransformKind {
    /// Type-II discrete cosine transform.
    DctII,
    /// Type-III discrete cosine transform.
    DctIII,
    /// Type-II discrete sine transform.
    DstII,
    /// Type-III discrete sine transform.
    DstIII,
}

/// Validated DCT/DST plan configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealTransformConfig {
    len: usize,
    kind: RealTransformKind,
}

impl RealTransformConfig {
    /// Create a validated real-to-real transform configuration.
    pub fn new(len: usize, kind: RealTransformKind) -> DctDstResult<Self> {
        if len == 0 {
            return Err(DctDstError::EmptyLength);
        }
        Ok(Self { len, kind })
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return true when the transform length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return transform kind.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.kind
    }
}
