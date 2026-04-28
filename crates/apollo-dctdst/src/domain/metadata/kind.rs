//! Real-to-real DCT/DST transform metadata.

use crate::domain::contracts::error::{DctDstError, DctDstResult};
use serde::{Deserialize, Serialize};

/// Supported real-to-real transform families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealTransformKind {
    /// Type-I discrete cosine transform (N ≥ 2).
    ///
    /// Definition (Rao & Yip 1990, eq. 2.4):
    /// X_k = x_0 + (−1)^k x_{N−1} + 2 Σ_{n=1}^{N−2} x_n cos(πnk/(N−1))
    ///
    /// Self-inverse: DCT-I(DCT-I(x)) = 2(N−1)·x.
    DctI,
    /// Type-II discrete cosine transform.
    DctII,
    /// Type-III discrete cosine transform.
    DctIII,
    /// Type-IV discrete cosine transform.
    ///
    /// Definition (Rao & Yip 1990, eq. 2.6):
    /// X_k = Σ_{n=0}^{N−1} x_n cos(π(n+½)(k+½)/N)
    ///
    /// Self-inverse: DCT-IV(DCT-IV(x)) = (N/2)·x.
    DctIV,
    /// Type-II discrete sine transform.
    DstII,
    /// Type-III discrete sine transform.
    DstIII,
    /// Type-I discrete sine transform.
    ///
    /// Definition (Rao & Yip 1990, eq. 3.4):
    /// X_k = 2 Σ_{n=0}^{N−1} x_n sin(π(n+1)(k+1)/(N+1))
    ///
    /// Self-inverse: DST-I(DST-I(x)) = 2(N+1)·x.
    DstI,
    /// Type-IV discrete sine transform.
    ///
    /// Definition (Rao & Yip 1990, eq. 3.6):
    /// X_k = Σ_{n=0}^{N−1} x_n sin(π(n+½)(k+½)/N)
    ///
    /// Self-inverse: DST-IV(DST-IV(x)) = (N/2)·x.
    DstIV,
}

/// Validated DCT/DST plan configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealTransformConfig {
    len: usize,
    kind: RealTransformKind,
}

impl RealTransformConfig {
    /// Create a validated real-to-real transform configuration.
    ///
    /// # Errors
    ///
    /// - [`DctDstError::EmptyLength`] when `len == 0`.
    /// - [`DctDstError::UnsupportedLength`] when `kind == DctI` and `len < 2`.
    ///   DCT-I requires N ≥ 2 because the formula references both boundary samples
    ///   x₀ and x_{N−1}, and the frequency grid step π/(N−1) is undefined for N = 1.
    pub fn new(len: usize, kind: RealTransformKind) -> DctDstResult<Self> {
        if len == 0 {
            return Err(DctDstError::EmptyLength);
        }
        if matches!(kind, RealTransformKind::DctI) && len < 2 {
            return Err(DctDstError::UnsupportedLength);
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
