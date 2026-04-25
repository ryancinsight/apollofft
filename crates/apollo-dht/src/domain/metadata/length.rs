//! Validated Hartley transform length.

use crate::domain::contracts::error::{DhtError, DhtResult};
use serde::{Deserialize, Serialize};

/// Positive DHT transform length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HartleyLength {
    len: usize,
}

impl HartleyLength {
    /// Create a validated transform length.
    pub fn new(len: usize) -> DhtResult<Self> {
        if len == 0 {
            return Err(DhtError::EmptySignal);
        }
        Ok(Self { len })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn get(self) -> usize {
        self.len
    }

    /// Return true when the length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}
