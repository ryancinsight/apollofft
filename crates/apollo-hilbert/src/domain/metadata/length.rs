//! Validated Hilbert transform signal length.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use serde::{Deserialize, Serialize};

/// Positive signal length for Hilbert analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalLength {
    len: usize,
}

impl SignalLength {
    /// Create a validated signal length.
    pub fn new(len: usize) -> HilbertResult<Self> {
        if len == 0 {
            return Err(HilbertError::EmptySignal);
        }
        Ok(Self { len })
    }

    /// Return the signal length.
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
