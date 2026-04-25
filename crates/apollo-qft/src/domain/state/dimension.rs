//! Quantum state vector dimension contract.

use crate::domain::contracts::error::{QftError, QftResult};
use serde::{Deserialize, Serialize};

/// Validate a QFT length contract.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

/// Validated quantum amplitude vector dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantumStateDimension {
    n: usize,
}

impl QuantumStateDimension {
    /// Create a validated quantum state dimension.
    pub fn new(n: usize) -> QftResult<Self> {
        if !is_valid_length(n) {
            return Err(QftError::EmptyLength);
        }
        Ok(Self { n })
    }

    /// Return the state dimension.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return true when the dimension is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }
}
