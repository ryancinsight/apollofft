//! Sliding DFT window configuration.

use crate::domain::contracts::error::{SdftError, SdftResult};
use serde::{Deserialize, Serialize};

/// Validated sliding DFT configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlidingDftConfig {
    window_len: usize,
    bin_count: usize,
}

impl SlidingDftConfig {
    /// Create a validated sliding DFT configuration.
    pub fn new(window_len: usize, bin_count: usize) -> SdftResult<Self> {
        if window_len == 0 {
            return Err(SdftError::EmptyWindow);
        }
        if bin_count == 0 {
            return Err(SdftError::EmptyBinCount);
        }
        if bin_count > window_len {
            return Err(SdftError::BinCountExceedsWindow);
        }
        Ok(Self {
            window_len,
            bin_count,
        })
    }

    /// Return the sliding window length.
    #[must_use]
    pub const fn window_len(self) -> usize {
        self.window_len
    }

    /// Return the tracked frequency bin count.
    #[must_use]
    pub const fn bin_count(self) -> usize {
        self.bin_count
    }
}
