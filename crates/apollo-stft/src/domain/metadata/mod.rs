//! STFT metadata: validated configuration for the short-time Fourier transform.

use crate::domain::contracts::error::{StftError, StftResult};
use serde::{Deserialize, Serialize};

/// Validated STFT configuration: frame length and hop size.
///
/// # Invariants
/// `frame_len > 0`, `hop_len > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StftConfig {
    frame_len: usize,
    hop_len: usize,
}

impl StftConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    /// Returns `Err(StftError::InvalidFrameLength)` if either length is zero.
    pub fn new(frame_len: usize, hop_len: usize) -> StftResult<Self> {
        if frame_len == 0 {
            return Err(StftError::EmptyFrameLength);
        }
        if hop_len == 0 {
            return Err(StftError::EmptyHopSize);
        }
        Ok(Self { frame_len, hop_len })
    }

    /// Return the frame length.
    pub const fn frame_len(self) -> usize {
        self.frame_len
    }

    /// Return the hop length.
    pub const fn hop_len(self) -> usize {
        self.hop_len
    }
}
