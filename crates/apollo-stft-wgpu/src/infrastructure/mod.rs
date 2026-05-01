//! WGPU infrastructure for the STFT backend.

/// Bluestein/Chirp-Z resources for non-power-of-two `frame_len`.
pub(crate) mod chirp;
/// Pre-allocated GPU buffer sets for reusable dispatch.
pub mod buffers;
/// WGPU device acquisition and backend orchestration.
pub mod device;
/// GPU compute kernel for the forward STFT.
pub mod kernel;
