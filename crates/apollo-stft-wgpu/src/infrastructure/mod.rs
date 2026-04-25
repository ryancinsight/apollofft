//! WGPU infrastructure for the STFT backend.

/// WGPU device acquisition and backend orchestration.
pub mod device;
/// GPU compute kernel for the forward STFT.
pub mod kernel;
