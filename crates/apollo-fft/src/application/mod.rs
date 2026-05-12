//! Application-layer execution and orchestration modules.

/// Executable kernels and transform plans.
pub mod execution;
/// Plan-cache orchestration.
pub mod orchestration;

/// FFT utility functions (fftfreq, rfftfreq, fftshift, ifftshift).
pub mod utilities;
