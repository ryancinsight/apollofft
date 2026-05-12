//! CPU SIMD execution components.

/// Batched short-DFT kernels that vectorize across independent instances.
pub mod batched;
/// CPU implementation of application-owned matrix FFT kernel contracts.
pub(crate) mod matrix_backend;
/// Power-of-two FFT kernels.
pub mod power_of_two;
