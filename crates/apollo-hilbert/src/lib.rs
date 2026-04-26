#![warn(missing_docs)]
//! Hilbert transform and analytic-signal plans for Apollo.
//!
//! The discrete Hilbert transform shifts positive-frequency components by
//! `-pi / 2` and negative-frequency components by `+pi / 2`. Combining the
//! original real signal with this quadrature component yields the analytic
//! signal `z[n] = x[n] + i H{x}[n]`.
//!
//! This crate owns Hilbert-domain contracts, analytic-signal storage,
//! frequency-domain masking kernels, and value-semantic verification. The
//! implementation uses Apollo-owned direct DFT/IDFT kernels and does not
//! depend on an external FFT implementation.

/// Application-layer Hilbert plans.
pub mod application;
/// Domain contracts, metadata, and signal storage.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::hilbert::{HilbertPlan, HilbertStorage};
pub use domain::contracts::error::{HilbertError, HilbertResult};
pub use domain::metadata::length::SignalLength;
pub use domain::signal::analytic::AnalyticSignal;
pub use infrastructure::kernel::direct::{analytic_signal, hilbert_transform};
