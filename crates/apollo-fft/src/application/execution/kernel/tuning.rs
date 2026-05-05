//! Kernel-level tuning constants.
//!
//! Currently empty: the parallel-dispatch threshold for the 3D separable plan is
//! defined locally in the plan layer (`application::execution::plan::fft::dimension_3d`),
//! where it can be co-located with the data it governs (SSOT).  This module is
//! reserved for future kernel-level constants such as cache-line sizing or
//! per-radix unroll depths.
