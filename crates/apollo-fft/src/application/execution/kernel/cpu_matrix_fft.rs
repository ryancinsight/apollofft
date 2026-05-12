//! Application-owned CPU adapter for matrix-factorized FFT plans.
//!
//! `SixStepF32Workspace` remains the orchestration object and is generic over
//! its kernel contract. This adapter selects the concrete CPU implementation at
//! the kernel boundary, keeping plan files free of direct infrastructure
//! imports while preserving monomorphized static dispatch.

use crate::application::execution::plan::fft::matrix_workspace::SixStepF32Workspace;
use crate::infrastructure::cpu::simd::matrix_backend::CpuSixStepF32Kernel;
use num_complex::Complex32;

/// Native-f32 CPU six-step workspace selected by application planning.
pub(crate) struct CpuSixStepF32Plan {
    inner: SixStepF32Workspace<CpuSixStepF32Kernel>,
}

impl CpuSixStepF32Plan {
    /// Create a CPU six-step plan workspace for `N = R * 2^m`.
    #[inline]
    pub(crate) fn new(n: usize) -> Option<Self> {
        SixStepF32Workspace::<CpuSixStepF32Kernel>::new(n).map(|inner| Self { inner })
    }

    /// Execute an unnormalized forward real-to-complex FFT.
    #[inline]
    pub(crate) fn forward_real(&mut self, input: &[f32], output: &mut [Complex32]) {
        self.inner.forward_real(input, output);
    }

    /// Execute a normalized inverse complex-to-real FFT.
    #[inline]
    pub(crate) fn inverse_real(&mut self, input: &[Complex32], output: &mut [f32]) {
        self.inner.inverse_real(input, output);
    }
}
