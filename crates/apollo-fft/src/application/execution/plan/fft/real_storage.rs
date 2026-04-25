//! Generic real-storage FFT dispatch.

use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use half::f16;
use ndarray::{Array1, Array2, Array3};
use num_complex::{Complex32, Complex64};

/// Real-domain storage type supported by Apollo FFT plans.
///
/// The associated spectrum type matches the storage family chosen by the
/// backend/profile. `f64` uses `Complex64`, while lower-precision storage
/// variants use `Complex32`.
pub trait RealFftData: Clone + Send + Sync + 'static {
    /// Complex spectrum element type produced by this storage type.
    type Spectrum: Clone + Send + Sync + 'static;

    /// Forward 1D transform dispatch.
    fn forward_1d(plan: &FftPlan1D, input: &Array1<Self>) -> Array1<Self::Spectrum>;
    /// Inverse 1D transform dispatch.
    fn inverse_1d(plan: &FftPlan1D, input: &Array1<Self::Spectrum>) -> Array1<Self>;

    /// Forward 2D transform dispatch.
    fn forward_2d(plan: &FftPlan2D, input: &Array2<Self>) -> Array2<Self::Spectrum>;
    /// Inverse 2D transform dispatch.
    fn inverse_2d(plan: &FftPlan2D, input: &Array2<Self::Spectrum>) -> Array2<Self>;

    /// Forward 3D transform dispatch.
    fn forward_3d(plan: &FftPlan3D, input: &Array3<Self>) -> Array3<Self::Spectrum>;
    /// Inverse 3D transform dispatch.
    fn inverse_3d(plan: &FftPlan3D, input: &Array3<Self::Spectrum>) -> Array3<Self>;
}

impl RealFftData for f64 {
    type Spectrum = Complex64;

    fn forward_1d(plan: &FftPlan1D, input: &Array1<Self>) -> Array1<Self::Spectrum> {
        plan.forward(input)
    }

    fn inverse_1d(plan: &FftPlan1D, input: &Array1<Self::Spectrum>) -> Array1<Self> {
        plan.inverse(input)
    }

    fn forward_2d(plan: &FftPlan2D, input: &Array2<Self>) -> Array2<Self::Spectrum> {
        plan.forward(input)
    }

    fn inverse_2d(plan: &FftPlan2D, input: &Array2<Self::Spectrum>) -> Array2<Self> {
        plan.inverse(input)
    }

    fn forward_3d(plan: &FftPlan3D, input: &Array3<Self>) -> Array3<Self::Spectrum> {
        plan.forward(input)
    }

    fn inverse_3d(plan: &FftPlan3D, input: &Array3<Self::Spectrum>) -> Array3<Self> {
        plan.inverse(input)
    }
}

impl RealFftData for f32 {
    type Spectrum = Complex32;

    fn forward_1d(plan: &FftPlan1D, input: &Array1<Self>) -> Array1<Self::Spectrum> {
        plan.forward_f32(input)
    }

    fn inverse_1d(plan: &FftPlan1D, input: &Array1<Self::Spectrum>) -> Array1<Self> {
        plan.inverse_f32(input)
    }

    fn forward_2d(plan: &FftPlan2D, input: &Array2<Self>) -> Array2<Self::Spectrum> {
        plan.forward_f32(input)
    }

    fn inverse_2d(plan: &FftPlan2D, input: &Array2<Self::Spectrum>) -> Array2<Self> {
        plan.inverse_f32(input)
    }

    fn forward_3d(plan: &FftPlan3D, input: &Array3<Self>) -> Array3<Self::Spectrum> {
        plan.forward_f32(input)
    }

    fn inverse_3d(plan: &FftPlan3D, input: &Array3<Self::Spectrum>) -> Array3<Self> {
        plan.inverse_f32(input)
    }
}

impl RealFftData for f16 {
    type Spectrum = Complex32;

    fn forward_1d(plan: &FftPlan1D, input: &Array1<Self>) -> Array1<Self::Spectrum> {
        plan.forward_f16(input)
    }

    fn inverse_1d(plan: &FftPlan1D, input: &Array1<Self::Spectrum>) -> Array1<Self> {
        plan.inverse_f16(input)
    }

    fn forward_2d(plan: &FftPlan2D, input: &Array2<Self>) -> Array2<Self::Spectrum> {
        plan.forward_f16(input)
    }

    fn inverse_2d(plan: &FftPlan2D, input: &Array2<Self::Spectrum>) -> Array2<Self> {
        plan.inverse_f16(input)
    }

    fn forward_3d(plan: &FftPlan3D, input: &Array3<Self>) -> Array3<Self::Spectrum> {
        plan.forward_f16(input)
    }

    fn inverse_3d(plan: &FftPlan3D, input: &Array3<Self::Spectrum>) -> Array3<Self> {
        plan.inverse_f16(input)
    }
}
