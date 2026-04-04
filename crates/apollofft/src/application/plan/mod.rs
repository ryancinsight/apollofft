//! FFT plan implementations.

mod plan1d;
mod plan2d;
mod plan3d;
mod real_storage;

pub use plan1d::FftPlan1D;
pub use plan2d::FftPlan2D;
pub use plan3d::FftPlan3D;
pub use real_storage::RealFftData;

use num_complex::Complex32;
use num_complex::Complex64;
use std::cell::RefCell;

thread_local! {
    /// Thread-local scratch buffer for 2D and 3D axis transforms.
    pub(super) static AXIS_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local scratch workspace used by `rustfft::Fft::process_with_scratch`.
    pub(super) static AXIS_SCRATCH: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local scratch buffer for `Complex32` axis transforms.
    pub(super) static AXIS_BUF_32: RefCell<Vec<Complex32>> = RefCell::new(Vec::new());
    /// Thread-local scratch workspace for `Complex32` FFT execution.
    pub(super) static AXIS_SCRATCH_32: RefCell<Vec<Complex32>> = RefCell::new(Vec::new());
    /// Thread-local transposed scratch buffer for large strided sweeps.
    pub(super) static AXIS_BUF_2D: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local transposed scratch buffer for `Complex32` strided sweeps.
    pub(super) static AXIS_BUF_2D_32: RefCell<Vec<Complex32>> = RefCell::new(Vec::new());
    /// Thread-local real-valued scratch buffer used by the RFFT Z pass.
    pub(super) static RFFT_REAL_BUF: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    /// Thread-local half-spectrum scratch used by Hermitian-aware inverse paths.
    pub(super) static HALF_SPECTRUM_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local full-volume workspace used to avoid repeated complex-volume allocations.
    pub(super) static VOLUME_COMPLEX_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
}
