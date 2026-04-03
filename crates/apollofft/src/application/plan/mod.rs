//! FFT plan implementations.

mod plan1d;
mod plan2d;
mod plan3d;

pub use plan1d::FftPlan1D;
pub use plan2d::FftPlan2D;
pub use plan3d::FftPlan3D;

use num_complex::Complex64;
use std::cell::RefCell;

thread_local! {
    /// Thread-local scratch buffer for 2D and 3D axis transforms.
    pub(super) static AXIS_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local transposed scratch buffer for large strided sweeps.
    pub(super) static AXIS_BUF_2D: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Thread-local real-valued scratch buffer used by the RFFT Z pass.
    pub(super) static RFFT_REAL_BUF: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

