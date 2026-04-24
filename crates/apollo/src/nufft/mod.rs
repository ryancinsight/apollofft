//! Non-uniform FFT utilities and reusable NUFFT plans.

pub mod dim1;
pub mod dim3;
pub mod math;

pub use dim1::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type2_1d, nufft_type2_1d_fast, NufftPlan1D,
};
pub use dim3::{nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_3d, NufftPlan3D};

/// Default kernel half-width.
pub const DEFAULT_NUFFT_KERNEL_WIDTH: usize = 6;

/// Default oversampling factor.
pub const DEFAULT_NUFFT_OVERSAMPLING: usize = 2;
