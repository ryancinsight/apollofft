use crate::application::execution::plan::ntt::dimension_1d::NttPlan;
use crate::domain::contracts::error::NttError;
use ndarray::Array1;

/// Forward NTT convenience wrapper; constructs a default-modulus plan and executes forward.
pub fn ntt(input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
    NttPlan::new(input.len())?.forward(input)
}

/// Inverse NTT convenience wrapper; constructs a default-modulus plan and executes inverse.
pub fn intt(input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
    NttPlan::new(input.len())?.inverse(input)
}
