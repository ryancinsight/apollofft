use crate::application::execution::plan::czt::dimension_1d::CztPlan;
use crate::domain::contracts::error::CztError;
use ndarray::Array1;
use num_complex::Complex64;

/// Computes the forward CZT using the synchronous standard CPU pipeline.
pub fn czt(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    CztPlan::new(input.len(), output_len, a, w)?.forward(input)
}

/// Computes the forward CZT using strict direct $O(NM)$ methods.
pub fn czt_direct(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    CztPlan::new(input.len(), output_len, a, w)?.forward_direct(input)
}
