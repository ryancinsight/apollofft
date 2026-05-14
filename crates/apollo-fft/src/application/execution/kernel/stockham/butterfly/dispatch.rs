use super::super::avx::fixed_len64_avx_fma;
use super::super::precision::F64StockhamAvxFma;
use super::super::transform::{transform, transform_len4096_four_triples};
use num_complex::Complex64;

pub(crate) unsafe fn forward64_avx_with_scratch(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    if data.len() == 64 {
        fixed_len64_avx_fma(data, scratch, twiddles);
        return;
    }
    if data.len() == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<F64StockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<F64StockhamAvxFma>(data, scratch, twiddles, None);
}
