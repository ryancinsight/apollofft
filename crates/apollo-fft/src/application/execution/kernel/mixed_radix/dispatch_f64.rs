use super::*;// f64 dispatch.

/// In-place forward FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if forward_short_winograd(data) {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_fwd_64(data.len());
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::forward_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::forward_inplace_64(data);
    }
}



/// In-place inverse FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(
    data: &mut [Complex64],
    twiddles: Option<&[Complex64]>,
) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, false) {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_inv_64(data.len());
            with_stockham_scratch_64(data.len(), |scratch| {
                <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_unnorm_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, true) {
        return;
    }
    if data.len().is_power_of_two() {
        inverse_inplace_unnorm_64_with_twiddles(data, twiddles);
        normalize_inplace_c64(data, 1.0 / data.len() as f64);
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_64(data);
    }
}

/// In-place forward FFT (unnormalized, f64).
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if forward_short_winograd(data) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_64(data.len());
        forward_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f64).
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, false) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_unnorm_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, true) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_64_with_twiddles(data, None);
    }
}

