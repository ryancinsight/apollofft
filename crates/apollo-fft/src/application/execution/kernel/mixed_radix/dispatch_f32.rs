use super::*;// f32 dispatch.

/// In-place forward FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if forward_short_winograd(data) {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_fwd_32(data.len());
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::forward_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::forward_inplace_32(data);
    }
}

/// In-place inverse FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(
    data: &mut [Complex32],
    twiddles: Option<&[Complex32]>,
) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, false) {
        return;
    }
    if data.len().is_power_of_two() {
        if let Some(tw) = twiddles {
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw);
            });
        } else {
            let tw = cached_twiddle_inv_32(data.len());
            with_stockham_scratch_32(data.len(), |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
            });
        }
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_unnorm_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_unnorm_32(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, true) {
        return;
    }
    if data.len().is_power_of_two() {
        inverse_inplace_unnorm_32_with_twiddles(data, twiddles);
        normalize_inplace_c32(data, 1.0 / data.len() as f32);
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                radix_composite::inverse_inplace_with_radices(data, &radices);
                return;
            }
        }
        bluestein::inverse_inplace_32(data);
    }
}

/// In-place forward FFT (unnormalized, f32).
#[inline]
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if forward_short_winograd(data) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_32(data.len());
        with_stockham_scratch_32(data.len(), |scratch| {
            <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, tw.as_ref());
        });
    } else {
        forward_inplace_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f32).
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, false) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_unnorm_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if inverse_short_winograd(data, true) {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_32_with_twiddles(data, None);
    }
}

