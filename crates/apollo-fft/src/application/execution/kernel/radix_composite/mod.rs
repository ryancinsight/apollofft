#![allow(clippy::uninit_vec)]
use num_complex::Complex;

mod butterfly;
mod cache;

pub use cache::CompositeCache;
use butterfly::stockham_stage;
use crate::application::execution::kernel::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD;
use crate::application::execution::kernel::radix_stage::normalize_inplace;

#[inline]
pub fn forward_inplace_with_radices<F: CompositeCache>(data: &mut [Complex<F>], radices: &[usize]) {
    composite_core_with_radices(data, false, radices);
}

#[inline]
pub fn inverse_inplace_unnorm_with_radices<F: CompositeCache>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    composite_core_with_radices(data, true, radices);
}

#[inline]
pub fn inverse_inplace_with_radices<F: CompositeCache>(data: &mut [Complex<F>], radices: &[usize]) {
    composite_core_with_radices(data, true, radices);
    normalize_inplace(data, F::cast_f64(1.0 / data.len() as f64));
}

fn composite_core_with_radices<F: CompositeCache>(
    data: &mut [Complex<F>],
    inverse: bool,
    radices: &[usize],
) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);
    debug_assert!(radices.iter().all(|r| [2usize, 3, 4, 5, 7, 8].contains(r)));

    let (all_twiddles, stage_offsets) = F::cached_twiddles(inverse, radices);

    F::with_scratch(n, |scratch| {
        let mut src_is_data = true;
        let mut prev_len = 1usize;

        for (stage_idx, &r) in radices.iter().enumerate() {
            let stage_len = prev_len * r;
            let groups = n / stage_len;
            let offset = stage_offsets[stage_idx];
            let stage_twiddles = &all_twiddles[offset..offset + prev_len];
            let use_parallel =
                n >= RADIX_PARALLEL_CHUNK_THRESHOLD && stage_len >= 512 && groups >= 4;

            if src_is_data {
                if use_parallel {
                    stockham_stage::<F, crate::application::execution::policy::ParallelPolicy>(
                        data, scratch, r, prev_len, groups, stage_len, stage_twiddles, inverse,
                    );
                } else {
                    stockham_stage::<F, crate::application::execution::policy::SyncPolicy>(
                        data, scratch, r, prev_len, groups, stage_len, stage_twiddles, inverse,
                    );
                }
            } else {
                if use_parallel {
                    stockham_stage::<F, crate::application::execution::policy::ParallelPolicy>(
                        scratch, data, r, prev_len, groups, stage_len, stage_twiddles, inverse,
                    );
                } else {
                    stockham_stage::<F, crate::application::execution::policy::SyncPolicy>(
                        scratch, data, r, prev_len, groups, stage_len, stage_twiddles, inverse,
                    );
                }
            }

            src_is_data = !src_is_data;
            prev_len = stage_len;
        }

        if !src_is_data {
            data.copy_from_slice(scratch);
        }
    });
}

#[cfg(test)]
#[path = "../tests_radix_composite.rs"]
mod tests;
