use crate::domain::contracts::math::{bit_reverse_permute, mod_add, mod_mul, mod_sub};
use ndarray::Array1;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 1024;

/// Computes the NTT using a cached transposed twiddle vector splitting threaded array limits.
pub fn ntt_kernel(data: &mut [u64], twiddles: &[u64], modulus: u64) {
    let n = data.len();
    bit_reverse_permute(data);

    let mut offset = 0;
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let layer_twiddles = &twiddles[offset..offset + half];

        if len >= PAR_THRESHOLD {
            data.par_chunks_mut(len).for_each(|chunk| {
                let (left, right) = chunk.split_at_mut(half);
                for i in 0..half {
                    let u = left[i];
                    let v = mod_mul(right[i], layer_twiddles[i], modulus);
                    left[i] = mod_add(u, v, modulus);
                    right[i] = mod_sub(u, v, modulus);
                }
            });
        } else {
            for chunk in data.chunks_mut(len) {
                let (left, right) = chunk.split_at_mut(half);
                for i in 0..half {
                    let u = left[i];
                    let v = mod_mul(right[i], layer_twiddles[i], modulus);
                    left[i] = mod_add(u, v, modulus);
                    right[i] = mod_sub(u, v, modulus);
                }
            }
        }

        offset += half;
        len *= 2;
    }
}
