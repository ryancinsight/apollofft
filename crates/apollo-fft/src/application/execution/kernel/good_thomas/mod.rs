//! Good-Thomas Prime Factor Algorithm (PFA)

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

// Extended Euclidean algorithm to find the greatest common divisor and the coefficients of Bézout's identity
fn extended_gcd(a: usize, b: usize) -> (usize, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b as i64 / a as i64) * x, x)
}

fn mod_inverse(a: usize, m: usize) -> usize {
    let (g, x, _) = extended_gcd(a, m);
    assert_eq!(g, 1, "Modular inverse does not exist");
    let res = (x % m as i64 + m as i64) % m as i64;
    res as usize
}

/// Good-Thomas (Prime Factor Algorithm)
pub(crate) fn pfa_fft<F: MixedRadixScalar>(
    data: &mut [F::Complex],
    inverse: bool,
    n1: usize,
    n2: usize,
) {
    let n = n1 * n2;
    debug_assert!(data.len() >= n);

    F::with_pfa_scratch(n, |scratch| {
        // 1. Permute into 2D grid based on input CRT mapping into scratch
        // n_idx = (i1 * N2 + i2 * N1) mod N
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                let n_idx = (i1 * n2 + i2 * n1) % n;
                scratch[i1 * n2 + i2] = data[n_idx];
            }
        }

        // 2. Perform N1-point FFTs along rows (stride N2)
        // Rows are contiguous N2 elements.
        for i1 in 0..n1 {
            let row_start = i1 * n2;
            let row_end = row_start + n2;
            let row_slice = &mut scratch[row_start..row_end];

            if inverse {
                crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(
                    row_slice,
                );
            } else {
                crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(row_slice);
            }
        }

        // 3. Transpose N2xN1 into data buffer to perform contiguous columns
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                data[i2 * n1 + i1] = scratch[i1 * n2 + i2];
            }
        }

        // 4. Perform N2-point FFTs along rows of the transposed matrix (contiguous N1 elements)
        for i2 in 0..n2 {
            let col_start = i2 * n1;
            let col_end = col_start + n1;
            let col_slice = &mut data[col_start..col_end];

            if inverse {
                crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(
                    col_slice,
                );
            } else {
                crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(col_slice);
            }
        }

        // 5. Permute back to 1D based on output CRT mapping
        // k_idx = (k1 * N2 * (N2^-1 mod N1) + k2 * N1 * (N1^-1 mod N2)) mod N
        let inv_n2_n1 = mod_inverse(n2, n1);
        let inv_n1_n2 = mod_inverse(n1, n2);

        // Copy back into scratch since data currently holds the transposed results
        scratch[..n].copy_from_slice(&data[..n]);

        for k1 in 0..n1 {
            for k2 in 0..n2 {
                let k_idx = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
                data[k_idx] = scratch[k2 * n1 + k1]; // Note: scratch is N2xN1 transposed layout
            }
        }
    });
}
