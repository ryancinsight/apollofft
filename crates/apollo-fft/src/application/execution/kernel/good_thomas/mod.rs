use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use num_complex::Complex;

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
pub fn pfa_fft<F: MixedRadixScalar>(
    data: &mut [F::Complex],
    inverse: bool,
    n1: usize,
    n2: usize,
) {
    let n = n1 * n2;
    debug_assert_eq!(data.len(), n);
    
    // 1. Permute into 2D grid based on input CRT mapping
    // n_idx = (n1_idx * N2 + n2_idx * N1) mod N
    let mut grid = vec![F::complex(0.0, 0.0); n];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            let n_idx = (i1 * n2 + i2 * n1) % n;
            grid[i1 * n2 + i2] = data[n_idx];
        }
    }

    // 2. Perform N1-point FFTs along columns (stride N2)
    // Actually, our grid is row-major. So rows are N2 contiguous elements, columns are N1 elements spaced by N2.
    // Let's do N2-point FFTs on rows first (contiguous).
    for i1 in 0..n1 {
        let row_start = i1 * n2;
        let row_end = row_start + n2;
        let mut row_buf = grid[row_start..row_end].to_vec();
        
        if inverse {
            crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(&mut row_buf);
        } else {
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut row_buf);
        }
        grid[row_start..row_end].copy_from_slice(&row_buf);
    }

    // 3. Perform N1-point FFTs on columns (stride N2)
    for i2 in 0..n2 {
        let mut col_buf = vec![F::complex(0.0, 0.0); n1];
        for i1 in 0..n1 {
            col_buf[i1] = grid[i1 * n2 + i2];
        }
        
        if inverse {
            crate::application::execution::kernel::mixed_radix::inverse_inplace_unnorm::<F>(&mut col_buf);
        } else {
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut col_buf);
        }
        
        for i1 in 0..n1 {
            grid[i1 * n2 + i2] = col_buf[i1];
        }
    }

    // 4. Permute back to 1D based on output CRT mapping
    // k_idx = (k1 * N2 * (N2^-1 mod N1) + k2 * N1 * (N1^-1 mod N2)) mod N
    let inv_n2_n1 = mod_inverse(n2, n1);
    let inv_n1_n2 = mod_inverse(n1, n2);
    
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let k_idx = (k1 * n2 * inv_n2_n1 + k2 * n1 * inv_n1_n2) % n;
            data[k_idx] = grid[k1 * n2 + k2];
        }
    }
}
