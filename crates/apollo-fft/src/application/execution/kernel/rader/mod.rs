pub mod generator;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use num_complex::Complex;

// Rader's algorithm for prime lengths
pub fn rader_fft<F: MixedRadixScalar>(data: &mut [F::Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // 1. Primitive root
    let g = generator::primitive_root(n);

    // 2. Generate permutation indices
    let mut g_pow = vec![0usize; n - 1];
    let mut g_inv_pow = vec![0usize; n - 1];
    
    let mut curr = 1;
    for q in 0..(n - 1) {
        g_pow[q] = curr;
        curr = (curr * g) % n;
    }
    
    let g_inv = mod_inverse(g, n);
    let mut curr_inv = 1;
    for q in 0..(n - 1) {
        g_inv_pow[q] = curr_inv;
        curr_inv = (curr_inv * g_inv) % n;
    }

    // 3. Permute input data
    let mut a = vec![F::complex(0.0, 0.0); n - 1];
    for q in 0..(n - 1) {
        a[q] = data[g_pow[q]];
    }

    // 4. Generate twiddle filter B
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut b = vec![F::complex(0.0, 0.0); n - 1];
    for q in 0..(n - 1) {
        let angle = sign * std::f64::consts::TAU * (g_inv_pow[q] as f64) / (n as f64);
        b[q] = F::complex(angle.cos(), angle.sin());
    }

    // 5. Cyclic convolution of A and B
    // To do exact cyclic convolution of length N-1, we use Bluestein or an exact padded convolution.
    // For now, we delegate to a circular convolution helper.
    let c = circular_convolution::<F>(&a, &b, inverse);

    // 6. Reconstruct output
    let x0 = data[0];
    let mut sum_x = x0;
    for i in 1..n {
        sum_x = sum_x + data[i];
    }
    
    data[0] = sum_x;
    for q in 0..(n - 1) {
        data[g_inv_pow[q]] = x0 + c[q];
    }
}

fn mod_inverse(a: usize, m: usize) -> usize {
    let mut m0 = m as i64;
    let mut y = 0i64;
    let mut x = 1i64;
    let mut a_i64 = a as i64;

    if m == 1 {
        return 0;
    }

    while a_i64 > 1 {
        let q = a_i64 / m0;
        let mut t = m0;
        m0 = a_i64 % m0;
        a_i64 = t;
        t = y;
        y = x - q * y;
        x = t;
    }

    if x < 0 {
        x += m as i64;
    }
    x as usize
}

// Computes exact circular convolution using length L FFT
fn circular_convolution<F: MixedRadixScalar>(
    a: &[F::Complex],
    b: &[F::Complex],
    _inverse: bool,
) -> Vec<F::Complex> {
    let l = a.len();
    
    let mut a_freq = a.to_vec();
    let mut b_freq = b.to_vec();
    
    // forward transform (unnormalized)
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut a_freq);
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut b_freq);
    
    // pointwise multiply
    for i in 0..l {
        a_freq[i] = a_freq[i] * b_freq[i];
    }
    
    // inverse transform (normalized)
    crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(&mut a_freq);
    
    a_freq
}
