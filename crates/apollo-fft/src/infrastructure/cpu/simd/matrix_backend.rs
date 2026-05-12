//! CPU SIMD implementation of the application matrix FFT execution contract.
//!
//! This module is infrastructure-owned because it binds abstract six-step
//! orchestration to concrete CPU batched prime-radix kernels and power-of-two
//! row FFT twiddle tables.

use crate::application::execution::kernel::stockham::StockhamKernel;
use crate::application::execution::plan::fft::matrix_workspace::SixStepF32Kernel;
use crate::infrastructure::cpu::simd::batched::{
    prime_real_sweep, radix5_real_sweep, ForwardRealSweep, InverseRealSweep,
};
use crate::infrastructure::cpu::simd::power_of_two::radix2;
use num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute2f128_ps,
    _mm256_storeu_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
};

const TRANSPOSE_BLOCK: usize = 16;

/// Native-f32 CPU backend for six-step column and row kernels.
///
/// The type carries no state. It exists only as a monomorphization key for the
/// application workspace, preserving static dispatch across the dependency
/// boundary.
pub(crate) struct CpuSixStepF32Kernel;

impl SixStepF32Kernel for CpuSixStepF32Kernel {
    fn build_forward_row_twiddles(n: usize) -> Vec<Complex32> {
        radix2::build_forward_twiddle_table(n)
    }

    fn build_inverse_row_twiddles(n: usize) -> Vec<Complex32> {
        radix2::build_inverse_twiddle_table(n)
    }

    fn column_forward_real<const R: usize>(
        input: &[f32],
        n2: usize,
        re: &mut [f32],
        im: &mut [f32],
    ) {
        run_column_forward_real::<R>(input, n2, re, im);
    }

    fn column_inverse_real<const R: usize>(
        re: &[f32],
        im: &[f32],
        n2: usize,
        output: &mut [f32],
        scale: f32,
    ) {
        run_column_inverse_real::<R>(re, im, n2, output, scale);
    }

    fn load_row_and_twiddle_forward(
        re: &[f32],
        im: &[f32],
        twiddle_re: &[f32],
        twiddle_im: &[f32],
        row: &mut [Complex32],
        radix_idx: usize,
        n2: usize,
    ) {
        load_row_and_twiddle(re, im, twiddle_re, twiddle_im, row, radix_idx, n2);
    }

    fn store_row_and_twiddle_inverse(
        row: &[Complex32],
        re: &mut [f32],
        im: &mut [f32],
        twiddle_re: &[f32],
        twiddle_im: &[f32],
        radix_idx: usize,
        n2: usize,
    ) {
        store_row_and_twiddle(row, re, im, twiddle_re, twiddle_im, radix_idx, n2);
    }

    fn row_forward(row: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]) {
        f32::forward_with_scratch(row, scratch, twiddles);
    }

    fn row_inverse_unnorm(
        row: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        f32::inverse_unnorm_with_scratch(row, scratch, twiddles);
    }

    fn write_transposed_row(
        row: &[Complex32],
        output: &mut [Complex32],
        radix_row: usize,
        radix: usize,
    ) {
        write_transposed_row_unrolled(row, output, radix_row, radix);
    }
}

/// Run the `N2` independent `R`-point column transforms.
///
/// Radix-5 rows are already lane-contiguous in six-step row-major storage, so
/// that path operates directly over row slices. Other prime radices retain the
/// generic batched storage path until each receives a direct sweep kernel.
fn run_column_forward_real<const R: usize>(
    input: &[f32],
    n2: usize,
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    assert_eq!(input.len(), R * n2, "column input length mismatch");
    assert_eq!(out_re.len(), R * n2, "column real output length mismatch");
    assert_eq!(
        out_im.len(),
        R * n2,
        "column imaginary output length mismatch"
    );

    if R == 5 {
        let mut sweep = ForwardRealSweep::checked::<R>(input, n2, out_re, out_im);
        radix5_real_sweep::<false, _>(&mut sweep);
        return;
    }
    if matches!(R, 3 | 7 | 11) {
        let mut sweep = ForwardRealSweep::checked::<R>(input, n2, out_re, out_im);
        prime_real_sweep::<R, false, _>(&mut sweep);
        return;
    }
    unreachable!("unsupported six-step column radix");
}

/// Run the `N2` independent inverse `R`-point columns and apply the final scale.
fn run_column_inverse_real<const R: usize>(
    re_in: &[f32],
    im_in: &[f32],
    n2: usize,
    output: &mut [f32],
    scale: f32,
) {
    assert_eq!(re_in.len(), R * n2, "column real input length mismatch");
    assert_eq!(
        im_in.len(),
        R * n2,
        "column imaginary input length mismatch"
    );
    assert_eq!(output.len(), R * n2, "column output length mismatch");

    if R == 5 {
        let mut sweep = InverseRealSweep::checked::<R>(re_in, im_in, n2, output, scale);
        radix5_real_sweep::<true, _>(&mut sweep);
        return;
    }
    if matches!(R, 3 | 7 | 11) {
        let mut sweep = InverseRealSweep::checked::<R>(re_in, im_in, n2, output, scale);
        prime_real_sweep::<R, true, _>(&mut sweep);
        return;
    }
    unreachable!("unsupported six-step column radix");
}

/// Apply six-step twiddle factors to contiguous planar f32 workspace.
///
/// # Proof
///
/// For each matrix coordinate `(k1,n2)`, the six-step factorization requires
/// the pointwise product `z * exp(sign*2*pi*i*k1*n2/N)`. With
/// `z = xr + i*xi` and `w = wr + i*wi`, multiplication is
/// `(wr*xr - wi*xi) + i(wr*xi + wi*xr)`. The inverse path stores the forward
/// sine table and negates `wi`, which gives
/// `(wr*xr + wi*xi) + i(wr*xi - wi*xr)`. The AVX/FMA implementation evaluates
/// these identities lane-wise over eight independent coordinates; there are no
/// cross-lane dependencies.
fn load_row_and_twiddle(
    re: &[f32],
    im: &[f32],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
    row: &mut [Complex32],
    radix_idx: usize,
    n2: usize,
) {
    let mut start = 0usize;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        start = unsafe {
            load_row_and_twiddle_avx_fma(re, im, twiddle_re, twiddle_im, row, radix_idx, n2)
        };
    }

    for c in start..n2 {
        let idx = radix_idx * n2 + c;
        let wr = twiddle_re[idx];
        let wi = twiddle_im[idx];
        let xr = re[idx];
        let xi = im[idx];
        row[c] = Complex32::new(wr.mul_add(xr, -wi * xi), wr.mul_add(xi, wi * xr));
    }
}

fn store_row_and_twiddle(
    row: &[Complex32],
    re: &mut [f32],
    im: &mut [f32],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
    radix_idx: usize,
    n2: usize,
) {
    let mut start = 0usize;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        start = unsafe {
            store_row_and_twiddle_avx_fma(row, re, im, twiddle_re, twiddle_im, radix_idx, n2)
        };
    }

    for c in start..n2 {
        let idx = radix_idx * n2 + c;
        let wr = twiddle_re[idx];
        let wi = twiddle_im[idx];
        let xr = row[c].re;
        let xi = row[c].im;
        re[idx] = wr.mul_add(xr, wi * xi);
        im[idx] = wr.mul_add(xi, -wi * xr);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn load_row_and_twiddle_avx_fma(
    re: &[f32],
    im: &[f32],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
    row: &mut [Complex32],
    radix_idx: usize,
    n2: usize,
) -> usize {
    let chunks = n2 / 8;
    let lanes = chunks * 8;
    let mut c = 0usize;
    let offset = radix_idx * n2;
    while c < lanes {
        let idx = offset + c;
        unsafe {
            let xr = _mm256_loadu_ps(re.as_ptr().add(idx));
            let xi = _mm256_loadu_ps(im.as_ptr().add(idx));
            let wr = _mm256_loadu_ps(twiddle_re.as_ptr().add(idx));
            let wi = _mm256_loadu_ps(twiddle_im.as_ptr().add(idx));

            let wr_xr = _mm256_mul_ps(wr, xr);
            let wr_xi = _mm256_mul_ps(wr, xi);

            let out_re = _mm256_fnmadd_ps(wi, xi, wr_xr);
            let out_im = _mm256_fmadd_ps(wi, xr, wr_xi);

            // Interleave real and imaginary parts
            let tr1 = _mm256_unpacklo_ps(out_re, out_im);
            let tr2 = _mm256_unpackhi_ps(out_re, out_im);

            // Reorder to match [R0, I0, R1, I1, ...]
            let q1 = _mm256_permute2f128_ps(tr1, tr2, 0x20); // 0b00100000 = lo(tr1), lo(tr2)
            let q2 = _mm256_permute2f128_ps(tr1, tr2, 0x31); // 0b00110001 = hi(tr1), hi(tr2)

            _mm256_storeu_ps(row.as_mut_ptr().add(c) as *mut f32, q1);
            _mm256_storeu_ps(row.as_mut_ptr().add(c + 4) as *mut f32, q2);
        }
        c += 8;
    }
    lanes
}

/// Write one row into transposed natural-order output.
///
/// # Transpose mapping proof
///
/// For fixed `radix_row = k1`, input row position `k2` maps to flat output
/// index `k1 + radix*k2`. The mapping is injective because equal output
/// indices imply `radix*(k2-k2') = 0`, hence `k2 = k2'`. AVX2 has no f32
/// scatter instruction, so this backend uses an unrolled scalar scatter that
/// preserves the same ownership and non-overlap invariant. An AVX-512 scatter
/// implementation can replace this hook without changing application code.
fn write_transposed_row_unrolled(
    row: &[Complex32],
    output: &mut [Complex32],
    radix_row: usize,
    radix: usize,
) {
    debug_assert!(output.len() >= radix * row.len());
    for (block_idx, block) in row.chunks(TRANSPOSE_BLOCK).enumerate() {
        let base = block_idx * TRANSPOSE_BLOCK;
        let mut offset = 0usize;
        while offset + 4 <= block.len() {
            let k0 = base + offset;
            output[radix_row + radix * k0] = block[offset];
            output[radix_row + radix * (k0 + 1)] = block[offset + 1];
            output[radix_row + radix * (k0 + 2)] = block[offset + 2];
            output[radix_row + radix * (k0 + 3)] = block[offset + 3];
            offset += 4;
        }
        while offset < block.len() {
            let k2 = base + offset;
            output[radix_row + radix * k2] = block[offset];
            offset += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn store_row_and_twiddle_avx_fma(
    row: &[Complex32],
    re: &mut [f32],
    im: &mut [f32],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
    radix_idx: usize,
    n2: usize,
) -> usize {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute2f128_ps,
        _mm256_shuffle_ps, _mm256_storeu_ps,
    };
    let chunks = n2 / 8;
    let lanes = chunks * 8;
    let mut c = 0usize;
    let offset = radix_idx * n2;
    while c < lanes {
        let idx = offset + c;
        unsafe {
            // Load 8 Complex32 from row
            let q1 = _mm256_loadu_ps(row.as_ptr().add(c) as *const f32);
            let q2 = _mm256_loadu_ps(row.as_ptr().add(c + 4) as *const f32);

            // Deinterleave [R0, I0, R1, I1, ...] into planar xr, xi
            // A more standard way without shuffle masks:
            let tr1 = _mm256_permute2f128_ps(q1, q2, 0x20); // R0 I0 R1 I1 R4 I4 R5 I5
            let tr2 = _mm256_permute2f128_ps(q1, q2, 0x31); // R2 I2 R3 I3 R6 I6 R7 I7

            let xr = _mm256_shuffle_ps(tr1, tr2, 0x88); // 10001000: R0 R1 R2 R3 ...
            let xi = _mm256_shuffle_ps(tr1, tr2, 0xdd); // 11011101: I0 I1 I2 I3 ...

            let wr = _mm256_loadu_ps(twiddle_re.as_ptr().add(idx));
            let wi = _mm256_loadu_ps(twiddle_im.as_ptr().add(idx));

            let wr_xr = _mm256_mul_ps(wr, xr);
            let wr_xi = _mm256_mul_ps(wr, xi);

            // Inverse: out_re = wr*xr + wi*xi, out_im = wr*xi - wi*xr
            let out_re = _mm256_fmadd_ps(wi, xi, wr_xr);
            let out_im = _mm256_fnmadd_ps(wi, xr, wr_xi);

            _mm256_storeu_ps(re.as_mut_ptr().add(idx), out_re);
            _mm256_storeu_ps(im.as_mut_ptr().add(idx), out_im);
        }
        c += 8;
    }
    lanes
}
#[cfg(test)]
mod tests {
    #[test]
    fn six_step_cpu_backend_rows_use_stockham_autosort_scratch() {
        let source = include_str!("matrix_backend.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        assert!(
            production.contains("f32::forward_with_scratch")
                && production.contains("f32::inverse_unnorm_with_scratch"),
            "six-step row routing must use Stockham autosort with plan-owned scratch"
        );
    }

    #[test]
    fn six_step_prime_columns_use_direct_sweep_kernels() {
        let source = include_str!("matrix_backend.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        for required in [
            "radix5_real_sweep::<false, _>",
            "radix5_real_sweep::<true, _>",
            "prime_real_sweep::<R, false, _>",
            "prime_real_sweep::<R, true, _>",
            "matches!(R, 3 | 7 | 11)",
        ] {
            assert!(
                production.contains(required),
                "six-step prime column routing must use `{required}`"
            );
        }
        for prohibited in [
            "radix5_forward_real_sweep",
            "radix5_inverse_real_sweep",
            "prime_forward_real_sweep",
            "prime_inverse_real_sweep",
        ] {
            assert!(
                !production.contains(prohibited),
                "six-step prime column routing must not retain direction-specific `{prohibited}`"
            );
        }
    }

    #[test]
    fn real_sweep_avx_loaders_use_contiguous_storage_pointers() {
        for (name, source) in [
            ("prime", include_str!("batched/prime.rs")),
            ("radix5", include_str!("batched/radix5.rs")),
        ] {
            assert!(
                source.contains("_mm256_loadu_ps(sweep.re_ptr(point, col))"),
                "{name} real sweep must load real lanes directly from storage pointers"
            );
            assert!(
                source.contains("_mm256_loadu_ps(sweep.im_ptr(point, col))"),
                "{name} real sweep must load imaginary lanes directly from storage pointers"
            );
            assert!(
                !source.contains("sweep.load_re(point, col + 7)"),
                "{name} real sweep must not rebuild AVX lanes through scalar stack gathers"
            );
        }
    }
}
