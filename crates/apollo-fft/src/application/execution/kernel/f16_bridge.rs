use super::radix2_f16::Cf16;
use half::f16;
use num_complex::Complex32;

#[inline]
pub(crate) fn run_f16_via_f32<F>(data: &mut [Cf16], kernel: F)
where
    F: FnOnce(&mut [Complex32]),
{
    let mut promoted = promote_cf16(data);
    kernel(&mut promoted);
    demote_cf16_into(data, &promoted);
}

#[inline]
pub(crate) fn run_f16_via_f32_with_twiddles<F>(data: &mut [Cf16], twiddles: &[Cf16], kernel: F)
where
    F: FnOnce(&mut [Complex32], &[Complex32]),
{
    let mut promoted = promote_cf16(data);
    let tw32 = promote_cf16(twiddles);
    kernel(&mut promoted, &tw32);
    demote_cf16_into(data, &promoted);
}

#[inline]
fn promote_cf16(input: &[Cf16]) -> Vec<Complex32> {
    input
        .iter()
        .map(|v| Complex32::new(v.re.to_f32(), v.im.to_f32()))
        .collect()
}

#[inline]
fn demote_cf16_into(output: &mut [Cf16], promoted: &[Complex32]) {
    for (dst, src) in output.iter_mut().zip(promoted.iter()) {
        dst.re = f16::from_f32(src.re);
        dst.im = f16::from_f32(src.im);
    }
}
