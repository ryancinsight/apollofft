use crate::application::execution::kernel::winograd::traits::WinogradScalar;

pub(crate) trait Dft13Scalar: WinogradScalar {
    const C00: Self;
    const C01: Self;
    const C02: Self;
    const C03: Self;
    const C04: Self;
    const C05: Self;
    const C10: Self;
    const C11: Self;
    const C12: Self;
    const C13: Self;
    const C14: Self;
    const C15: Self;
    const C20: Self;
    const C21: Self;
    const C22: Self;
    const C23: Self;
    const C24: Self;
    const C25: Self;
    const C30: Self;
    const C31: Self;
    const C32: Self;
    const C33: Self;
    const C34: Self;
    const C35: Self;
    const C40: Self;
    const C41: Self;
    const C42: Self;
    const C43: Self;
    const C44: Self;
    const C45: Self;
    const C50: Self;
    const C51: Self;
    const C52: Self;
    const C53: Self;
    const C54: Self;
    const C55: Self;
    const S00: Self;
    const S01: Self;
    const S02: Self;
    const S03: Self;
    const S04: Self;
    const S05: Self;
    const S10: Self;
    const S11: Self;
    const S12: Self;
    const S13: Self;
    const S14: Self;
    const S15: Self;
    const S20: Self;
    const S21: Self;
    const S22: Self;
    const S23: Self;
    const S24: Self;
    const S25: Self;
    const S30: Self;
    const S31: Self;
    const S32: Self;
    const S33: Self;
    const S34: Self;
    const S35: Self;
    const S40: Self;
    const S41: Self;
    const S42: Self;
    const S43: Self;
    const S44: Self;
    const S45: Self;
    const S50: Self;
    const S51: Self;
    const S52: Self;
    const S53: Self;
    const S54: Self;
    const S55: Self;
}

impl Dft13Scalar for f64 {
    const C00: Self = 0.88545602565320991;
    const C01: Self = 0.56806474673115592;
    const C02: Self = 0.12053668025532301;
    const C03: Self = -0.35460488704253545;
    const C04: Self = -0.74851074817110119;
    const C05: Self = -0.97094181742605201;
    const C10: Self = 0.56806474673115592;
    const C11: Self = -0.35460488704253545;
    const C12: Self = -0.97094181742605201;
    const C13: Self = -0.7485107481711013;
    const C14: Self = 0.1205366802553232;
    const C15: Self = 0.88545602565321002;
    const C20: Self = 0.12053668025532301;
    const C21: Self = -0.97094181742605201;
    const C22: Self = -0.3546048870425359;
    const C23: Self = 0.88545602565321002;
    const C24: Self = 0.5680647467311567;
    const C25: Self = -0.74851074817110064;
    const C30: Self = -0.35460488704253545;
    const C31: Self = -0.7485107481711013;
    const C32: Self = 0.88545602565321002;
    const C33: Self = 0.12053668025532369;
    const C34: Self = -0.9709418174260519;
    const C35: Self = 0.56806474673115614;
    const C40: Self = -0.74851074817110119;
    const C41: Self = 0.1205366802553232;
    const C42: Self = 0.5680647467311567;
    const C43: Self = -0.9709418174260519;
    const C44: Self = 0.88545602565320991;
    const C45: Self = -0.35460488704253357;
    const C50: Self = -0.97094181742605201;
    const C51: Self = 0.88545602565321002;
    const C52: Self = -0.74851074817110064;
    const C53: Self = 0.56806474673115614;
    const C54: Self = -0.35460488704253357;
    const C55: Self = 0.12053668025532184;
    const S00: Self = 0.46472317204376851;
    const S01: Self = 0.82298386589365635;
    const S02: Self = 0.99270887409805397;
    const S03: Self = 0.93501624268541483;
    const S04: Self = 0.66312265824079519;
    const S05: Self = 0.23931566428755768;
    const S10: Self = 0.82298386589365635;
    const S11: Self = 0.93501624268541483;
    const S12: Self = 0.23931566428755768;
    const S13: Self = -0.66312265824079497;
    const S14: Self = -0.99270887409805397;
    const S15: Self = -0.4647231720437684;
    const S20: Self = 0.99270887409805397;
    const S21: Self = 0.23931566428755768;
    const S22: Self = -0.93501624268541472;
    const S23: Self = -0.4647231720437684;
    const S24: Self = 0.8229838658936558;
    const S25: Self = 0.66312265824079564;
    const S30: Self = 0.93501624268541483;
    const S31: Self = -0.66312265824079497;
    const S32: Self = -0.4647231720437684;
    const S33: Self = 0.99270887409805386;
    const S34: Self = -0.23931566428755807;
    const S35: Self = -0.82298386589365624;
    const S40: Self = 0.66312265824079519;
    const S41: Self = -0.99270887409805397;
    const S42: Self = 0.8229838658936558;
    const S43: Self = -0.23931566428755807;
    const S44: Self = -0.46472317204376862;
    const S45: Self = 0.93501624268541561;
    const S50: Self = 0.23931566428755768;
    const S51: Self = -0.4647231720437684;
    const S52: Self = 0.66312265824079564;
    const S53: Self = -0.82298386589365624;
    const S54: Self = 0.93501624268541561;
    const S55: Self = -0.99270887409805419;
}

impl Dft13Scalar for f32 {
    const C00: Self = (0.88545602565320991 as f32);
    const C01: Self = (0.56806474673115592 as f32);
    const C02: Self = (0.12053668025532301 as f32);
    const C03: Self = (-0.35460488704253545 as f32);
    const C04: Self = (-0.74851074817110119 as f32);
    const C05: Self = (-0.97094181742605201 as f32);
    const C10: Self = (0.56806474673115592 as f32);
    const C11: Self = (-0.35460488704253545 as f32);
    const C12: Self = (-0.97094181742605201 as f32);
    const C13: Self = (-0.7485107481711013 as f32);
    const C14: Self = (0.1205366802553232 as f32);
    const C15: Self = (0.88545602565321002 as f32);
    const C20: Self = (0.12053668025532301 as f32);
    const C21: Self = (-0.97094181742605201 as f32);
    const C22: Self = (-0.3546048870425359 as f32);
    const C23: Self = (0.88545602565321002 as f32);
    const C24: Self = (0.5680647467311567 as f32);
    const C25: Self = (-0.74851074817110064 as f32);
    const C30: Self = (-0.35460488704253545 as f32);
    const C31: Self = (-0.7485107481711013 as f32);
    const C32: Self = (0.88545602565321002 as f32);
    const C33: Self = (0.12053668025532369 as f32);
    const C34: Self = (-0.9709418174260519 as f32);
    const C35: Self = (0.56806474673115614 as f32);
    const C40: Self = (-0.74851074817110119 as f32);
    const C41: Self = (0.1205366802553232 as f32);
    const C42: Self = (0.5680647467311567 as f32);
    const C43: Self = (-0.9709418174260519 as f32);
    const C44: Self = (0.88545602565320991 as f32);
    const C45: Self = (-0.35460488704253357 as f32);
    const C50: Self = (-0.97094181742605201 as f32);
    const C51: Self = (0.88545602565321002 as f32);
    const C52: Self = (-0.74851074817110064 as f32);
    const C53: Self = (0.56806474673115614 as f32);
    const C54: Self = (-0.35460488704253357 as f32);
    const C55: Self = (0.12053668025532184 as f32);
    const S00: Self = (0.46472317204376851 as f32);
    const S01: Self = (0.82298386589365635 as f32);
    const S02: Self = (0.99270887409805397 as f32);
    const S03: Self = (0.93501624268541483 as f32);
    const S04: Self = (0.66312265824079519 as f32);
    const S05: Self = (0.23931566428755768 as f32);
    const S10: Self = (0.82298386589365635 as f32);
    const S11: Self = (0.93501624268541483 as f32);
    const S12: Self = (0.23931566428755768 as f32);
    const S13: Self = (-0.66312265824079497 as f32);
    const S14: Self = (-0.99270887409805397 as f32);
    const S15: Self = (-0.4647231720437684 as f32);
    const S20: Self = (0.99270887409805397 as f32);
    const S21: Self = (0.23931566428755768 as f32);
    const S22: Self = (-0.93501624268541472 as f32);
    const S23: Self = (-0.4647231720437684 as f32);
    const S24: Self = (0.8229838658936558 as f32);
    const S25: Self = (0.66312265824079564 as f32);
    const S30: Self = (0.93501624268541483 as f32);
    const S31: Self = (-0.66312265824079497 as f32);
    const S32: Self = (-0.4647231720437684 as f32);
    const S33: Self = (0.99270887409805386 as f32);
    const S34: Self = (-0.23931566428755807 as f32);
    const S35: Self = (-0.82298386589365624 as f32);
    const S40: Self = (0.66312265824079519 as f32);
    const S41: Self = (-0.99270887409805397 as f32);
    const S42: Self = (0.8229838658936558 as f32);
    const S43: Self = (-0.23931566428755807 as f32);
    const S44: Self = (-0.46472317204376862 as f32);
    const S45: Self = (0.93501624268541561 as f32);
    const S50: Self = (0.23931566428755768 as f32);
    const S51: Self = (-0.4647231720437684 as f32);
    const S52: Self = (0.66312265824079564 as f32);
    const S53: Self = (-0.82298386589365624 as f32);
    const S54: Self = (0.93501624268541561 as f32);
    const S55: Self = (-0.99270887409805419 as f32);
}

/// In-place prime DFT-13.
///
/// Pair symmetry reduces the 13x13 DFT matrix to six conjugate input pairs.
/// For each `m=1..6`, define `p_m=x[m]+x[13-m]` and
/// `i_m=i*(x[m]-x[13-m])`. Output rows `k` and `13-k` share the cosine
/// projection and differ only by the sine projection sign.
#[inline(never)]
pub(crate) fn dft13_impl<F: Dft13Scalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 13);
    let sign = if INVERSE {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let x5 = data[5];
    let x6 = data[6];
    let x7 = data[7];
    let x8 = data[8];
    let x9 = data[9];
    let x10 = data[10];
    let x11 = data[11];
    let x12 = data[12];
    let p1_re = x1.re + x12.re;
    let p1_im = x1.im + x12.im;
    let d1_re = x1.re - x12.re;
    let d1_im = x1.im - x12.im;
    let i1_re = -d1_im;
    let i1_im = d1_re;
    let p2_re = x2.re + x11.re;
    let p2_im = x2.im + x11.im;
    let d2_re = x2.re - x11.re;
    let d2_im = x2.im - x11.im;
    let i2_re = -d2_im;
    let i2_im = d2_re;
    let p3_re = x3.re + x10.re;
    let p3_im = x3.im + x10.im;
    let d3_re = x3.re - x10.re;
    let d3_im = x3.im - x10.im;
    let i3_re = -d3_im;
    let i3_im = d3_re;
    let p4_re = x4.re + x9.re;
    let p4_im = x4.im + x9.im;
    let d4_re = x4.re - x9.re;
    let d4_im = x4.im - x9.im;
    let i4_re = -d4_im;
    let i4_im = d4_re;
    let p5_re = x5.re + x8.re;
    let p5_im = x5.im + x8.im;
    let d5_re = x5.re - x8.re;
    let d5_im = x5.im - x8.im;
    let i5_re = -d5_im;
    let i5_im = d5_re;
    let p6_re = x6.re + x7.re;
    let p6_im = x6.im + x7.im;
    let d6_re = x6.re - x7.re;
    let d6_im = x6.im - x7.im;
    let i6_re = -d6_im;
    let i6_im = d6_re;
    data[0] = num_complex::Complex::new(
        x0.re + p1_re + p2_re + p3_re + p4_re + p5_re + p6_re,
        x0.im + p1_im + p2_im + p3_im + p4_im + p5_im + p6_im,
    );
    let r1_re = x0.re
        + p1_re * F::C00
        + p2_re * F::C01
        + p3_re * F::C02
        + p4_re * F::C03
        + p5_re * F::C04
        + p6_re * F::C05;
    let r1_im = x0.im
        + p1_im * F::C00
        + p2_im * F::C01
        + p3_im * F::C02
        + p4_im * F::C03
        + p5_im * F::C04
        + p6_im * F::C05;
    let q1_re = sign
        * (i1_re * F::S00
            + i2_re * F::S01
            + i3_re * F::S02
            + i4_re * F::S03
            + i5_re * F::S04
            + i6_re * F::S05);
    let q1_im = sign
        * (i1_im * F::S00
            + i2_im * F::S01
            + i3_im * F::S02
            + i4_im * F::S03
            + i5_im * F::S04
            + i6_im * F::S05);
    data[1] = num_complex::Complex::new(r1_re + q1_re, r1_im + q1_im);
    data[12] = num_complex::Complex::new(r1_re - q1_re, r1_im - q1_im);
    let r2_re = x0.re
        + p1_re * F::C10
        + p2_re * F::C11
        + p3_re * F::C12
        + p4_re * F::C13
        + p5_re * F::C14
        + p6_re * F::C15;
    let r2_im = x0.im
        + p1_im * F::C10
        + p2_im * F::C11
        + p3_im * F::C12
        + p4_im * F::C13
        + p5_im * F::C14
        + p6_im * F::C15;
    let q2_re = sign
        * (i1_re * F::S10
            + i2_re * F::S11
            + i3_re * F::S12
            + i4_re * F::S13
            + i5_re * F::S14
            + i6_re * F::S15);
    let q2_im = sign
        * (i1_im * F::S10
            + i2_im * F::S11
            + i3_im * F::S12
            + i4_im * F::S13
            + i5_im * F::S14
            + i6_im * F::S15);
    data[2] = num_complex::Complex::new(r2_re + q2_re, r2_im + q2_im);
    data[11] = num_complex::Complex::new(r2_re - q2_re, r2_im - q2_im);
    let r3_re = x0.re
        + p1_re * F::C20
        + p2_re * F::C21
        + p3_re * F::C22
        + p4_re * F::C23
        + p5_re * F::C24
        + p6_re * F::C25;
    let r3_im = x0.im
        + p1_im * F::C20
        + p2_im * F::C21
        + p3_im * F::C22
        + p4_im * F::C23
        + p5_im * F::C24
        + p6_im * F::C25;
    let q3_re = sign
        * (i1_re * F::S20
            + i2_re * F::S21
            + i3_re * F::S22
            + i4_re * F::S23
            + i5_re * F::S24
            + i6_re * F::S25);
    let q3_im = sign
        * (i1_im * F::S20
            + i2_im * F::S21
            + i3_im * F::S22
            + i4_im * F::S23
            + i5_im * F::S24
            + i6_im * F::S25);
    data[3] = num_complex::Complex::new(r3_re + q3_re, r3_im + q3_im);
    data[10] = num_complex::Complex::new(r3_re - q3_re, r3_im - q3_im);
    let r4_re = x0.re
        + p1_re * F::C30
        + p2_re * F::C31
        + p3_re * F::C32
        + p4_re * F::C33
        + p5_re * F::C34
        + p6_re * F::C35;
    let r4_im = x0.im
        + p1_im * F::C30
        + p2_im * F::C31
        + p3_im * F::C32
        + p4_im * F::C33
        + p5_im * F::C34
        + p6_im * F::C35;
    let q4_re = sign
        * (i1_re * F::S30
            + i2_re * F::S31
            + i3_re * F::S32
            + i4_re * F::S33
            + i5_re * F::S34
            + i6_re * F::S35);
    let q4_im = sign
        * (i1_im * F::S30
            + i2_im * F::S31
            + i3_im * F::S32
            + i4_im * F::S33
            + i5_im * F::S34
            + i6_im * F::S35);
    data[4] = num_complex::Complex::new(r4_re + q4_re, r4_im + q4_im);
    data[9] = num_complex::Complex::new(r4_re - q4_re, r4_im - q4_im);
    let r5_re = x0.re
        + p1_re * F::C40
        + p2_re * F::C41
        + p3_re * F::C42
        + p4_re * F::C43
        + p5_re * F::C44
        + p6_re * F::C45;
    let r5_im = x0.im
        + p1_im * F::C40
        + p2_im * F::C41
        + p3_im * F::C42
        + p4_im * F::C43
        + p5_im * F::C44
        + p6_im * F::C45;
    let q5_re = sign
        * (i1_re * F::S40
            + i2_re * F::S41
            + i3_re * F::S42
            + i4_re * F::S43
            + i5_re * F::S44
            + i6_re * F::S45);
    let q5_im = sign
        * (i1_im * F::S40
            + i2_im * F::S41
            + i3_im * F::S42
            + i4_im * F::S43
            + i5_im * F::S44
            + i6_im * F::S45);
    data[5] = num_complex::Complex::new(r5_re + q5_re, r5_im + q5_im);
    data[8] = num_complex::Complex::new(r5_re - q5_re, r5_im - q5_im);
    let r6_re = x0.re
        + p1_re * F::C50
        + p2_re * F::C51
        + p3_re * F::C52
        + p4_re * F::C53
        + p5_re * F::C54
        + p6_re * F::C55;
    let r6_im = x0.im
        + p1_im * F::C50
        + p2_im * F::C51
        + p3_im * F::C52
        + p4_im * F::C53
        + p5_im * F::C54
        + p6_im * F::C55;
    let q6_re = sign
        * (i1_re * F::S50
            + i2_re * F::S51
            + i3_re * F::S52
            + i4_re * F::S53
            + i5_re * F::S54
            + i6_re * F::S55);
    let q6_im = sign
        * (i1_im * F::S50
            + i2_im * F::S51
            + i3_im * F::S52
            + i4_im * F::S53
            + i5_im * F::S54
            + i6_im * F::S55);
    data[6] = num_complex::Complex::new(r6_re + q6_re, r6_im + q6_im);
    data[7] = num_complex::Complex::new(r6_re - q6_re, r6_im - q6_im);
}
