//! Real-input column sweep storage adapters.

static ZERO_IMAGINARY_LANES: [f32; 8] = [0.0; 8];

/// Storage contract for monomorphized real-sweep kernels.
///
/// `INVERSE = false` reads row-major real samples and writes planar complex
/// workspace. `INVERSE = true` reads planar complex workspace and writes scaled
/// row-major real samples. The direction is a const parameter so inactive
/// adapter behavior is eliminated per monomorphized call site.
pub(crate) trait RealSweep<const INVERSE: bool, const R: usize> {
    fn n2(&self) -> usize;
    fn re_ptr(&self, point: usize, col: usize) -> *const f32;
    fn im_ptr(&self, point: usize, col: usize) -> *const f32;
    fn load_re(&self, point: usize, col: usize) -> f32;
    fn load_im(&self, point: usize, col: usize) -> f32;
    fn store(&mut self, point: usize, col: usize, re: f32, im: f32);

    #[cfg(target_arch = "x86_64")]
    unsafe fn store_avx(
        &mut self,
        point: usize,
        col: usize,
        re: std::arch::x86_64::__m256,
        im: std::arch::x86_64::__m256,
    );
}

pub(crate) struct ForwardRealSweep<'a> {
    input: &'a [f32],
    n2: usize,
    out_re: &'a mut [f32],
    out_im: &'a mut [f32],
}

impl<'a> ForwardRealSweep<'a> {
    pub(crate) fn checked<const R: usize>(
        input: &'a [f32],
        n2: usize,
        out_re: &'a mut [f32],
        out_im: &'a mut [f32],
    ) -> Self {
        assert_eq!(input.len(), R * n2, "real sweep input length mismatch");
        assert_eq!(
            out_re.len(),
            R * n2,
            "real sweep real output length mismatch"
        );
        assert_eq!(
            out_im.len(),
            R * n2,
            "real sweep imaginary output length mismatch"
        );
        Self {
            input,
            n2,
            out_re,
            out_im,
        }
    }
}

impl<const R: usize> RealSweep<false, R> for ForwardRealSweep<'_> {
    #[inline(always)]
    fn n2(&self) -> usize {
        self.n2
    }

    #[inline(always)]
    fn re_ptr(&self, point: usize, col: usize) -> *const f32 {
        self.input.as_ptr().wrapping_add(point * self.n2 + col)
    }

    #[inline(always)]
    fn im_ptr(&self, _point: usize, _col: usize) -> *const f32 {
        ZERO_IMAGINARY_LANES.as_ptr()
    }

    #[inline(always)]
    fn load_re(&self, point: usize, col: usize) -> f32 {
        self.input[point * self.n2 + col]
    }

    #[inline(always)]
    fn load_im(&self, _point: usize, _col: usize) -> f32 {
        0.0
    }

    #[inline(always)]
    fn store(&mut self, point: usize, col: usize, re: f32, im: f32) {
        let idx = point * self.n2 + col;
        self.out_re[idx] = re;
        self.out_im[idx] = im;
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn store_avx(
        &mut self,
        point: usize,
        col: usize,
        re: std::arch::x86_64::__m256,
        im: std::arch::x86_64::__m256,
    ) {
        let idx = point * self.n2 + col;
        std::arch::x86_64::_mm256_storeu_ps(self.out_re.as_mut_ptr().add(idx), re);
        std::arch::x86_64::_mm256_storeu_ps(self.out_im.as_mut_ptr().add(idx), im);
    }
}

pub(crate) struct InverseRealSweep<'a> {
    re: &'a [f32],
    im: &'a [f32],
    n2: usize,
    output: &'a mut [f32],
    scale: f32,
}

impl<'a> InverseRealSweep<'a> {
    pub(crate) fn checked<const R: usize>(
        re: &'a [f32],
        im: &'a [f32],
        n2: usize,
        output: &'a mut [f32],
        scale: f32,
    ) -> Self {
        assert_eq!(re.len(), R * n2, "real sweep real input length mismatch");
        assert_eq!(
            im.len(),
            R * n2,
            "real sweep imaginary input length mismatch"
        );
        assert_eq!(output.len(), R * n2, "real sweep output length mismatch");
        Self {
            re,
            im,
            n2,
            output,
            scale,
        }
    }
}

impl<const R: usize> RealSweep<true, R> for InverseRealSweep<'_> {
    #[inline(always)]
    fn n2(&self) -> usize {
        self.n2
    }

    #[inline(always)]
    fn re_ptr(&self, point: usize, col: usize) -> *const f32 {
        self.re.as_ptr().wrapping_add(point * self.n2 + col)
    }

    #[inline(always)]
    fn im_ptr(&self, point: usize, col: usize) -> *const f32 {
        self.im.as_ptr().wrapping_add(point * self.n2 + col)
    }

    #[inline(always)]
    fn load_re(&self, point: usize, col: usize) -> f32 {
        self.re[point * self.n2 + col]
    }

    #[inline(always)]
    fn load_im(&self, point: usize, col: usize) -> f32 {
        self.im[point * self.n2 + col]
    }

    #[inline(always)]
    fn store(&mut self, point: usize, col: usize, re: f32, _im: f32) {
        self.output[point * self.n2 + col] = re * self.scale;
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn store_avx(
        &mut self,
        point: usize,
        col: usize,
        re: std::arch::x86_64::__m256,
        _im: std::arch::x86_64::__m256,
    ) {
        let idx = point * self.n2 + col;
        let scale = std::arch::x86_64::_mm256_set1_ps(self.scale);
        let scaled = std::arch::x86_64::_mm256_mul_ps(re, scale);
        std::arch::x86_64::_mm256_storeu_ps(self.output.as_mut_ptr().add(idx), scaled);
    }
}
