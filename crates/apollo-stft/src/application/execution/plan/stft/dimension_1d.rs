//! 1D Short-Time Fourier Transform plan.

use crate::application::execution::kernel::hann::hann_window;
use crate::domain::contracts::error::{StftError, StftResult};
use apollo_fft::application::plan::FftPlan1D;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

/// Return `true` when `n > 0`.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

/// Reusable short-time Fourier transform plan.
///
/// Stores a validated frame length, hop size, Hann analysis window, and FFT plan.
/// Construct with `StftPlan::new`; the FFT plan is allocated once and reused.
pub struct StftPlan {
    frame_len: usize,
    hop_len: usize,
    window: Array1<f64>,
    fft_plan: FftPlan1D,
}

impl StftPlan {
    /// Create a validated STFT plan with a Hann analysis window.
    ///
    /// # Errors
    /// Returns `Err` if `frame_len == 0`, `hop_len == 0`, or `hop_len > frame_len`.
    pub fn new(frame_len: usize, hop_len: usize) -> StftResult<Self> {
        if frame_len == 0 {
            return Err(StftError::EmptyFrameLength);
        }
        if hop_len == 0 {
            return Err(StftError::EmptyHopSize);
        }
        if hop_len > frame_len {
            return Err(StftError::HopExceedsFrame);
        }
        let window = hann_window(frame_len);
        let fft_plan = FftPlan1D::new(
            apollo_fft::types::Shape1D::new(frame_len).expect("STFT frame length must be valid"),
        );
        Ok(Self {
            frame_len,
            hop_len,
            window,
            fft_plan,
        })
    }

    /// Return the frame length.
    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Return the hop length.
    #[must_use]
    pub const fn hop_len(&self) -> usize {
        self.hop_len
    }

    /// Return the analysis window.
    #[must_use]
    pub fn window(&self) -> &Array1<f64> {
        &self.window
    }

    /// Return the number of frequency bins (equal to `frame_len`).
    #[must_use]
    pub const fn spectrum_len(&self) -> usize {
        self.frame_len
    }

    /// Return the number of frames for a signal of length `signal_len`.
    ///
    /// Frames are centered at `m * hop_len` for m in 0..frames.
    /// Includes frames whose non-zero window extent overlaps with \[0, signal_len).
    /// Returns 0 when `signal_len < frame_len`.
    #[must_use]
    pub fn frame_count(&self, signal_len: usize) -> usize {
        if signal_len < self.frame_len {
            0
        } else {
            1 + signal_len.div_ceil(self.hop_len)
        }
    }

    /// Forward STFT of a real-valued signal using the internal Hann window.
    ///
    /// Applies the Hann analysis window to each frame and computes the DFT.
    /// Returns a flat array of shape `[frames * spectrum_len]`.
    ///
    /// # Errors
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    pub fn forward(&self, signal: &Array1<f64>) -> StftResult<Array1<Complex64>> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        let mut output = Array1::<Complex64>::zeros(frames * self.spectrum_len());
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Forward STFT with a user-supplied analysis window.
    ///
    /// # Errors
    /// Returns `Err(StftError::WindowLengthMismatch)` when `window.len() != frame_len`.
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    pub fn forward_with_window(
        &self,
        signal: &Array1<f64>,
        window: &[f64],
    ) -> StftResult<Array1<Complex64>> {
        if window.len() != self.frame_len {
            return Err(StftError::WindowLengthMismatch);
        }
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        let mut output = Array1::<Complex64>::zeros(frames * self.spectrum_len());
        self.forward_with_window_inner(signal, window, &mut output)?;
        Ok(output)
    }

    /// Forward STFT into a pre-allocated output buffer.
    ///
    /// # Errors
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    /// Returns `Err(StftError::LengthMismatch)` when `output.len() != frames * spectrum_len`.
    pub fn forward_into(
        &self,
        signal: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) -> StftResult<()> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let window_slice = self.window.as_slice().expect("window must be contiguous");
        self.forward_with_window_inner(signal, window_slice, output)
    }

    /// Forward STFT for typed real input and typed complex output storage.
    pub fn forward_typed_into<T: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        signal: &Array1<T>,
        output: &mut Array1<O>,
        profile: PrecisionProfile,
    ) -> StftResult<()> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        if output.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        let signal64 = Array1::from_iter(signal.iter().copied().map(T::to_f64));
        let mut output64 = Array1::<Complex64>::zeros(output.len());
        self.forward_into(&signal64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = O::from_complex64(value);
        }
        Ok(())
    }

    fn forward_with_window_inner(
        &self,
        signal: &Array1<f64>,
        window: &[f64],
        output: &mut Array1<Complex64>,
    ) -> StftResult<()> {
        let frames = self.frame_count(signal.len());
        if output.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        output
            .as_slice_mut()
            .expect("output buffer must be contiguous")
            .par_chunks_mut(self.spectrum_len())
            .enumerate()
            .for_each(|(m, out_chunk)| {
                let start = m as isize * self.hop_len as isize - (self.frame_len / 2) as isize;
                for n in 0..self.frame_len {
                    let signal_index = start + n as isize;
                    out_chunk[n] = if signal_index >= 0 && (signal_index as usize) < signal.len() {
                        Complex64::new(signal[signal_index as usize] * window[n], 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                }
                self.fft_plan.forward_complex_slice_inplace(out_chunk);
            });
        Ok(())
    }

    /// Inverse STFT via weighted overlap-add (WOLA).
    ///
    /// Normalization: each sample is divided by the sum of squared window values
    /// across all contributing frames. Returns zeros at positions with zero total weight.
    ///
    /// # Errors
    /// Returns `Err(StftError::LengthMismatch)` when spectrum length is inconsistent.
    pub fn inverse(
        &self,
        spectrum: &Array1<Complex64>,
        signal_len: usize,
    ) -> StftResult<Array1<f64>> {
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        let mut output = Array1::<f64>::zeros(signal_len);
        self.inverse_into(spectrum, signal_len, &mut output)?;
        Ok(output)
    }

    /// Inverse STFT into a pre-allocated output buffer.
    ///
    /// Frame IFFTs are computed in parallel; overlap-add accumulation is sequential
    /// to avoid data races on shared output positions.
    ///
    /// # Errors
    /// Returns `Err(StftError::LengthMismatch)` when lengths are inconsistent.
    /// Returns `Err(StftError::InputTooShort)` when `signal_len < frame_len`.
    pub fn inverse_into(
        &self,
        spectrum: &Array1<Complex64>,
        signal_len: usize,
        output: &mut Array1<f64>,
    ) -> StftResult<()> {
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        if output.len() != signal_len {
            return Err(StftError::LengthMismatch);
        }
        if signal_len < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let mut flat_frames = vec![0.0f64; frames * self.frame_len];
        let mut flat_complex = vec![Complex64::new(0.0, 0.0); frames * self.frame_len];
        flat_complex
            .par_chunks_mut(self.frame_len)
            .zip(flat_frames.par_chunks_mut(self.frame_len))
            .enumerate()
            .for_each(|(m, (frame_complex, frame_out))| {
                let offset = m * self.spectrum_len();
                for k in 0..self.spectrum_len() {
                    frame_complex[k] = spectrum[offset + k];
                }
                self.fft_plan.inverse_complex_slice_inplace(frame_complex);
                for n in 0..self.frame_len {
                    frame_out[n] = frame_complex[n].re * self.window[n];
                }
            });

        // Sequential overlap-add: avoids data races on shared output positions.
        let mut overlap = vec![0.0f64; signal_len];
        let mut weight = vec![0.0f64; signal_len];
        for (m, frame_vals) in flat_frames.chunks(self.frame_len).enumerate() {
            let start = m as isize * self.hop_len as isize - (self.frame_len / 2) as isize;
            for n in 0..self.frame_len {
                let signal_index = start + n as isize;
                if signal_index >= 0 && (signal_index as usize) < signal_len {
                    let idx = signal_index as usize;
                    overlap[idx] += frame_vals[n];
                    weight[idx] += self.window[n] * self.window[n];
                }
            }
        }
        for i in 0..signal_len {
            output[i] = if weight[i] > 0.0 {
                overlap[i] / weight[i]
            } else {
                0.0
            };
        }
        Ok(())
    }

    /// Inverse STFT for typed complex spectrum and typed real output storage.
    pub fn inverse_typed_into<T: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        spectrum: &Array1<T>,
        signal_len: usize,
        output: &mut Array1<O>,
        profile: PrecisionProfile,
    ) -> StftResult<()> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() || output.len() != signal_len {
            return Err(StftError::LengthMismatch);
        }
        if signal_len < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let spectrum64 = Array1::from_iter(spectrum.iter().copied().map(T::to_complex64));
        let mut output64 = Array1::<f64>::zeros(signal_len);
        self.inverse_into(&spectrum64, signal_len, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = O::from_f64(value);
        }
        Ok(())
    }

    /// Forward STFT (alias for `forward`).
    pub fn forward_inplace(&self, signal: &Array1<f64>) -> StftResult<Array1<Complex64>> {
        self.forward(signal)
    }

    /// Inverse STFT (alias for `inverse`).
    pub fn inverse_inplace(
        &self,
        spectrum: &Array1<Complex64>,
        signal_len: usize,
    ) -> StftResult<Array1<f64>> {
        self.inverse(spectrum, signal_len)
    }
}

/// Real input storage accepted by typed STFT forward paths.
pub trait StftRealStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `f64` arithmetic.
    fn to_f64(self) -> f64;
}

impl StftRealStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }
}

impl StftRealStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl StftRealStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

/// Real output storage accepted by typed STFT inverse paths.
pub trait StftRealOutputStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert owner arithmetic output into storage.
    fn from_f64(value: f64) -> Self;
}

impl StftRealOutputStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn from_f64(value: f64) -> Self {
        value
    }
}

impl StftRealOutputStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl StftRealOutputStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

/// Complex output storage accepted by typed STFT forward paths.
pub trait StftSpectrumStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert owner complex result into storage.
    fn from_complex64(value: Complex64) -> Self;
}

impl StftSpectrumStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn from_complex64(value: Complex64) -> Self {
        value
    }
}

impl StftSpectrumStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl StftSpectrumStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

/// Complex input storage accepted by typed STFT inverse paths.
pub trait StftSpectrumInput: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `Complex64` arithmetic.
    fn to_complex64(self) -> Complex64;
}

impl StftSpectrumInput for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }
}

impl StftSpectrumInput for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl StftSpectrumInput for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> StftResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(StftError::PrecisionMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn hann_window_is_symmetric() {
        let window = hann_window(8);
        for i in 0..8 {
            assert_relative_eq!(window[i], window[7 - i], epsilon = 1.0e-12);
        }
    }

    #[test]
    fn forward_and_inverse_roundtrip_for_cola_case() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec(vec![
            1.0, -1.0, 0.5, 2.0, -0.75, 0.25, 1.5, -0.5, 0.125, 0.875, -1.25, 0.75,
        ]);
        let spectrum = plan.forward(&signal).expect("forward");
        let recovered = plan.inverse(&spectrum, signal.len()).expect("inverse");
        for (actual, expected) in recovered.iter().zip(signal.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-8);
        }
    }

    #[test]
    fn forward_into_matches_allocating_path() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
        let expected = plan.forward(&signal).expect("forward");
        let mut actual = Array1::<Complex64>::zeros(expected.len());
        plan.forward_into(&signal, &mut actual)
            .expect("forward_into");
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal64 = Array1::from_vec((0..16).map(|i| (i as f64 * 0.2).sin()).collect());
        let expected = plan.forward(&signal64).expect("forward");

        let mut out64 = Array1::<Complex64>::zeros(expected.len());
        plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let signal32 = signal64.mapv(|value| value as f32);
        let represented32 = signal32.mapv(f64::from);
        let expected32 = plan
            .forward(&represented32)
            .expect("represented f32 forward");
        let mut out32 = Array1::<Complex32>::zeros(expected32.len());
        plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected32.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let mut recovered32 = Array1::<f32>::zeros(signal32.len());
        plan.inverse_typed_into(
            &out32,
            signal32.len(),
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 inverse");
        for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-4);
        }

        let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
        let represented16 = signal16.mapv(|value| f64::from(value.to_f32()));
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut out16 = Array1::from_elem(expected16.len(), [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 forward");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec(vec![1.0_f32; 16]);
        let mut output =
            Array1::<Complex32>::zeros(plan.frame_count(signal.len()) * plan.spectrum_len());
        assert!(matches!(
            plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(StftError::PrecisionMismatch)
        ));
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            StftPlan::new(0, 4),
            Err(StftError::EmptyFrameLength)
        ));
        assert!(matches!(StftPlan::new(8, 0), Err(StftError::EmptyHopSize)));
        assert!(matches!(
            StftPlan::new(4, 8),
            Err(StftError::HopExceedsFrame)
        ));
    }

    #[test]
    fn input_too_short_is_rejected() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec(vec![0.0; 4]);
        assert!(matches!(
            plan.forward(&signal),
            Err(StftError::InputTooShort)
        ));
    }

    #[test]
    fn forward_with_window_rejects_wrong_length() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec(vec![1.0f64; 12]);
        let bad_window = vec![1.0f64; 6];
        assert!(matches!(
            plan.forward_with_window(&signal, &bad_window),
            Err(StftError::WindowLengthMismatch)
        ));
    }

    #[test]
    fn forward_with_custom_window_matches_internal_hann() {
        let plan = StftPlan::new(8, 4).expect("valid plan");
        let signal = Array1::from_vec((0..12).map(|i| (i as f64 * 0.3).sin()).collect());
        let expected = plan.forward(&signal).expect("forward");
        let window: Vec<f64> = hann_window(8).to_vec();
        let actual = plan
            .forward_with_window(&signal, &window)
            .expect("forward_with_window");
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
        }
    }

    proptest::proptest! {
        #[test]
        fn roundtrip_holds_for_random_signals(
            signal_len in 8usize..128,
            frame_len in 2usize..17,
            hop_len in 1usize..9,
        ) {
            // COLA coverage: hop_len <= frame_len - 2 ensures every signal position
            // is covered by at least one non-zero Hann window value.
            // (Hann window is zero at both endpoints; dead zones appear when hop > frame_len-2.)
            prop_assume!(frame_len <= signal_len);
            prop_assume!(hop_len <= frame_len);
            prop_assume!(hop_len + 2 <= frame_len);
            let plan = StftPlan::new(frame_len, hop_len).expect("valid plan");
            let signal = Array1::from_vec(
                (0..signal_len).map(|i| (i as f64 * 0.37).sin()).collect(),
            );
            let spectrum = plan.forward(&signal).expect("forward");
            let recovered = plan.inverse(&spectrum, signal_len).expect("inverse");
            let err = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            prop_assert!(err < 0.5, "roundtrip error too large: {}", err);
        }
    }
}
