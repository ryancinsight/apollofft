//! Reusable discrete wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::DiscreteWavelet;
use crate::domain::spectrum::coefficients::DwtCoefficients;
use crate::infrastructure::kernel::discrete::{analysis_stage_into, synthesis_stage_into};
use crate::CwtPlan;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array2;

/// Reusable 1D orthogonal DWT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwtPlan {
    len: usize,
    levels: usize,
    wavelet: DiscreteWavelet,
}

impl DwtPlan {
    /// Create a DWT plan for a power-of-two signal length.
    pub fn new(len: usize, levels: usize, wavelet: DiscreteWavelet) -> WaveletResult<Self> {
        if len == 0 {
            return Err(WaveletError::EmptySignal);
        }
        if !len.is_power_of_two() {
            return Err(WaveletError::NonPowerOfTwoLength);
        }
        if levels == 0 {
            return Err(WaveletError::EmptyLevelCount);
        }
        if levels > len.trailing_zeros() as usize {
            return Err(WaveletError::LevelExceedsLength);
        }
        Ok(Self {
            len,
            levels,
            wavelet,
        })
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return decomposition level count.
    #[must_use]
    pub const fn levels(self) -> usize {
        self.levels
    }

    /// Return wavelet family.
    #[must_use]
    pub const fn wavelet(self) -> DiscreteWavelet {
        self.wavelet
    }

    /// Execute a multilevel forward DWT.
    pub fn forward(&self, signal: &[f64]) -> WaveletResult<DwtCoefficients> {
        if signal.len() != self.len {
            return Err(WaveletError::LengthMismatch);
        }
        let mut current = signal.to_vec();
        let mut details = Vec::with_capacity(self.levels);
        for _ in 0..self.levels {
            let half = current.len() / 2;
            let mut approximation = vec![0.0; half];
            let mut detail = vec![0.0; half];
            analysis_stage_into(&current, self.wavelet, &mut approximation, &mut detail);
            details.push(detail);
            current = approximation;
        }
        Ok(DwtCoefficients::new(
            self.len,
            self.levels,
            current,
            details,
        ))
    }

    /// Execute inverse multilevel DWT.
    pub fn inverse(&self, coefficients: &DwtCoefficients) -> WaveletResult<Vec<f64>> {
        if coefficients.len() != self.len || coefficients.levels() != self.levels {
            return Err(WaveletError::CoefficientShapeMismatch);
        }
        let mut current = coefficients.approximation().to_vec();
        for detail in coefficients.details().iter().rev() {
            let n = current.len() * 2;
            let mut output = vec![0.0; n];
            synthesis_stage_into(&current, detail, self.wavelet, &mut output);
            current = output;
        }
        Ok(current)
    }

    /// Execute a multilevel forward DWT for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// The owner kernel remains the `f64` orthogonal filter bank. Typed paths
    /// convert represented input into owner arithmetic and quantize once when
    /// writing caller-owned approximation/detail buffers.
    pub fn forward_typed_into<T: WaveletStorage>(
        &self,
        signal: &[T],
        approximation: &mut [T],
        details: &mut [Vec<T>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        T::forward_dwt_into(self, signal, approximation, details, profile)
    }

    /// Execute inverse multilevel DWT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: WaveletStorage>(
        &self,
        approximation: &[T],
        details: &[Vec<T>],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        T::inverse_dwt_into(self, approximation, details, output, profile)
    }

    fn coefficient_shapes(&self) -> impl Iterator<Item = usize> {
        let len = self.len;
        let levels = self.levels;
        (0..levels).map(move |level| len >> (level + 1))
    }
}

/// Real storage accepted by typed wavelet paths.
pub trait WaveletStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage value to owner `f64` arithmetic.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute typed forward DWT into caller-owned buffers.
    fn forward_dwt_into(
        plan: &DwtPlan,
        signal: &[Self],
        approximation: &mut [Self],
        details: &mut [Vec<Self>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if signal.len() != plan.len {
            return Err(WaveletError::LengthMismatch);
        }
        let signal64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        let coefficients = plan.forward(&signal64)?;
        for (slot, value) in approximation
            .iter_mut()
            .zip(coefficients.approximation().iter().copied())
        {
            *slot = Self::from_f64(value);
        }
        for (detail_out, detail_in) in details.iter_mut().zip(coefficients.details()) {
            for (slot, value) in detail_out.iter_mut().zip(detail_in.iter().copied()) {
                *slot = Self::from_f64(value);
            }
        }
        Ok(())
    }

    /// Execute typed inverse DWT into a caller-owned signal buffer.
    fn inverse_dwt_into(
        plan: &DwtPlan,
        approximation: &[Self],
        details: &[Vec<Self>],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if output.len() != plan.len {
            return Err(WaveletError::LengthMismatch);
        }
        let approximation64: Vec<f64> = approximation.iter().copied().map(Self::to_f64).collect();
        let details64: Vec<Vec<f64>> = details
            .iter()
            .map(|detail| detail.iter().copied().map(Self::to_f64).collect())
            .collect();
        let coefficients = DwtCoefficients::new(plan.len, plan.levels, approximation64, details64);
        let signal = plan.inverse(&coefficients)?;
        for (slot, value) in output.iter_mut().zip(signal.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute typed CWT into caller-owned storage.
    fn transform_cwt_into(
        plan: &CwtPlan,
        signal: &[Self],
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_cwt_output_shape(plan, output)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let signal64: Vec<f64> = signal.iter().copied().map(Self::to_f64).collect();
        let coefficients = plan.transform(&signal64)?;
        for (slot, value) in output.iter_mut().zip(coefficients.values().iter().copied()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl WaveletStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_dwt_into(
        plan: &DwtPlan,
        signal: &[Self],
        approximation: &mut [Self],
        details: &mut [Vec<Self>],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if signal.len() != plan.len {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = plan.forward(signal)?;
        approximation.copy_from_slice(coefficients.approximation());
        for (detail_out, detail_in) in details.iter_mut().zip(coefficients.details()) {
            detail_out.copy_from_slice(detail_in);
        }
        Ok(())
    }

    fn inverse_dwt_into(
        plan: &DwtPlan,
        approximation: &[Self],
        details: &[Vec<Self>],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_dwt_output_shapes(plan, approximation.len(), details)?;
        if output.len() != plan.len {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = DwtCoefficients::new(
            plan.len,
            plan.levels,
            approximation.to_vec(),
            details.to_vec(),
        );
        let signal = plan.inverse(&coefficients)?;
        output.copy_from_slice(&signal);
        Ok(())
    }

    fn transform_cwt_into(
        plan: &CwtPlan,
        signal: &[Self],
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> WaveletResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_cwt_output_shape(plan, output)?;
        if signal.len() != plan.len() {
            return Err(WaveletError::LengthMismatch);
        }
        let coefficients = plan.transform(signal)?;
        output.assign(coefficients.values());
        Ok(())
    }
}

impl WaveletStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl WaveletStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> WaveletResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(WaveletError::PrecisionMismatch)
    }
}

fn validate_dwt_output_shapes<T>(
    plan: &DwtPlan,
    approximation_len: usize,
    details: &[Vec<T>],
) -> WaveletResult<()> {
    let expected_approximation_len = plan.len >> plan.levels;
    if approximation_len != expected_approximation_len || details.len() != plan.levels {
        return Err(WaveletError::CoefficientShapeMismatch);
    }
    if details
        .iter()
        .map(Vec::len)
        .zip(plan.coefficient_shapes())
        .any(|(actual, expected)| actual != expected)
    {
        return Err(WaveletError::CoefficientShapeMismatch);
    }
    Ok(())
}

fn validate_cwt_output_shape<T>(plan: &CwtPlan, output: &Array2<T>) -> WaveletResult<()> {
    if output.nrows() == plan.scales().len() && output.ncols() == plan.len() {
        Ok(())
    } else {
        Err(WaveletError::CoefficientShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn detail_buffers<T: Copy>(plan: &DwtPlan, fill: T) -> Vec<Vec<T>> {
        plan.coefficient_shapes()
            .map(|len| vec![fill; len])
            .collect()
    }

    #[test]
    fn typed_dwt_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("valid DWT plan");
        let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let expected = plan.forward(&signal64).expect("forward");

        let mut approx64 = vec![0.0_f64; 1];
        let mut details64 = detail_buffers(&plan, 0.0_f64);
        plan.forward_typed_into(
            &signal64,
            &mut approx64,
            &mut details64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed f64 forward");
        for (actual, expected) in approx64.iter().zip(expected.approximation()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual_detail, expected_detail) in details64.iter().zip(expected.details()) {
            for (actual, expected) in actual_detail.iter().zip(expected_detail) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
            }
        }

        let signal32 = signal64.map(|value| value as f32);
        let mut approx32 = vec![0.0_f32; 1];
        let mut details32 = detail_buffers(&plan, 0.0_f32);
        plan.forward_typed_into(
            &signal32,
            &mut approx32,
            &mut details32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 forward");
        let expected32 = plan
            .forward(&signal32.map(f64::from))
            .expect("represented f32 forward");
        for (actual, expected) in approx32.iter().zip(expected32.approximation()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let mut recovered32 = [0.0_f32; 8];
        plan.inverse_typed_into(
            &approx32,
            &details32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 inverse");
        for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.map(|value| f16::from_f32(value as f32));
        let represented16 = signal16.map(|value| f64::from(value.to_f32()));
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut approx16 = vec![f16::from_f32(0.0); 1];
        let mut details16 = detail_buffers(&plan, f16::from_f32(0.0));
        plan.forward_typed_into(
            &signal16,
            &mut approx16,
            &mut details16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 forward");
        for (actual, expected) in approx16.iter().zip(expected16.approximation()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn typed_dwt_rejects_profile_and_shape_mismatch() {
        let plan = DwtPlan::new(4, 2, DiscreteWavelet::Haar).expect("valid DWT plan");
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let mut approximation = vec![0.0_f32; 1];
        let mut details = detail_buffers(&plan, 0.0_f32);
        assert!(matches!(
            plan.forward_typed_into(
                &signal,
                &mut approximation,
                &mut details,
                PrecisionProfile::HIGH_ACCURACY_F64
            ),
            Err(WaveletError::PrecisionMismatch)
        ));

        details[0].pop();
        assert!(matches!(
            plan.forward_typed_into(
                &signal,
                &mut approximation,
                &mut details,
                PrecisionProfile::LOW_PRECISION_F32
            ),
            Err(WaveletError::CoefficientShapeMismatch)
        ));
    }
}
