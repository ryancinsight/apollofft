//! Precision storage contract for FrFT execution.

use crate::application::execution::plan::frft::dimension_1d::FrftPlan;
use crate::domain::contracts::error::FrftError;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};

/// Complex storage accepted by typed FrFT paths.
pub trait FrftStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `Complex64` arithmetic path.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FrftError::LengthMismatch {
                input: input.len().max(output.len()),
                plan: plan.len(),
            });
        }
        let input64 = Array1::from_iter(input.iter().copied().map(Self::to_complex64));
        let mut output64 = Array1::<Complex64>::zeros(plan.len());
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = Self::from_complex64(value);
        }
        Ok(())
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FrftError::LengthMismatch {
                input: input.len().max(output.len()),
                plan: plan.len(),
            });
        }
        let input64 = Array1::from_iter(input.iter().copied().map(Self::to_complex64));
        let mut output64 = Array1::<Complex64>::zeros(plan.len());
        plan.inverse_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = Self::from_complex64(value);
        }
        Ok(())
    }
}

impl FrftStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn inverse_into(
        plan: &FrftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(input, output)
    }
}

impl FrftStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl FrftStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), FrftError> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(FrftError::PrecisionMismatch)
    }
}
