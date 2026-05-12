//! Precision storage contract for FWHT execution.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::application::execution::plan::fwht::dimension_1d::FwhtPlan;
use crate::domain::contracts::error::FwhtError;
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;
use std::cell::RefCell;

thread_local! {
    static TYPED_INPUT64_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static TYPED_OUTPUT64_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static TYPED_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Real storage accepted by typed FWHT paths.
pub trait FwhtStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert an owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FwhtError::LengthMismatch);
        }
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.forward_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FwhtError::LengthMismatch);
        }
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.inverse_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }
}

impl FwhtStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn inverse_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(input, output)
    }
}

impl FwhtStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn forward_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FwhtError::LengthMismatch);
        }
        output.assign(input);
        wht_inplace(output.as_slice_mut().expect("Array must be contiguous"));
        Ok(())
    }

    fn inverse_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        Self::forward_into(plan, input, output, profile)?;
        let scale = 1.0_f32 / plan.len() as f32;
        output.mapv_inplace(|value| value * scale);
        Ok(())
    }
}

impl FwhtStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }

    fn forward_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FwhtError::LengthMismatch);
        }
        with_f32_workspace(plan.len(), |compute| {
            for (slot, value) in compute.iter_mut().zip(input.iter()) {
                *slot = value.to_f32();
            }
            wht_inplace(compute);
            for (slot, value) in output.iter_mut().zip(compute.iter().copied()) {
                *slot = f16::from_f32(value);
            }
            Ok(())
        })
    }

    fn inverse_into(
        plan: &FwhtPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(FwhtError::LengthMismatch);
        }
        with_f32_workspace(plan.len(), |compute| {
            for (slot, value) in compute.iter_mut().zip(input.iter()) {
                *slot = value.to_f32();
            }
            wht_inplace(compute);
            let scale = 1.0_f32 / plan.len() as f32;
            for (slot, value) in output.iter_mut().zip(compute.iter().copied()) {
                *slot = f16::from_f32(value * scale);
            }
            Ok(())
        })
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), FwhtError> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(FwhtError::PrecisionMismatch)
    }
}

fn with_f64_workspaces<R>(n: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            let mut input_scratch = input_scratch.borrow_mut();
            if input_scratch.len() < n {
                input_scratch.resize(n, 0.0);
            }

            let mut output_scratch = output_scratch.borrow_mut();
            if output_scratch.len() < n {
                output_scratch.resize(n, 0.0);
            }

            f(&mut input_scratch[..n], &mut output_scratch[..n])
        })
    })
}

fn with_f32_workspace<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TYPED_F32_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < n {
            scratch.resize(n, 0.0);
        }
        f(&mut scratch[..n])
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            TYPED_F32_SCRATCH.with(|f32_scratch| {
                (
                    input_scratch.borrow().capacity(),
                    output_scratch.borrow().capacity(),
                    f32_scratch.borrow().capacity(),
                )
            })
        })
    })
}
