//! Mellin scale-domain configuration.

use crate::domain::contracts::error::{MellinError, MellinResult};
use serde::{Deserialize, Serialize};

/// Validated Mellin transform scale configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MellinScaleConfig {
    samples: usize,
    min_scale: f64,
    max_scale: f64,
}

impl MellinScaleConfig {
    /// Create a validated Mellin scale configuration.
    pub fn new(samples: usize, min_scale: f64, max_scale: f64) -> MellinResult<Self> {
        if samples == 0 {
            return Err(MellinError::EmptySampleCount);
        }
        if !min_scale.is_finite() || !max_scale.is_finite() || min_scale <= 0.0 || max_scale <= 0.0
        {
            return Err(MellinError::InvalidScaleBound);
        }
        if min_scale >= max_scale {
            return Err(MellinError::InvalidScaleOrder);
        }
        Ok(Self {
            samples,
            min_scale,
            max_scale,
        })
    }

    /// Return sample count.
    #[must_use]
    pub const fn samples(self) -> usize {
        self.samples
    }

    /// Return minimum scale.
    #[must_use]
    pub const fn min_scale(self) -> f64 {
        self.min_scale
    }

    /// Return maximum scale.
    #[must_use]
    pub const fn max_scale(self) -> f64 {
        self.max_scale
    }
}
