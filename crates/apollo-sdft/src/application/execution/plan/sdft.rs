//! Reusable sliding DFT plan and state.
//!
//! `SdftState` owns the current real-valued window and tracked DFT bins. Each
//! update removes the oldest sample, appends the new sample, and updates all
//! tracked bins through the sliding DFT recurrence.

use crate::domain::contracts::error::{SdftError, SdftResult};
use crate::domain::metadata::window::SlidingDftConfig;
use crate::infrastructure::kernel::sliding::{direct_bins, update_bins, update_twiddles};
use num_complex::Complex64;
use std::collections::VecDeque;

/// Reusable SDFT plan.
#[derive(Debug, Clone, PartialEq)]
pub struct SdftPlan {
    config: SlidingDftConfig,
    twiddles: Vec<Complex64>,
}

impl SdftPlan {
    /// Create a validated SDFT plan.
    pub fn new(window_len: usize, bin_count: usize) -> SdftResult<Self> {
        let config = SlidingDftConfig::new(window_len, bin_count)?;
        let twiddles = update_twiddles(window_len, bin_count);
        Ok(Self { config, twiddles })
    }

    /// Return the validated configuration.
    #[must_use]
    pub const fn config(&self) -> SlidingDftConfig {
        self.config
    }

    /// Return the plan window length.
    #[must_use]
    pub const fn window_len(&self) -> usize {
        self.config.window_len()
    }

    /// Return the tracked bin count.
    #[must_use]
    pub const fn bin_count(&self) -> usize {
        self.config.bin_count()
    }

    /// Create zero-initialized streaming state.
    #[must_use]
    pub fn zero_state(&self) -> SdftState {
        let window = vec![0.0; self.window_len()];
        SdftState::from_validated_window(self.clone(), window)
    }

    /// Create streaming state from an initial full window.
    pub fn state_from_window(&self, window: &[f64]) -> SdftResult<SdftState> {
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        Ok(SdftState::from_validated_window(
            self.clone(),
            window.to_vec(),
        ))
    }

    /// Compute direct DFT bins for a full window using this plan's bin count.
    pub fn direct_bins(&self, window: &[f64]) -> SdftResult<Vec<Complex64>> {
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        direct_bins(window, self.bin_count())
    }
}

/// Stateful sliding DFT stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SdftState {
    plan: SdftPlan,
    window: VecDeque<f64>,
    bins: Vec<Complex64>,
    updates: usize,
}

impl SdftState {
    fn from_validated_window(plan: SdftPlan, window: Vec<f64>) -> Self {
        let bins = direct_bins(&window, plan.bin_count())
            .expect("invariant: validated plan has consistent window_len and bin_count");
        Self {
            plan,
            window: VecDeque::from(window),
            bins,
            updates: 0,
        }
    }

    /// Push one new sample and return the updated bins.
    pub fn update(&mut self, sample: f64) -> &[Complex64] {
        let outgoing = self
            .window
            .pop_front()
            .expect("validated SDFT window is non-empty");
        self.window.push_back(sample);
        update_bins(&mut self.bins, &self.plan.twiddles, outgoing, sample);
        self.updates += 1;
        &self.bins
    }

    /// Return current tracked bins.
    #[must_use]
    pub fn bins(&self) -> &[Complex64] {
        &self.bins
    }

    /// Return current window in oldest-to-newest order.
    #[must_use]
    pub fn window(&self) -> Vec<f64> {
        self.window.iter().copied().collect()
    }

    /// Return the number of update calls applied to this state.
    #[must_use]
    pub const fn updates(&self) -> usize {
        self.updates
    }
}
