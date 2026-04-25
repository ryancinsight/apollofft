//! Wavelet family descriptors.

use serde::{Deserialize, Serialize};

/// Orthogonal supported wavelets supported by the DWT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscreteWavelet {
    /// Haar wavelet, also known as Daubechies-1.
    Haar,
    /// Daubechies-4 orthogonal wavelet.
    Daubechies4,
}

/// Continuous analysis wavelets supported by the CWT plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ContinuousWavelet {
    /// Ricker wavelet, also known as the Mexican hat wavelet.
    Ricker,
    /// Real Morlet wavelet with angular carrier frequency `omega0`.
    ///
    /// # Admissibility theorem
    ///
    /// Apollo uses the DC-corrected real Morlet wavelet
    ///
    /// `psi(t) = pi^(-1/4) (cos(omega0 t) - exp(-omega0^2/2)) exp(-t^2/2)`.
    ///
    /// Since `int cos(omega0 t) exp(-t^2/2) dt = sqrt(2pi) exp(-omega0^2/2)`
    /// and `int exp(-t^2/2) dt = sqrt(2pi)`, subtracting
    /// `exp(-omega0^2/2)` gives zero integral and therefore zero DC response
    /// in the continuous limit.
    Morlet {
        /// Dimensionless angular carrier frequency.
        omega0: f64,
    },
}
