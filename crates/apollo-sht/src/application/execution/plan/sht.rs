//! Reusable spherical harmonic transform plan.
//!
//! The plan uses Gauss-Legendre latitude nodes and uniform longitude nodes.
//! Forward transforms compute coefficients
//! `a_lm = integral f(theta, phi) conj(Y_lm(theta, phi)) dOmega` by product
//! quadrature. Inverse transforms evaluate
//! `f(theta, phi) = sum_l sum_m a_lm Y_lm(theta, phi)` on the same grid.

use crate::domain::contracts::error::{ShtError, ShtResult};
use crate::domain::metadata::grid::SphericalGridSpec;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::{
    gauss_legendre_nodes_weights, spherical_harmonic,
};
use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

/// Reusable spherical harmonic transform (SHT) plan.
///
/// Pre-computes Gauss-Legendre nodes and weights for the latitude axis and caches the
/// validated [`SphericalGridSpec`]. The same plan can be reused for multiple transforms
/// without recomputing the quadrature rule.
///
/// # Complexity Theorem
///
/// Let `L = max_degree`, `N_lat = latitudes`, `N_lon = longitudes`, and
/// `M = (L + 1)^2` (total number of spectral modes).
///
/// | Transform | Complexity          | Description                                          |
/// |-----------|---------------------|------------------------------------------------------|
/// | Forward   | O(M · N_lat · N_lon) | Quadrature sum over all grid points for each mode   |
/// | Inverse   | O(N_lat · N_lon · M) | Synthesis sum over all modes for each grid point    |
///
/// Both operations are equivalent to dense matrix–vector products of dimension
/// `(M) × (N_lat · N_lon)`. Rayon parallelism distributes the `N_lat` latitude rows
/// across available threads, giving a practical wall-time factor of `1/P` for `P` cores
/// on the outer loop.
///
/// # Quadrature Exactness
///
/// The Gauss-Legendre nodes guarantee exact integration for products of spherical
/// harmonics of degree `<= L` provided the following grid constraints hold:
///
/// - `N_lat > L`: The `N_lat`-point GL rule is exact for polynomials of degree
///   `<= 2*N_lat - 1 >= 2L`, which covers all products `Y_l^m * conj(Y_{l'}^{m'})`
///   with `l, l' <= L` (degree `<= 2L` in `cos θ`). See Theorem 2 and Theorem 4 in
///   [`crate::infrastructure::kernel::spherical_harmonic`].
/// - `N_lon >= 2L + 1`: The uniform longitude grid recovers all azimuthal modes
///   `|m| <= L` without aliasing (DFT orthogonality identity).
///
/// Under these constraints, `inverse(forward(f)) = f` in exact arithmetic for any
/// field `f` band-limited to degree `<= L` (Theorem 4 in
/// [`crate::infrastructure::kernel::spherical_harmonic`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ShtPlan {
    grid: SphericalGridSpec,
    cos_theta_nodes: Vec<f64>,
    theta_weights: Vec<f64>,
}

impl ShtPlan {
    /// Create a validated SHT plan.
    pub fn new(latitudes: usize, longitudes: usize, max_degree: usize) -> ShtResult<Self> {
        let grid = SphericalGridSpec::new(latitudes, longitudes, max_degree)?;
        let (cos_theta_nodes, theta_weights) = gauss_legendre_nodes_weights(latitudes);
        Ok(Self {
            grid,
            cos_theta_nodes,
            theta_weights,
        })
    }

    /// Return the validated grid specification.
    #[must_use]
    pub const fn grid(&self) -> SphericalGridSpec {
        self.grid
    }

    /// Return colatitude angle for a latitude index.
    #[must_use]
    pub fn theta(&self, latitude_index: usize) -> f64 {
        self.cos_theta_nodes[latitude_index].acos()
    }

    /// Return longitude angle for a longitude index.
    #[must_use]
    pub fn phi(&self, longitude_index: usize) -> f64 {
        std::f64::consts::TAU * longitude_index as f64 / self.grid.longitudes() as f64
    }

    /// Forward SHT for real-valued samples on the plan grid.
    pub fn forward_real(&self, samples: &Array2<f64>) -> ShtResult<SphericalHarmonicCoefficients> {
        self.check_sample_shape(samples.dim())?;
        let complex_samples = samples.mapv(|value| Complex64::new(value, 0.0));
        self.forward_complex(&complex_samples)
    }

    /// Forward SHT for complex-valued samples on the plan grid.
    pub fn forward_complex(
        &self,
        samples: &Array2<Complex64>,
    ) -> ShtResult<SphericalHarmonicCoefficients> {
        self.check_sample_shape(samples.dim())?;
        let max_degree = self.grid.max_degree();
        let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
        let longitude_weight = std::f64::consts::TAU / self.grid.longitudes() as f64;
        let n_lat = self.grid.latitudes();
        let n_lon = self.grid.longitudes();

        // Pre-collect all (degree, order) mode pairs for deterministic indexing.
        let all_modes: Vec<(usize, isize)> = (0..=max_degree)
            .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
            .collect();

        // Parallelize over latitude rows; each row contributes to all modes independently.
        let contributions: Vec<Vec<Complex64>> = (0..n_lat)
            .into_par_iter()
            .map(|lat| {
                let theta = self.theta(lat);
                let weight = self.theta_weights[lat];
                all_modes
                    .iter()
                    .map(|&(degree, order)| {
                        let lon_sum: Complex64 = (0..n_lon)
                            .map(|lon| {
                                let phi = self.phi(lon);
                                samples[[lat, lon]]
                                    * spherical_harmonic(degree, order, theta, phi).conj()
                            })
                            .sum();
                        lon_sum * (weight * longitude_weight)
                    })
                    .collect()
            })
            .collect();

        // Accumulate all latitude contributions into coefficients.
        for lat_contribs in contributions {
            for (mode_idx, coeff) in lat_contribs.into_iter().enumerate() {
                let (degree, order) = all_modes[mode_idx];
                let existing = coefficients.get(degree, order);
                coefficients.set(degree, order, existing + coeff);
            }
        }

        Ok(coefficients)
    }

    /// Inverse SHT evaluating real-valued samples on the plan grid.
    pub fn inverse_real(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<Array2<f64>> {
        Ok(self.inverse_complex(coefficients)?.mapv(|value| value.re))
    }

    /// Inverse SHT evaluating complex-valued samples on the plan grid.
    pub fn inverse_complex(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<Array2<Complex64>> {
        self.check_coefficient_shape(coefficients)?;
        let max_degree = self.grid.max_degree();
        let n_lat = self.grid.latitudes();
        let n_lon = self.grid.longitudes();

        // Pre-collect all (degree, order) mode pairs for deterministic iteration.
        let all_modes: Vec<(usize, isize)> = (0..=max_degree)
            .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
            .collect();

        // Parallelize over latitude rows; each row is computed independently.
        let row_values: Vec<Vec<Complex64>> = (0..n_lat)
            .into_par_iter()
            .map(|lat| {
                let theta = self.theta(lat);
                (0..n_lon)
                    .map(|lon| {
                        let phi = self.phi(lon);
                        all_modes
                            .iter()
                            .map(|&(degree, order)| {
                                coefficients.get(degree, order)
                                    * spherical_harmonic(degree, order, theta, phi)
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();

        // Assemble into output array.
        let mut samples = Array2::<Complex64>::zeros((n_lat, n_lon));
        for (lat, row) in row_values.into_iter().enumerate() {
            for (lon, value) in row.into_iter().enumerate() {
                samples[[lat, lon]] = value;
            }
        }

        Ok(samples)
    }

    fn check_sample_shape(&self, shape: (usize, usize)) -> ShtResult<()> {
        if shape != (self.grid.latitudes(), self.grid.longitudes()) {
            return Err(ShtError::SampleShapeMismatch);
        }
        Ok(())
    }

    fn check_coefficient_shape(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<()> {
        let expected = (self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1);
        if coefficients.max_degree() != self.grid.max_degree()
            || coefficients.values().dim() != expected
        {
            return Err(ShtError::CoefficientShapeMismatch);
        }
        Ok(())
    }
}
