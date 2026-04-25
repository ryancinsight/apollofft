//! Sparse Fourier transform plan.
//!
//! # Mathematical contract
//!
//! For `x in C^N`, the plan computes the dense FFT coefficients in `O(N log N)`,
//! ranks them by squared magnitude with frequency-index tie-breaking via a
//! min-heap of size `K` in `O(N log K)`, and stores the largest `K` coefficients
//! in a sparse spectrum. Reconstruction expands the sparse spectrum to a dense
//! coefficient vector and evaluates the inverse FFT.
//!
//! ## Complexity
//! Spectrum density computation: `O(N log N)` via apollo-fft auto-selecting kernel.
//! Top-K selection: `O(N log K)` via min-heap of size `K`.
//!
//! The recovery kernel is dense and deterministic. Sparse plan parameters
//! remain explicit domain data so a later sublinear isolation kernel can
//! replace the infrastructure layer without changing this public API.
//!
//! ## Theorem: Top-K Coefficient Selection Optimality
//!
//! **Statement.** For a signal x ∈ ℂ^N with DFT coefficients X = FFT(x), the
//! K-sparse approximation X_K obtained by retaining the K largest-magnitude
//! coefficients minimizes the squared reconstruction error:
//!
//! ```text
//! ‖x - IFFT(X_K)‖² = Σ_{j=K+1}^{N} |X[π(j)]|² / N²
//! ```
//!
//! where π is the permutation sorting |X| in descending order.
//!
//! **Proof sketch.** Parseval's theorem states ‖u - v‖² = (1/N)‖FFT(u) - FFT(v)‖²
//! for all u, v ∈ ℂ^N. The reconstruction error after zeroing all but the K
//! retained DFT bins equals (1/N) Σ_{j: not in top-K} |X[j]|². This sum is
//! minimised by retaining exactly the K coefficients with the largest |X[j]|,
//! since any other selection of K bins leaves a strictly larger or equal residual
//! energy. □
//!
//! **Reference.** Candès & Wakin (2008), "An Introduction to Compressive Sensing",
//! IEEE Signal Processing Magazine 25(2), pp. 21–30.
//!
//! ## Theorem: Exact Recovery for K-Sparse Signals
//!
//! **Statement.** If x ∈ ℂ^N is exactly K-sparse in the DFT domain (at most K
//! frequency components are nonzero), then `SparseFftPlan::new(N, K)?.forward(x)`
//! returns a `SparseSpectrum` containing exactly those K nonzero components, with
//! values matching FFT(x)[k] to within FFT numerical precision
//! (O(N log N · ε_machine)).
//!
//! **Proof sketch.** In exact arithmetic the N-K non-support DFT bins are zero.
//! The top-K heap selector retains all K nonzero bins because their squared
//! magnitudes are strictly positive while the remaining N-K bins have squared
//! magnitude zero; no nonzero bin can be displaced by a zero bin. The threshold
//! filter (default `threshold = 0.0`) passes every retained bin since each has
//! norm > 0. The only error is accumulated floating-point rounding in the
//! O(N log N) butterfly network, bounding per-coefficient error at
//! O(N log N · ε_machine). □

use crate::domain::plan::config::SparseFftConfig;
use crate::domain::spectrum::sparse::SparseSpectrum;
use apollo_fft::error::{ApolloError, ApolloResult};
use ndarray::Array1;
use num_complex::Complex64;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Ordering key for (magnitude_squared, frequency_index) in the top-K heap.
///
/// `Ord` compares magnitude ascending, then frequency index descending so that
/// equal-magnitude coefficients at higher indices are evicted first from a
/// `BinaryHeap<Reverse<MagIdx>>`. The K coefficients with the greatest magnitudes
/// are retained; ties are broken in favour of the lower frequency index,
/// matching the stable descending-magnitude ascending-index sort.
#[derive(PartialEq)]
struct MagIdx(f64, usize);

impl Eq for MagIdx {}

impl PartialOrd for MagIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MagIdx {
    /// Ascending magnitude; descending index for equal magnitudes.
    ///
    /// ### Theorem: Tie-Breaking Invariance
    ///
    /// **Statement.** For equal magnitudes |X[i]|² = |X[j]|², the lower-frequency
    /// index i < j is preferred. The K surviving entries are therefore the K
    /// largest-magnitude coefficients with, among ties, the smallest frequency
    /// indices — independent of insertion order.
    ///
    /// **Proof.** With `self = MagIdx(m, i)` and `other = MagIdx(m, j)`:
    ///
    /// ```text
    /// self.cmp(other) = m.total_cmp(m)       [Equal]
    ///                     .then_with(|| j.cmp(i))
    /// ```
    ///
    /// When i < j: `j.cmp(i)` = Greater → `MagIdx(m, i) > MagIdx(m, j)`.
    /// Lower frequency indices therefore map to *larger* `MagIdx` values.
    ///
    /// `BinaryHeap<Reverse<MagIdx>>::pop()` removes the *maximum* `Reverse`
    /// element, which equals the *minimum* `MagIdx`. Among equal-magnitude
    /// entries, minimum `MagIdx` is the entry with the highest index. That
    /// entry is evicted. After K elements remain, they carry the K largest
    /// magnitudes; among magnitude ties the lowest frequency indices survive. □
    ///
    /// For equal magnitudes m: `MagIdx(m, i).cmp(MagIdx(m, j)) = j.cmp(i)`,
    /// so higher indices sort smaller. `BinaryHeap<Reverse<MagIdx>>.pop()`
    /// evicts the minimum `MagIdx`, i.e. the smallest magnitude or, on ties,
    /// the highest index. The heap therefore retains K elements with the
    /// largest magnitudes, breaking ties by lowest frequency index.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// Sparse FFT plan.
#[derive(Clone, Debug)]
pub struct SparseFftPlan {
    config: SparseFftConfig,
}

impl SparseFftPlan {
    /// Create a new sparse FFT plan.
    pub fn new(n: usize, k: usize) -> ApolloResult<Self> {
        Ok(Self {
            config: SparseFftConfig::new(n, k)?,
        })
    }

    /// Return the validated plan configuration.
    #[must_use]
    pub const fn config(&self) -> SparseFftConfig {
        self.config
    }

    /// Return the signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.config.len()
    }

    /// Return whether the configured signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.config.is_empty()
    }

    /// Return the target sparsity.
    #[must_use]
    pub const fn sparsity(&self) -> usize {
        self.config.sparsity()
    }

    /// Return the bucket count used by the aliasing model.
    #[must_use]
    pub const fn bucket_count(&self) -> usize {
        self.config.bucket_count()
    }

    /// Return the number of recovery trials.
    #[must_use]
    pub const fn trials(&self) -> usize {
        self.config.trials()
    }

    /// Return the coefficient selection threshold.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.config.threshold()
    }

    /// Forward transform from a complex signal to a sparse spectrum.
    ///
    /// ## Complexity
    /// Spectrum density computation: O(N log N) via apollo-fft auto-selecting kernel.
    /// Top-K selection: O(N log K) via min-heap of size K.
    ///
    /// ## Recovery guarantee
    /// If the signal is K-sparse in the frequency domain, recovery is exact up to
    /// FFT numerical precision (O(N log N · ε_machine)); see module-level
    /// *Exact Recovery for K-Sparse Signals* theorem.
    /// For approximately sparse signals, the K retained coefficients minimise the
    /// squared time-domain reconstruction residual by Parseval's theorem; see
    /// module-level *Top-K Coefficient Selection Optimality* theorem.
    pub fn forward(&self, signal: &[Complex64]) -> ApolloResult<SparseSpectrum> {
        if signal.len() != self.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: self.len().to_string(),
                actual: signal.len().to_string(),
            });
        }

        // O(N log N) via apollo-fft auto-selecting kernel.
        let dense: Vec<Complex64> = {
            let mut arr = Array1::from_vec(signal.to_vec());
            apollo_fft::fft_1d_complex_inplace(&mut arr);
            let (data, offset) = arr.into_raw_vec_and_offset();
            debug_assert_eq!(offset.unwrap_or(0), 0);
            data
        };

        // O(N log K) top-K selection via min-heap of size K.
        // Invariant: heap holds at most K entries. heap.pop() evicts the entry
        // with the smallest magnitude (highest index on tie) -- the K-th best.
        let k = self.sparsity();
        let mut heap: BinaryHeap<Reverse<MagIdx>> = BinaryHeap::with_capacity(k + 1);
        for (i, coeff) in dense.iter().enumerate() {
            heap.push(Reverse(MagIdx(coeff.norm_sqr(), i)));
            if heap.len() > k {
                heap.pop();
            }
        }

        // Collect in ascending frequency order and apply threshold filter.
        let mut top_k: Vec<(usize, Complex64)> = heap
            .into_iter()
            .map(|Reverse(MagIdx(_, idx))| (idx, dense[idx]))
            .collect();
        top_k.sort_by_key(|&(idx, _)| idx);

        let mut spectrum = SparseSpectrum::new(self.len());
        for (frequency, value) in top_k {
            if value.norm() > self.threshold() {
                spectrum.insert(frequency, value)?;
            }
        }

        spectrum.validate()?;
        Ok(spectrum)
    }

    /// Inverse transform from sparse spectrum to a dense complex signal.
    ///
    /// Uses FFTW-compatible normalised inverse FFT (divides by N), matching
    /// the standard IDFT: x_n = (1/N) sum_k X_k exp(2 pi i k n / N).
    pub fn inverse(&self, spectrum: &SparseSpectrum) -> ApolloResult<Vec<Complex64>> {
        spectrum.validate()?;
        if spectrum.n != self.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: self.len().to_string(),
                actual: spectrum.n.to_string(),
            });
        }

        let mut arr = Array1::from_vec(spectrum.to_dense());
        apollo_fft::ifft_1d_complex_inplace(&mut arr);
        let (data, offset) = arr.into_raw_vec_and_offset();
        debug_assert_eq!(offset.unwrap_or(0), 0);
        Ok(data)
    }

    /// Return the retained support as a list of (frequency, coefficient) pairs.
    #[must_use]
    pub fn support(&self, spectrum: &SparseSpectrum) -> Vec<(usize, Complex64)> {
        spectrum
            .frequencies
            .iter()
            .copied()
            .zip(spectrum.values.iter().copied())
            .collect()
    }
}
