//! Sparse FFT plan configuration.

use apollo_fft::error::{ApolloError, ApolloResult};

/// Validated sparse FFT configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseFftConfig {
    n: usize,
    k: usize,
    bucket_count: usize,
    trials: usize,
    threshold: f64,
}

impl SparseFftConfig {
    /// Create a validated sparse FFT configuration.
    ///
    /// The bucket count is `min(max(4k, 1), n)`. The number of isolation
    /// trials is `max(4, floor(log2(n)) + 1)`.
    pub fn new(n: usize, k: usize) -> ApolloResult<Self> {
        if n == 0 {
            return Err(ApolloError::validation(
                "n",
                n.to_string(),
                "signal length must be non-zero",
            ));
        }
        if k == 0 {
            return Err(ApolloError::validation(
                "k",
                k.to_string(),
                "sparsity must be non-zero",
            ));
        }

        Ok(Self {
            n,
            k,
            bucket_count: (4 * k).max(1).min(n),
            trials: (n.ilog2() as usize + 1).max(4),
            threshold: 0.0,
        })
    }

    /// Create a sparse FFT configuration with an explicit threshold.
    ///
    /// `n` is the signal length, `k` is the max coefficients to retain, and
    /// `threshold` is the minimum magnitude a coefficient must exceed to be
    /// retained (0.0 = keep top-K regardless, excluding exact zero).
    pub fn new_with_threshold(n: usize, k: usize, threshold: f64) -> ApolloResult<Self> {
        let mut cfg = Self::new(n, k)?;
        cfg.threshold = threshold;
        Ok(cfg)
    }
    /// Return the signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the configured signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Return the target sparsity.
    #[must_use]
    pub const fn sparsity(self) -> usize {
        self.k
    }

    /// Return the aliasing bucket count.
    #[must_use]
    pub const fn bucket_count(self) -> usize {
        self.bucket_count
    }

    /// Return the number of deterministic recovery trials.
    #[must_use]
    pub const fn trials(self) -> usize {
        self.trials
    }

    /// Return the coefficient selection threshold.
    #[must_use]
    pub const fn threshold(self) -> f64 {
        self.threshold
    }
}
