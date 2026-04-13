//! Shared cache implementations for reusable FFT plans.

use crate::application::plan::{FftPlan1D, FftPlan2D, FftPlan3D};
use crate::types::PrecisionProfile;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Cache key for 1D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft1dCacheKey(pub usize, pub PrecisionProfile);

/// Cache key for 2D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft2dCacheKey(pub usize, pub usize, pub PrecisionProfile);

/// Cache key for 3D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft3dCacheKey {
    /// X dimension.
    pub nx: usize,
    /// Y dimension.
    pub ny: usize,
    /// Z dimension.
    pub nz: usize,
    /// Precision profile.
    pub precision: PrecisionProfile,
}

/// Shared 1D plan cache.
pub struct Fft1dCache {
    cache: RwLock<HashMap<Fft1dCacheKey, Arc<FftPlan1D>>>,
}

/// Shared 2D plan cache.
pub struct Fft2dCache {
    cache: RwLock<HashMap<Fft2dCacheKey, Arc<FftPlan2D>>>,
}

/// Shared 3D plan cache.
pub struct Fft3dCache {
    cache: RwLock<HashMap<Fft3dCacheKey, Arc<FftPlan3D>>>,
}

impl std::fmt::Debug for Fft1dCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft1dCache")
            .field("cache_size", &self.cache.read().len())
            .finish()
    }
}

impl std::fmt::Debug for Fft2dCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft2dCache")
            .field("cache_size", &self.cache.read().len())
            .finish()
    }
}

impl std::fmt::Debug for Fft3dCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft3dCache")
            .field("cache_size", &self.cache.read().len())
            .finish()
    }
}

impl Fft1dCache {
    /// Return the cached plan or create it on first use.
    pub fn get_or_create(&self, n: usize) -> Arc<FftPlan1D> {
        self.get_or_create_with_precision(n, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Return the cached plan or create it on first use for an explicit precision profile.
    pub fn get_or_create_with_precision(
        &self,
        n: usize,
        precision: PrecisionProfile,
    ) -> Arc<FftPlan1D> {
        let key = Fft1dCacheKey(n, precision);
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan1D::with_precision(n, precision));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

impl Fft2dCache {
    /// Return the cached plan or create it on first use.
    pub fn get_or_create(&self, nx: usize, ny: usize) -> Arc<FftPlan2D> {
        self.get_or_create_with_precision(nx, ny, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Return the cached plan or create it on first use for an explicit precision profile.
    pub fn get_or_create_with_precision(
        &self,
        nx: usize,
        ny: usize,
        precision: PrecisionProfile,
    ) -> Arc<FftPlan2D> {
        let key = Fft2dCacheKey(nx, ny, precision);
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan2D::with_precision(nx, ny, precision));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

impl Fft3dCache {
    /// Return the cached plan or create it on first use.
    pub fn get_or_create(&self, nx: usize, ny: usize, nz: usize) -> Arc<FftPlan3D> {
        self.get_or_create_with_precision(nx, ny, nz, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Return the cached plan or create it on first use for an explicit precision profile.
    pub fn get_or_create_with_precision(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        precision: PrecisionProfile,
    ) -> Arc<FftPlan3D> {
        let key = Fft3dCacheKey {
            nx,
            ny,
            nz,
            precision,
        };
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan3D::with_precision(nx, ny, nz, precision));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

/// Process-wide 1D plan cache.
pub static FFT_CACHE_1D: std::sync::LazyLock<Fft1dCache> = std::sync::LazyLock::new(|| Fft1dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Process-wide 2D plan cache.
pub static FFT_CACHE_2D: std::sync::LazyLock<Fft2dCache> = std::sync::LazyLock::new(|| Fft2dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Process-wide 3D plan cache.
pub static FFT_CACHE_3D: std::sync::LazyLock<Fft3dCache> = std::sync::LazyLock::new(|| Fft3dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Legacy compatibility alias for the 3D cache.
pub use FFT_CACHE_3D as FFT_CACHE;

/// Return a cached 3D plan for a given grid.
pub fn get_fft_for_grid(nx: usize, ny: usize, nz: usize) -> Arc<FftPlan3D> {
    FFT_CACHE_3D.get_or_create(nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_cached_3d_plan() {
        let lhs = FFT_CACHE_3D.get_or_create(8, 8, 8);
        let rhs = FFT_CACHE_3D.get_or_create(8, 8, 8);
        assert!(Arc::ptr_eq(&lhs, &rhs));
    }

    #[test]
    fn separates_cache_entries_by_precision() {
        let low =
            FFT_CACHE_3D.get_or_create_with_precision(8, 8, 8, PrecisionProfile::LOW_PRECISION_F32);
        let mixed = FFT_CACHE_3D.get_or_create_with_precision(
            8,
            8,
            8,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        );
        assert!(!Arc::ptr_eq(&low, &mixed));
    }
}
