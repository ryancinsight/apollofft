//! Shared cache implementations for reusable FFT plans.

use crate::application::plan::{FftPlan1D, FftPlan2D, FftPlan3D};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Cache key for 1D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft1dCacheKey(pub usize);

/// Cache key for 2D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft2dCacheKey(pub usize, pub usize);

/// Cache key for 3D plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft3dCacheKey {
    /// X dimension.
    pub nx: usize,
    /// Y dimension.
    pub ny: usize,
    /// Z dimension.
    pub nz: usize,
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
        let key = Fft1dCacheKey(n);
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan1D::new(n));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

impl Fft2dCache {
    /// Return the cached plan or create it on first use.
    pub fn get_or_create(&self, nx: usize, ny: usize) -> Arc<FftPlan2D> {
        let key = Fft2dCacheKey(nx, ny);
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan2D::new(nx, ny));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

impl Fft3dCache {
    /// Return the cached plan or create it on first use.
    pub fn get_or_create(&self, nx: usize, ny: usize, nz: usize) -> Arc<FftPlan3D> {
        let key = Fft3dCacheKey { nx, ny, nz };
        if let Some(plan) = self.cache.read().get(&key) {
            return Arc::clone(plan);
        }
        let mut guard = self.cache.write();
        if let Some(plan) = guard.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(FftPlan3D::new(nx, ny, nz));
        guard.insert(key, Arc::clone(&plan));
        plan
    }
}

/// Process-wide 1D plan cache.
pub static FFT_CACHE_1D: Lazy<Fft1dCache> = Lazy::new(|| Fft1dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Process-wide 2D plan cache.
pub static FFT_CACHE_2D: Lazy<Fft2dCache> = Lazy::new(|| Fft2dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Process-wide 3D plan cache.
pub static FFT_CACHE_3D: Lazy<Fft3dCache> = Lazy::new(|| Fft3dCache {
    cache: RwLock::new(HashMap::new()),
});

/// Legacy compatibility alias for the 3D cache.
pub static FFT_CACHE: Lazy<Fft3dCache> = Lazy::new(|| Fft3dCache {
    cache: RwLock::new(HashMap::new()),
});

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
}

