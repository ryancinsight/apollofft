//! Execution policy abstractions for zero-cost static dispatch over parallelism regimes.

use rayon::prelude::*;

use rayon::prelude::*;
use rayon::prelude::*;
use std::future::Future;

/// Zero-cost execution policy trait mapping parallelism strategies to compile-time
/// monomorphized variants.
pub trait ExecutionPolicy: Send + Sync + 'static {
    /// Async abstraction
    type Future<T: Send>: Future<Output = T> + Send;

    /// Iterates over mutable chunks of `data`, passing `(index, chunk)` to `f`.
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync;
}

/// Sequential (synchronous) execution policy.
pub struct SyncPolicy;

impl ExecutionPolicy for SyncPolicy {
    type Future<T: Send> = std::future::Ready<T>;

    #[inline(always)]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}

/// Data-parallel execution policy using Rayon.
pub struct ParallelPolicy;

impl ExecutionPolicy for ParallelPolicy {
    // Parallel policy doesn't change the async await behavior here, just computation.
    // For full compliance, a Tokio threadpool future could be used, but Ready is fine for now.
    type Future<T: Send> = std::future::Ready<T>;

    #[inline(always)]
    fn for_each_chunk_mut_enumerated<T, F>(data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send + Sync,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}
