//! Batched radix-3 native-precision kernel entry points.

use crate::domain::storage::FftStorage;

/// Execute eight independent planar radix-3 DFT instances in f32.
pub fn radix3_batched_f32<S: FftStorage<f32>>(storage: &mut S, inverse: bool) {
    super::prime::prime_batched_f32::<3, S>(storage, inverse);
}
