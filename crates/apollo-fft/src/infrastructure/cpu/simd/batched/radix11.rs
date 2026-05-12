//! Batched radix-11 native-precision kernel entry points.

use crate::domain::storage::FftStorage;

/// Execute eight independent planar radix-11 DFT instances in f32.
pub fn radix11_batched_f32<S: FftStorage<f32>>(storage: &mut S, inverse: bool) {
    super::prime::prime_batched_f32::<11, S>(storage, inverse);
}
