//! Mixed-radix strategy facade.
//!
//! Routes in-place FFTs to the best available kernel for the given length:
//!
//! ## Dispatch hierarchy (f64 / f32)
//!
//! | Input length       | Kernel selected |
//! |--------------------|------------------|
//! | Power of two >= 2   | **Stockham autosort** - out-of-place ping-pong FFT between `data` and a thread-local scratch buffer; no bit-reversal permutation. AVX/FMA SIMD codelets for N = 4, 8, 64, 4096 (f32); N = 64 (f64). Falls back to generic `transform<F>` loop for other PoT sizes. |
//! | 2/3/5/7-smooth     | **Composite mixed-radix** - Cooley-Tukey DIT with digit-reversal permutation. |
//! | Other non-PoT      | **Bluestein chirp-Z** - pads to next PoT and runs Stockham internally. |
//!
//! ## Dispatch hierarchy (f16)
//!
//! `Complex<f16>` is a storage-only precision. All PoT sizes are promoted to f32, run
//! through the Stockham f32 kernel, and demoted back to f16 via
//! `run_via_complex32`. The bit-reversal radix2/radix4 kernels are **not** used
//! on the PoT path for any precision.
//!
//! ## SSOT principle
//!
//! Each precision x operation combination delegates to one of the above three
//! authoritative algorithm implementations. No algorithm body is duplicated
//! across precision variants.
//!
//! ## Precision strategy
//!
//! | Variant | Element type | Notes |
//! |---------|--------------|-------|
//! | `_64`   | `Complex64`  | Native f64; Stockham for PoT. |
//! | `_32`   | `Complex32`  | Native f32; Stockham for PoT. |
//! | `_f16`  | `Complex<f16>` | Promotes to f32 for CPU arithmetic; Stockham f32 for PoT. |

#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::uninit_vec)]

use super::precision_bridge::{run_via_complex32, Complex32Bridge};
use super::radix_shape::{factorize_composite, should_use_bluestein_instead_of_composite};
use super::radix_stage::{normalize_inplace, normalize_inplace_c32, normalize_inplace_c64};
use super::{bluestein, radix2, radix_composite, stockham, winograd};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;


pub(crate) mod caches;
pub(crate) mod traits;
pub(crate) mod dispatch_f64;
pub(crate) mod dispatch_f32;
pub(crate) mod dispatch_f16;
#[cfg(test)]
mod tests;

pub(crate) use caches::*;
pub(crate) use traits::*;
pub use dispatch_f64::*;
pub use dispatch_f32::*;
pub(crate) use dispatch_f16::*;
