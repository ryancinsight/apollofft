#![warn(missing_docs)]
//! DCT and DST real-to-real transform plans for Apollo.
//!
//! `apollo-dctdst` owns real-to-real cosine/sine transform metadata, direct
//! DCT-II/DCT-III/DST-II/DST-III kernels, and value-semantic verification.
//! These transforms encode even and odd boundary extensions used in image
//! compression, spectral methods, and compact real-domain analysis.

/// Application-layer DCT/DST plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::dctdst::DctDstPlan;
pub use domain::contracts::error::{DctDstError, DctDstResult};
pub use domain::metadata::kind::{RealTransformConfig, RealTransformKind};
pub use infrastructure::kernel::direct::{dct2, dct3, dst2, dst3};
