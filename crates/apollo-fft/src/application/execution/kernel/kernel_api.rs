//! SSOT macro for the 18-function public API shared by all Winograd-based radix-K modules.
//!
//! ## Pattern
//!
//! Each Winograd-based radix module (radix8, radix16, radix32, radix64) exposes the same
//! 18-function public surface:
//! - 6 typed (`Complex64`, `Complex32`) functions: forward + inverse-unnorm + inverse, each with
//!   and without caller-supplied twiddle tables.
//! - 6 analogous `Cf16` wrappers that promote to `Complex32` via `f16_bridge`.
//!
//! All bodies are structurally identical; only the power-of-radix predicate and the private inner
//! driver functions vary.  `radix_kernel_api!` generates the complete API from those three
//! identifiers, eliminating ~180 lines of repeated boilerplate per module.
//!
//! ## Usage
//!
//! ```ignore
//! kernel_api::radix_kernel_api! {
//!     check      = is_power_of_eight,
//!     inplace64  = winograd_r8_inplace_64,
//!     inplace32  = winograd_r8_inplace_32,
//!     description = "power-of-eight",
//! }
//! ```
//!
//! The macro must be invoked inside the module that defines the private `$inplace64` and
//! `$inplace32` functions and has the standard `use` imports in scope:
//! - `super::f16_bridge::{run_f16_via_f32, run_f16_via_f32_with_twiddles}`
//! - `super::radix2`
//! - `super::radix2_f16::Cf16`
//! - `num_complex::{Complex32, Complex64}`

/// Generate the 18-function public API for a Winograd-based radix-K FFT kernel.
///
/// Parameters:
/// - `check`: power-of-radix predicate visible in the same module scope.
/// - `inplace64`: private f64 inner driver with signature
///   `fn(data: &mut [Complex64], inverse: bool, twiddles: Option<&[Complex64]>)`.
/// - `inplace32`: private f32 inner driver with analogous signature.
/// - `description`: literal string for doc comments, e.g. `"power-of-eight"`.
macro_rules! radix_kernel_api {
    (
        check       = $check:expr,
        inplace64   = $inplace64:ident,
        inplace32   = $inplace32:ident,
        description = $desc:literal $(,)?
    ) => {
        // ── Complex64 with caller-supplied twiddles ────────────────────────────

        #[doc = concat!("Forward FFT (unnormalized) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace64(data, false, Some(twiddles));
        }

        #[doc = concat!("Inverse FFT (unnormalized) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace64(data, true, Some(twiddles));
        }

        #[doc = concat!("Inverse FFT normalized by 1/N for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace64(data, true, Some(twiddles));
            super::radix_stage::normalize_inplace(data, 1.0 / data.len() as f64);
        }

        // ── Complex64 without caller-supplied twiddles ─────────────────────────

        #[doc = concat!("Forward FFT (unnormalized) for ", $desc, " lengths.")]
        pub fn forward_inplace_64(data: &mut [Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_fwd_64(data.len());
            $inplace64(data, false, Some(twiddles.as_ref()));
        }

        #[doc = concat!("Inverse FFT (unnormalized) for ", $desc, " lengths.")]
        pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_inv_64(data.len());
            $inplace64(data, true, Some(twiddles.as_ref()));
        }

        #[doc = concat!("Inverse FFT normalized by 1/N for ", $desc, " lengths.")]
        pub fn inverse_inplace_64(data: &mut [Complex64]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_inv_64(data.len());
            $inplace64(data, true, Some(twiddles.as_ref()));
            super::radix_stage::normalize_inplace(data, 1.0 / data.len() as f64);
        }

        // ── Complex32 with caller-supplied twiddles ────────────────────────────

        #[doc = concat!("Forward FFT (unnormalized, f32) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace32(data, false, Some(twiddles));
        }

        #[doc = concat!("Inverse FFT (unnormalized, f32) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace32(data, true, Some(twiddles));
        }

        #[doc = concat!("Inverse FFT normalized by 1/N (f32) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            $inplace32(data, true, Some(twiddles));
            super::radix_stage::normalize_inplace(data, 1.0f32 / data.len() as f32);
        }

        // ── Complex32 without caller-supplied twiddles ─────────────────────────

        #[doc = concat!("Forward FFT (unnormalized, f32) for ", $desc, " lengths.")]
        pub fn forward_inplace_32(data: &mut [Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_fwd_32(data.len());
            $inplace32(data, false, Some(twiddles.as_ref()));
        }

        #[doc = concat!("Inverse FFT (unnormalized, f32) for ", $desc, " lengths.")]
        pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_inv_32(data.len());
            $inplace32(data, true, Some(twiddles.as_ref()));
        }

        #[doc = concat!("Inverse FFT normalized by 1/N (f32) for ", $desc, " lengths.")]
        pub fn inverse_inplace_32(data: &mut [Complex32]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            let twiddles = super::mixed_radix::cached_twiddle_inv_32(data.len());
            $inplace32(data, true, Some(twiddles.as_ref()));
            super::radix_stage::normalize_inplace(data, 1.0f32 / data.len() as f32);
        }

        // ── Cf16 with caller-supplied twiddles ─────────────────────────────────

        #[doc = concat!("Forward FFT (unnormalized, f16 storage) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn forward_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32_with_twiddles(data, twiddles, forward_inplace_32_with_twiddles);
        }

        #[doc = concat!("Inverse FFT (unnormalized, f16 storage) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_unnorm_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32_with_twiddles(data, twiddles, inverse_inplace_unnorm_32_with_twiddles);
        }

        #[doc = concat!("Inverse FFT normalized by 1/N (f16 storage) for ", $desc, " lengths using caller-provided twiddles.")]
        #[inline]
        pub fn inverse_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32_with_twiddles(data, twiddles, inverse_inplace_32_with_twiddles);
        }

        // ── Cf16 without caller-supplied twiddles ──────────────────────────────

        #[doc = concat!("Forward FFT (unnormalized, f16 storage) for ", $desc, " lengths.")]
        pub fn forward_inplace_f16(data: &mut [Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32(data, forward_inplace_32);
        }

        #[doc = concat!("Inverse FFT (unnormalized, f16 storage) for ", $desc, " lengths.")]
        pub fn inverse_inplace_unnorm_f16(data: &mut [Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32(data, inverse_inplace_unnorm_32);
        }

        #[doc = concat!("Inverse FFT normalized by 1/N (f16 storage) for ", $desc, " lengths.")]
        pub fn inverse_inplace_f16(data: &mut [Cf16]) {
            if data.len() <= 1 { return; }
            debug_assert!($check(data.len()));
            super::f16_bridge::run_f16_via_f32(data, inverse_inplace_32);
        }
    };
}

pub(crate) use radix_kernel_api;
