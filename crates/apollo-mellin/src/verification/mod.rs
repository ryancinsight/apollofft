//! Verification modules for Mellin.

#[cfg(test)]
mod tests {
    use crate::{calculate_log_resample, mellin_moment, MellinError, MellinPlan};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn plan_preserves_scale_configuration() {
        let plan = MellinPlan::new(128, 0.25, 4.0).expect("plan");
        assert_eq!(plan.config().samples(), 128);
        assert_eq!(plan.config().min_scale(), 0.25);
        assert_eq!(plan.config().max_scale(), 4.0);
    }

    #[test]
    fn rejects_invalid_scale_configuration() {
        assert_eq!(
            MellinPlan::new(0, 0.25, 4.0).unwrap_err(),
            MellinError::EmptySampleCount
        );
        assert_eq!(
            MellinPlan::new(8, f64::NAN, 4.0).unwrap_err(),
            MellinError::InvalidScaleBound
        );
        assert_eq!(
            MellinPlan::new(8, 4.0, 4.0).unwrap_err(),
            MellinError::InvalidScaleOrder
        );
    }

    #[test]
    fn log_resample_matches_linear_input_at_scale_grid_endpoints() {
        let plan = MellinPlan::new(3, 1.0, 4.0).expect("plan");
        let signal = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 3];
        plan.forward_resample(&signal, 1.0, 4.0, &mut output)
            .expect("resample");

        assert_abs_diff_eq!(output[0], 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(output[2], 4.0, epsilon = 1.0e-12);
        assert!(output[1] > output[0]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn constant_signal_moment_matches_analytical_integral() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let signal = [2.0; 65];
        let moment = plan.moment(&signal, 1.0, 4.0, 1.0).expect("moment");

        assert_abs_diff_eq!(moment, 6.0, epsilon = 1.0e-12);
    }

    #[test]
    fn power_signal_moment_matches_analytical_integral() {
        let min = 1.0;
        let max = 3.0;
        let samples = 4097;
        let step = (max - min) / (samples as f64 - 1.0);
        let signal: Vec<f64> = (0..samples)
            .map(|index| {
                let coordinate = min + index as f64 * step;
                coordinate * coordinate
            })
            .collect();
        let plan = MellinPlan::new(16, min, max).expect("plan");
        let moment = plan.moment(&signal, min, max, 1.0).expect("moment");
        let expected = (max.powi(3) - min.powi(3)) / 3.0;

        assert_abs_diff_eq!(moment, expected, epsilon = 1.0e-7);
    }

    #[test]
    fn constant_log_frequency_spectrum_has_only_dc_component() {
        let plan = MellinPlan::new(8, 1.0, 8.0).expect("plan");
        let signal = [5.0; 8];
        let spectrum = plan.forward_spectrum(&signal, 1.0, 8.0).expect("spectrum");

        assert_abs_diff_eq!(spectrum.values()[0].im, 0.0, epsilon = 1.0e-12);
        for coefficient in &spectrum.values()[1..] {
            assert_abs_diff_eq!(coefficient.norm(), 0.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_execution_contracts() {
        let plan = MellinPlan::new(4, 1.0, 2.0).expect("plan");
        let mut output = [0.0; 3];

        assert_eq!(
            plan.forward_resample(&[1.0], 1.0, 2.0, &mut output)
                .unwrap_err(),
            MellinError::LengthMismatch
        );
        assert_eq!(
            plan.moment(&[], 1.0, 2.0, 1.0).unwrap_err(),
            MellinError::EmptySignal
        );
        assert_eq!(
            plan.moment(&[1.0], 0.0, 2.0, 1.0).unwrap_err(),
            MellinError::InvalidSignalBound
        );
        assert_eq!(
            plan.moment(&[1.0], 2.0, 1.0, 1.0).unwrap_err(),
            MellinError::InvalidSignalOrder
        );
        assert_eq!(
            plan.moment(&[1.0], 1.0, 2.0, f64::NAN).unwrap_err(),
            MellinError::InvalidExponent
        );
    }

    proptest! {
        /// Log-resample of a uniform signal returns uniform values within interpolation error.
        #[test]
        fn log_resample_uniform_signal_is_uniform(
            constant in 0.1_f64..2.0_f64,
            n in 4_usize..33_usize,
        ) {
            // Signal domain [0.5, 20.0] strictly contains the scale grid [1.0, 10.0].
            // This prevents exp(ln(max_scale)) from landing epsilon above signal_max
            // in floating-point, which would otherwise return 0 for the boundary sample.
            let signal: Vec<f64> = vec![constant; n];
            let mut result = vec![0.0_f64; n];
            calculate_log_resample(&signal, 0.5, 20.0, &mut result, 1.0, 10.0);
            // All log-spaced samples in [1.0, 10.0] are strictly inside [0.5, 20.0],
            // so linear interpolation on a uniform signal returns constant exactly.
            let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
            for &v in &result {
                prop_assert!(
                    (v - mean).abs() < 1.0e-10,
                    "non-uniform result for constant signal: v={}, mean={}",
                    v,
                    mean
                );
            }
        }

        /// Trapezoidal Mellin moment of r^alpha satisfies M(s) = (max^(alpha+s) - min^(alpha+s)) / (alpha+s).
        ///
        /// Mathematical basis: integral_min^max r^alpha * r^(s-1) dr
        ///   = integral r^(alpha+s-1) dr = [r^(alpha+s) / (alpha+s)]_min^max.
        #[test]
        fn power_law_moment_analytical(
            alpha in 0.1_f64..3.0_f64,
            s_real in 0.5_f64..2.5_f64,
        ) {
            let n = 64_usize;
            let min_scale = 1.0_f64;
            let max_scale = 3.0_f64;
            let step = (max_scale - min_scale) / (n - 1) as f64;
            let signal: Vec<f64> = (0..n)
                .map(|i| (min_scale + i as f64 * step).powf(alpha))
                .collect();
            let computed = mellin_moment(&signal, min_scale, max_scale, s_real);
            let exponent = alpha + s_real;
            let analytical = (max_scale.powf(exponent) - min_scale.powf(exponent)) / exponent;
            prop_assert!(
                (computed - analytical).abs() / analytical.abs() < 0.05,
                "power law moment: computed={}, analytical={}",
                computed,
                analytical
            );
        }
    }
}
