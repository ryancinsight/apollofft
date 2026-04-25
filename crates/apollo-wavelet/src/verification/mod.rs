//! Verification modules for wavelet transforms.

#[cfg(test)]
mod tests {
    use crate::{ContinuousWavelet, CwtPlan, DiscreteWavelet, DwtPlan, WaveletError};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn haar_single_level_matches_analytical_coefficients() {
        let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar).expect("plan");
        let coefficients = plan.forward(&[1.0, 3.0, 2.0, 0.0]).expect("forward");
        let scale = std::f64::consts::FRAC_1_SQRT_2;

        assert_abs_diff_eq!(
            coefficients.approximation()[0],
            4.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            coefficients.approximation()[1],
            2.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            coefficients.details()[0][0],
            -2.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(coefficients.details()[0][1], 2.0 * scale, epsilon = 1.0e-12);
    }

    #[test]
    fn dwt_inverse_recovers_signal_for_supported_wavelets() {
        let signal = [1.0, -2.0, 0.5, 4.0, -1.25, 0.75, 3.0, -0.5];
        for wavelet in [DiscreteWavelet::Haar, DiscreteWavelet::Daubechies4] {
            let plan = DwtPlan::new(signal.len(), 3, wavelet).expect("plan");
            let coefficients = plan.forward(&signal).expect("forward");
            let recovered = plan.inverse(&coefficients).expect("inverse");
            for (actual, expected) in recovered.iter().zip(signal.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-10);
            }
        }
    }

    #[test]
    fn dwt_rejects_invalid_contracts() {
        assert_eq!(
            DwtPlan::new(0, 1, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::EmptySignal
        );
        assert_eq!(
            DwtPlan::new(6, 1, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::NonPowerOfTwoLength
        );
        assert_eq!(
            DwtPlan::new(8, 0, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::EmptyLevelCount
        );
        assert_eq!(
            DwtPlan::new(8, 4, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::LevelExceedsLength
        );
        let plan = DwtPlan::new(8, 1, DiscreteWavelet::Haar).expect("plan");
        assert_eq!(
            plan.forward(&[1.0, 2.0]).unwrap_err(),
            WaveletError::LengthMismatch
        );
    }

    #[test]
    fn cwt_impulse_response_peaks_at_impulse_shift() {
        let signal = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let plan = CwtPlan::new(7, vec![1.0], ContinuousWavelet::Ricker).expect("plan");
        let coefficients = plan.transform(&signal).expect("cwt");
        let row = coefficients.values().row(0);
        let (max_index, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.abs().partial_cmp(&rhs.abs()).unwrap())
            .expect("non-empty row");
        assert_eq!(max_index, 3);
    }

    #[test]
    fn cwt_rejects_invalid_contracts() {
        assert_eq!(
            CwtPlan::new(0, vec![1.0], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::EmptySignal
        );
        assert_eq!(
            CwtPlan::new(8, vec![], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::EmptyScales
        );
        assert_eq!(
            CwtPlan::new(8, vec![0.0], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::InvalidScale
        );
        let plan = CwtPlan::new(8, vec![1.0], ContinuousWavelet::Ricker).expect("plan");
        assert_eq!(
            plan.transform(&[1.0, 2.0]).unwrap_err(),
            WaveletError::LengthMismatch
        );
    }

    proptest! {
        #[test]
        fn haar_roundtrip_holds_for_random_power_of_two_signals(
            data in prop::collection::vec(-10.0f64..10.0f64, 8)
        ) {
            let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("plan");
            let coefficients = plan.forward(&data).expect("forward");
            let recovered = plan.inverse(&coefficients).expect("inverse");
            for (actual, expected) in recovered.iter().zip(data.iter()) {
                prop_assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn morlet_cwt_coefficients_are_finite() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.5).sin()).collect();
        let scales = vec![1.0, 2.0, 4.0];
        let plan = CwtPlan::new(
            signal.len(),
            scales,
            ContinuousWavelet::Morlet { omega0: 5.0 },
        )
        .unwrap();
        let coeffs = plan.transform(&signal).unwrap();
        for &c in coeffs.values().iter() {
            assert!(c.is_finite(), "CWT coefficient is not finite: {c}");
        }
    }

    #[test]
    fn morlet_mother_wavelet_has_bounded_dc_response() {
        let omega0 = 5.0;
        let step = 0.01;
        let support = 12.0;
        let sample_count = (2.0 * support / step) as usize + 1;
        let integral: f64 = (0..sample_count)
            .map(|i| {
                let t = -support + i as f64 * step;
                crate::infrastructure::kernel::continuous::mother_wavelet(
                    ContinuousWavelet::Morlet { omega0 },
                    t,
                ) * step
            })
            .sum();
        assert_abs_diff_eq!(integral, 0.0, epsilon = 1.0e-10);
    }

    #[test]
    fn ricker_cwt_zero_signal_gives_zero_coefficients() {
        let signal = vec![0.0_f64; 16];
        let scales = vec![1.0, 2.0];
        let plan = CwtPlan::new(signal.len(), scales, ContinuousWavelet::Ricker).unwrap();
        let coeffs = plan.transform(&signal).unwrap();
        for &c in coeffs.values().iter() {
            assert!(
                c.abs() < 1e-14,
                "Zero signal should give zero coefficients, got {c}"
            );
        }
    }
}
