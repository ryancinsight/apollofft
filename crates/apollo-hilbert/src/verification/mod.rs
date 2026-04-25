//! Verification modules for the Hilbert transform.

#[cfg(test)]
mod tests {
    use crate::{HilbertError, HilbertPlan};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn cosine_maps_to_positive_sine_quadrature() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let expected: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).sin())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let quadrature = plan.transform(&signal).expect("hilbert");

        for (actual, expected) in quadrature.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-11);
        }
    }

    #[test]
    fn analytic_signal_preserves_real_part_and_unit_cosine_envelope() {
        let len = 32;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * 3.0 * n as f64 / len as f64).cos())
            .collect();
        let plan = HilbertPlan::new(len).expect("plan");
        let analytic = plan.analytic_signal(&signal).expect("analytic");
        let envelope = analytic.envelope();

        for ((sample, original), envelope) in analytic
            .values()
            .iter()
            .zip(signal.iter())
            .zip(envelope.iter())
        {
            assert_abs_diff_eq!(sample.re, *original, epsilon = 1.0e-12);
            assert_abs_diff_eq!(*envelope, 1.0, epsilon = 1.0e-11);
        }
    }

    #[test]
    fn constant_signal_has_zero_quadrature_and_constant_envelope() {
        let signal = [4.0; 9];
        let plan = HilbertPlan::new(signal.len()).expect("plan");
        let analytic = plan.analytic_signal(&signal).expect("analytic");

        for sample in analytic.values() {
            assert_abs_diff_eq!(sample.re, 4.0, epsilon = 1.0e-12);
            assert_abs_diff_eq!(sample.im, 0.0, epsilon = 1.0e-12);
            assert_abs_diff_eq!(sample.norm(), 4.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn even_length_nyquist_component_has_zero_quadrature() {
        let signal: Vec<f64> = (0..10)
            .map(|n| if n % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let plan = HilbertPlan::new(signal.len()).expect("plan");
        let quadrature = plan.transform(&signal).expect("hilbert");

        for value in quadrature {
            assert_abs_diff_eq!(value, 0.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_contracts() {
        assert_eq!(HilbertPlan::new(0).unwrap_err(), HilbertError::EmptySignal);
        let plan = HilbertPlan::new(4).expect("plan");
        assert_eq!(
            plan.transform(&[1.0, 2.0]).unwrap_err(),
            HilbertError::LengthMismatch
        );
    }

    proptest! {
        #[test]
        fn analytic_real_part_matches_input_for_random_real_signals(
            data in prop::collection::vec(-10.0f64..10.0f64, 1..17)
        ) {
            let plan = HilbertPlan::new(data.len()).expect("plan");
            let analytic = plan.analytic_signal(&data).expect("analytic");
            for (actual, expected) in analytic.values().iter().zip(data.iter()) {
                prop_assert!((actual.re - expected).abs() < 1.0e-10);
            }
        }
    }
}
