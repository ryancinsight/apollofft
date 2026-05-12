use super::*;
use approx::assert_relative_eq;
use half::f16;
use proptest::prelude::*;
use proptest::proptest;

fn energy_time_domain(signal: &Array1<f64>) -> f64 {
    signal.iter().map(|value| value * value).sum()
}

fn energy_frequency_domain(spectrum: &Array1<Complex64>) -> f64 {
    spectrum.iter().map(|value| value.norm_sqr()).sum()
}

#[test]
fn forward_is_linear() {
    let shape = Shape1D::new(16).expect("shape");
    let plan = FftPlan1D::new(shape);
    let lhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.2).sin()));
    let rhs = Array1::from_iter((0..16).map(|i| (i as f64 * 0.3).cos()));
    let alpha = 1.75;
    let beta = -0.25;
    let combined = &lhs * alpha + &rhs * beta;

    let combined_hat = plan.forward_real_to_complex(&combined);
    let lhs_hat = plan.forward_real_to_complex(&lhs);
    let rhs_hat = plan.forward_real_to_complex(&rhs);

    for (actual, expected) in combined_hat.iter().zip(
        lhs_hat
            .iter()
            .zip(rhs_hat.iter())
            .map(|(x, y)| *x * alpha + *y * beta),
    ) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn parseval_identity_holds() {
    let shape = Shape1D::new(32).expect("shape");
    let plan = FftPlan1D::new(shape);
    let signal = Array1::from_iter((0..32).map(|i| {
        let x = i as f64;
        (0.17 * x).sin() + 0.5 * (0.41 * x).cos()
    }));

    let spectrum = plan.forward_real_to_complex(&signal);
    let time_energy = energy_time_domain(&signal);
    let spectral_energy = energy_frequency_domain(&spectrum) / shape.n as f64;

    assert_relative_eq!(time_energy, spectral_energy, epsilon = 1.0e-10);
}

#[test]
fn caller_owned_paths_match_allocating_paths() {
    let shape = Shape1D::new(24).expect("shape");
    let plan = FftPlan1D::new(shape);
    let signal = Array1::from_iter((0..24).map(|i| (i as f64 * 0.1).sin()));

    let expected_hat = plan.forward_real_to_complex(&signal);
    let mut actual_hat = Array1::<Complex64>::zeros(shape.n);
    plan.forward_real_to_complex_into(&signal, &mut actual_hat);
    for (expected, actual) in expected_hat.iter().zip(actual_hat.iter()) {
        assert_relative_eq!(expected.re, actual.re, epsilon = 1.0e-12);
        assert_relative_eq!(expected.im, actual.im, epsilon = 1.0e-12);
    }

    let mut actual_hat_slice = vec![Complex64::new(0.0, 0.0); shape.n];
    plan.forward_real_to_complex_slice_into(
        signal.as_slice().expect("signal must be contiguous"),
        &mut actual_hat_slice,
    );
    for (expected, actual) in expected_hat.iter().zip(actual_hat_slice.iter()) {
        assert_relative_eq!(expected.re, actual.re, epsilon = 1.0e-12);
        assert_relative_eq!(expected.im, actual.im, epsilon = 1.0e-12);
    }

    let expected_signal = plan.inverse_complex_to_real(&expected_hat);
    let mut actual_signal = Array1::<f64>::zeros(shape.n);
    let mut scratch = Array1::<Complex64>::zeros(shape.n);
    plan.inverse_complex_to_real_into(&expected_hat, &mut actual_signal, &mut scratch);
    for (expected, actual) in expected_signal.iter().zip(actual_signal.iter()) {
        assert_relative_eq!(expected, actual, epsilon = 1.0e-12);
    }
}

#[test]
#[should_panic(expected = "forward output length mismatch")]
fn forward_rejects_mismatched_output_length() {
    let shape = Shape1D::new(8).expect("shape");
    let plan = FftPlan1D::new(shape);
    let signal = Array1::from_elem(shape.n, 1.0);
    let mut output = Array1::<Complex64>::zeros(shape.n - 1);
    plan.forward_real_to_complex_into(&signal, &mut output);
}

#[test]
#[should_panic(expected = "forward input length mismatch")]
fn forward_slice_rejects_mismatched_input_length() {
    let shape = Shape1D::new(8).expect("shape");
    let plan = FftPlan1D::new(shape);
    let signal = [1.0; 7];
    let mut output = vec![Complex64::new(0.0, 0.0); shape.n];
    plan.forward_real_to_complex_slice_into(&signal, &mut output);
}

#[test]
#[should_panic(expected = "inverse scratch length mismatch")]
fn inverse_rejects_mismatched_scratch_length() {
    let shape = Shape1D::new(8).expect("shape");
    let plan = FftPlan1D::new(shape);
    let spectrum = Array1::<Complex64>::zeros(shape.n);
    let mut output = Array1::<f64>::zeros(shape.n);
    let mut scratch = Array1::<Complex64>::zeros(shape.n - 1);
    plan.inverse_complex_to_real_into(&spectrum, &mut output, &mut scratch);
}

#[test]
fn typed_precision_profiles_remain_bounded_for_quantized_inputs() {
    let shape = Shape1D::new(32).expect("shape");
    let reference = Array1::from_iter((0..32).map(|i| {
        let x = i as f64;
        (0.09 * x).sin() + 0.25 * (0.71 * x).cos()
    }));

    let high_plan = FftPlan1D::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64);
    let low_plan = FftPlan1D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
    let mixed_plan = FftPlan1D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);

    let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
    let quantized_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
    let low_input = mixed_input.mapv(|value| value.to_f32());

    let low_recovered: Array1<f32> = low_plan.inverse_typed(&low_plan.forward_typed(&low_input));
    let low_roundtrip = low_recovered.mapv(f64::from);
    let mixed_recovered: Array1<f16> =
        mixed_plan.inverse_typed(&mixed_plan.forward_typed(&mixed_input));
    let mixed_roundtrip: Array1<f64> = mixed_recovered.mapv(|value| f64::from(value.to_f32()));
    let high_roundtrip = high_plan.inverse(&high_plan.forward(&reference));

    let low_error: f64 = low_roundtrip
        .iter()
        .zip(quantized_reference.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum();
    let mixed_error: f64 = mixed_roundtrip
        .iter()
        .zip(quantized_reference.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum();
    let high_error: f64 = high_roundtrip
        .iter()
        .zip(reference.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum();

    assert!(high_error <= 1.0e-12);
    assert!(low_error <= 1.0e-4);
    // Native f16 working buffer accumulates per-stage f16 quantization error,
    // so mixed_error > low_error (precision hierarchy: f64 < f32 < f16).
    // Proof: each of 2×log₂N stages writes back through f16, introducing
    // ε_u_f16 ≈ 4.88×10⁻⁴ per element. For N=32, log₂32=5:
    // analytical budget = N × log₂N × ε_u_f16 × max|x| ≈ 32×5×4.88×10⁻⁴×1.25 ≈ 0.098.
    assert!(
        mixed_error >= low_error,
        "expected f16 error ≥ f32 error (precision hierarchy); \
             mixed={mixed_error:.4e}, low={low_error:.4e}"
    );
    assert!(
        mixed_error <= 5e-1,
        "f16 round-trip error {mixed_error:.4e} exceeds f16 precision budget 5e-1"
    );
}

#[test]
fn mixed_precision_non_power_of_two_roundtrip_stays_bounded() {
    let shape = Shape1D::new(30).expect("shape");
    let mixed_plan = FftPlan1D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);

    let signal = Array1::from_iter((0..30).map(|i| {
        let x = i as f32;
        f16::from_f32((0.21 * x).sin() + 0.2 * (0.47 * x).cos())
    }));

    let recovered: Array1<f16> = mixed_plan.inverse_typed(&mixed_plan.forward_typed(&signal));
    let max_err = signal
        .iter()
        .zip(recovered.iter())
        .map(|(expected, actual)| (expected.to_f32() - actual.to_f32()).abs())
        .fold(0.0f32, f32::max);

    // Non-PoT mixed path executes via f32 auto-kernel and quantizes at boundaries.
    assert!(
        max_err <= 2e-2,
        "non-PoT mixed roundtrip max error {max_err:.4e}"
    );
}

#[test]
fn mixed_precision_non_power_of_two_forward_tracks_low_precision_outputs() {
    let shape = Shape1D::new(30).expect("shape");
    let low_plan = FftPlan1D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
    let mixed_plan = FftPlan1D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);

    let mixed_input = Array1::from_iter((0..30).map(|i| {
        let x = i as f32;
        f16::from_f32((0.13 * x).sin() - 0.15 * (0.61 * x).cos())
    }));
    let low_input = mixed_input.mapv(|value| value.to_f32());

    let low_spec: Array1<Complex32> = low_plan.forward_typed(&low_input);
    let mixed_spec: Array1<Complex32> = mixed_plan.forward_typed(&mixed_input);

    let max_diff = low_spec
        .iter()
        .zip(mixed_spec.iter())
        .map(|(low, mixed)| (*low - *mixed).norm())
        .fold(0.0f32, f32::max);

    // Mixed path should remain close to low-precision spectrum on identical quantized input.
    assert!(
        max_diff <= 2e-2,
        "non-PoT forward spectrum max diff {max_diff:.4e}"
    );
}

proptest! {
    #[test]
    fn roundtrip_holds_for_random_real_signals(
        signal in (1usize..33).prop_flat_map(|len| {
            prop::collection::vec(-10.0f64..10.0f64, len)
        })
    ) {
        let shape = Shape1D::new(signal.len()).expect("shape");
        let plan = FftPlan1D::new(shape);
        let input = Array1::from_vec(signal);
        let recovered = plan.inverse_complex_to_real(&plan.forward_real_to_complex(&input));
        for (expected, actual) in input.iter().zip(recovered.iter()) {
            prop_assert!((expected - actual).abs() < 1.0e-9);
        }
    }
}
