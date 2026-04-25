#![warn(missing_docs)]
//! Python bindings for Apollo FFT and NUFFT.

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollo_fft::{
    f16, Complex32, Complex64, CpuBackend, FftBackend, FftPlan1D, FftPlan2D, FftPlan3D,
    PrecisionMode, PrecisionProfile, Shape1D, Shape2D, Shape3D, StoragePrecision,
};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn require_contiguous_1d<T: Element>(input: &PyReadonlyArray1<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn require_contiguous_2d<T: Element>(input: &PyReadonlyArray2<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn require_contiguous_3d<T: Element>(input: &PyReadonlyArray3<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn wgpu_backend_usable() -> bool {
    apollo_fft_wgpu::WgpuBackend::try_default().is_ok()
}

fn parse_precision(precision: Option<&str>) -> PyResult<PrecisionProfile> {
    match precision.unwrap_or("high_accuracy") {
        "high_accuracy" => Ok(PrecisionProfile::HIGH_ACCURACY_F64),
        "low_precision" => Ok(PrecisionProfile::LOW_PRECISION_F32),
        "mixed_precision" => Ok(PrecisionProfile::MIXED_PRECISION_F16_F32),
        other => Err(PyValueError::new_err(format!(
            "unsupported precision `{other}`; expected `high_accuracy`, `low_precision`, or `mixed_precision`"
        ))),
    }
}

fn precision_name(profile: PrecisionProfile) -> &'static str {
    match profile.mode {
        PrecisionMode::HighAccuracy => "high_accuracy",
        PrecisionMode::LowPrecision => "low_precision",
        PrecisionMode::MixedPrecision => "mixed_precision",
    }
}

fn require_profile_matches_f64(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F64 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float64/complex128 input but precision `{}` expects float32/complex64 storage",
            precision_name(profile)
        )))
    }
}

fn require_profile_matches_f32(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F32 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float32/complex64 input but precision `{}` expects float64/complex128 storage",
            precision_name(profile)
        )))
    }
}

/// Python wrapper for a reusable 1D FFT plan.
#[pyclass(name = "FftPlan1D")]
struct PyFftPlan1D {
    inner: FftPlan1D,
}

#[pymethods]
impl PyFftPlan1D {
    #[new]
    #[pyo3(signature = (n, precision=None))]
    fn new(n: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape = Shape1D::new(n).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: FftPlan1D::with_precision(shape, profile),
        })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
            require_contiguous_1d(&input64, "fft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "fft")?;
            Ok(
                PyArray1::from_owned_array(py, self.inner.forward(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        self.inner
                            .forward_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    require_profile_matches_f32(self.inner.precision_profile(), "fft")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        self.inner.forward_typed(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
            require_contiguous_1d(&input64, "ifft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "ifft")?;
            Ok(
                PyArray1::from_owned_array(py, self.inner.inverse(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
            require_contiguous_1d(&input32, "ifft input")?;
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => Ok(PyArray1::from_owned_array(
                    py,
                    self.inner
                        .inverse_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.inner.precision_profile(), "ifft")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        self.inner
                            .inverse_typed::<f32>(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }
}

/// Python wrapper for a reusable 2D FFT plan.
#[pyclass(name = "FftPlan2D")]
struct PyFftPlan2D {
    inner: FftPlan2D,
}

#[pymethods]
impl PyFftPlan2D {
    #[new]
    #[pyo3(signature = (nx, ny, precision=None))]
    fn new(nx: usize, ny: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape =
            Shape2D::new(nx, ny).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: FftPlan2D::with_precision(shape, profile),
        })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
            require_contiguous_2d(&input64, "fft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "fft")?;
            Ok(
                PyArray2::from_owned_array(py, self.inner.forward(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        self.inner
                            .forward_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    require_profile_matches_f32(self.inner.precision_profile(), "fft")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        self.inner.forward_typed(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
            require_contiguous_2d(&input64, "ifft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "ifft")?;
            Ok(
                PyArray2::from_owned_array(py, self.inner.inverse(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
            require_contiguous_2d(&input32, "ifft input")?;
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => Ok(PyArray2::from_owned_array(
                    py,
                    self.inner
                        .inverse_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.inner.precision_profile(), "ifft")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        self.inner
                            .inverse_typed::<f32>(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }
}

/// Python wrapper for a reusable 3D FFT plan.
#[pyclass(name = "FftPlan3D")]
struct PyFftPlan3D {
    inner: FftPlan3D,
}

#[pymethods]
impl PyFftPlan3D {
    #[new]
    #[pyo3(signature = (nx, ny, nz, precision=None))]
    fn new(nx: usize, ny: usize, nz: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape =
            Shape3D::new(nx, ny, nz).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: FftPlan3D::with_precision(shape, profile),
        })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
            require_contiguous_3d(&input64, "fft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "fft")?;
            Ok(
                PyArray3::from_owned_array(py, self.inner.forward(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        self.inner
                            .forward_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    require_profile_matches_f32(self.inner.precision_profile(), "fft")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        self.inner.forward_typed(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
            require_contiguous_3d(&input64, "ifft input")?;
            require_profile_matches_f64(self.inner.precision_profile(), "ifft")?;
            Ok(
                PyArray3::from_owned_array(py, self.inner.inverse(&input64.as_array().to_owned()))
                    .into_any()
                    .unbind(),
            )
        } else {
            let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
            require_contiguous_3d(&input32, "ifft input")?;
            match self.inner.precision_profile().storage {
                StoragePrecision::F16 => Ok(PyArray3::from_owned_array(
                    py,
                    self.inner
                        .inverse_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.inner.precision_profile(), "ifft")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        self.inner
                            .inverse_typed::<f32>(&input32.as_array().to_owned()),
                    )
                    .into_any()
                    .unbind())
                }
            }
        }
    }

    fn rfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        require_contiguous_3d(&input, "rfft input")?;
        let input = input.as_array().to_owned();
        let mut output = ndarray::Array3::<Complex64>::zeros(input.dim());
        self.inner.forward_into(&input, &mut output);
        Ok(PyArray3::from_owned_array(py, output))
    }

    fn irfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        require_contiguous_3d(&input, "irfft input")?;
        let spectrum = input.as_array().to_owned();
        if spectrum.dim() != self.inner.dimensions() {
            return Err(PyValueError::new_err(
                "irfft input shape does not match plan dimensions",
            ));
        }
        let (nx, ny, nz) = self.inner.dimensions();
        let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
        let mut scratch = ndarray::Array3::<Complex64>::zeros((nx, ny, nz));
        self.inner
            .inverse_complex_to_real_into(&spectrum, &mut output, &mut scratch);
        Ok(PyArray3::from_owned_array(py, output))
    }
}

/// Forward 1D FFT of a real signal.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
        require_contiguous_1d(&input64, "fft1 input")?;
        require_profile_matches_f64(profile, "fft1")?;
        Ok(
            PyArray1::from_owned_array(
                py,
                apollo_fft::fft_1d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::fft_1d_array_typed(
                        &input32.as_array().mapv(f16::from_f32),
                        profile,
                    ),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                require_profile_matches_f32(profile, "fft1")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::fft_1d_array_typed(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 1D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
        require_contiguous_1d(&input64, "ifft1 input")?;
        require_profile_matches_f64(profile, "ifft1")?;
        Ok(PyArray1::from_owned_array(
            py,
            apollo_fft::ifft_1d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
        require_contiguous_1d(&input32, "ifft1 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray1::from_owned_array(
                py,
                apollo_fft::ifft_1d_array_typed::<f16>(&input32.as_array().to_owned(), profile)
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft1")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::ifft_1d_array_typed::<f32>(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 2D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
        require_contiguous_2d(&input64, "fft2 input")?;
        require_profile_matches_f64(profile, "fft2")?;
        Ok(
            PyArray2::from_owned_array(
                py,
                apollo_fft::fft_2d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::fft_2d_array_typed(
                        &input32.as_array().mapv(f16::from_f32),
                        profile,
                    ),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                require_profile_matches_f32(profile, "fft2")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::fft_2d_array_typed(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 2D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
        require_contiguous_2d(&input64, "ifft2 input")?;
        require_profile_matches_f64(profile, "ifft2")?;
        Ok(PyArray2::from_owned_array(
            py,
            apollo_fft::ifft_2d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
        require_contiguous_2d(&input32, "ifft2 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray2::from_owned_array(
                py,
                apollo_fft::ifft_2d_array_typed::<f16>(&input32.as_array().to_owned(), profile)
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft2")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::ifft_2d_array_typed::<f32>(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 3D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
        require_contiguous_3d(&input64, "fft3 input")?;
        require_profile_matches_f64(profile, "fft3")?;
        Ok(
            PyArray3::from_owned_array(
                py,
                apollo_fft::fft_3d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::fft_3d_array_typed(
                        &input32.as_array().mapv(f16::from_f32),
                        profile,
                    ),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                require_profile_matches_f32(profile, "fft3")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::fft_3d_array_typed(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 3D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
        require_contiguous_3d(&input64, "ifft3 input")?;
        require_profile_matches_f64(profile, "ifft3")?;
        Ok(PyArray3::from_owned_array(
            py,
            apollo_fft::ifft_3d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
        require_contiguous_3d(&input32, "ifft3 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray3::from_owned_array(
                py,
                apollo_fft::ifft_3d_array_typed::<f16>(&input32.as_array().to_owned(), profile)
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft3")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::ifft_3d_array_typed::<f32>(&input32.as_array().to_owned(), profile),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 3D real-to-complex half-spectrum FFT.
#[pyfunction]
fn rfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "rfft3 input")?;
    let owned = input.as_array().to_owned();
    let shape = Shape3D::new(owned.dim().0, owned.dim().1, owned.dim().2)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan = FftPlan3D::new(shape);
    let mut output = ndarray::Array3::<Complex64>::zeros(owned.dim());
    plan.forward_into(&owned, &mut output);
    Ok(PyArray3::from_owned_array(py, output))
}

/// Inverse 3D half-spectrum FFT.
#[pyfunction]
fn irfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
    nz: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "irfft3 input")?;
    let owned = input.as_array().to_owned();
    let (nx, ny, nz_c) = owned.dim();
    if nz_c != nz {
        return Err(PyValueError::new_err(
            "irfft3 input shape and nz are inconsistent",
        ));
    }
    let shape =
        Shape3D::new(nx, ny, nz).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan = FftPlan3D::new(shape);
    let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = ndarray::Array3::<Complex64>::zeros((nx, ny, nz));
    plan.inverse_complex_to_real_into(&owned, &mut output, &mut scratch);
    Ok(PyArray3::from_owned_array(py, output))
}

/// Exact direct 1D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_1d")]
#[pyo3(signature = (positions, values, dx, n_out=None))]
fn nufft_type1_1d_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<Complex64>,
    dx: f64,
    n_out: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&positions, "nufft_type1_1d positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d values")?;
    let positions = positions.as_array().to_owned();
    let values = values.as_array().to_owned();
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray1::from_owned_array(
        py,
        nufft_type1_1d(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
        ),
    ))
}

/// Exact direct 1D type-2 NUFFT.
#[pyfunction(name = "nufft_type2_1d")]
fn nufft_type2_1d_py<'py>(
    py: Python<'py>,
    fourier_coeffs: PyReadonlyArray1<Complex64>,
    positions: PyReadonlyArray1<f64>,
    dx: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d positions")?;
    let coeffs = fourier_coeffs.as_array().to_owned();
    let positions = positions.as_array().to_owned();
    let domain = UniformDomain1D::new(coeffs.len(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(nufft_type2_1d(
        &coeffs,
        positions
            .as_slice()
            .expect("owned positions are contiguous"),
        domain,
    )
    .into_pyarray(py))
}

/// Exact direct 3D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_3d")]
fn nufft_type1_3d_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<Complex64>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_2d(&positions, "nufft_type1_3d positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d values")?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions must have shape (n_samples, 3)",
        ));
    }
    if positions.nrows() != values.as_array().len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let owned_values = values.as_array().to_owned();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray3::from_owned_array(
        py,
        nufft_type1_3d(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
        ),
    ))
}

/// Fast 1D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_1d_fast")]
#[pyo3(signature = (positions, values, dx, n_out=None, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type1_1d_fast_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<Complex64>,
    dx: f64,
    n_out: Option<usize>,
    kernel_width: usize,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&positions, "nufft_type1_1d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d_fast values")?;
    let positions = positions.as_array().to_owned();
    let values = values.as_array().to_owned();
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray1::from_owned_array(
        py,
        nufft_type1_1d_fast(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
            kernel_width,
        ),
    ))
}

/// Fast 1D type-2 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type2_1d_fast")]
#[pyo3(signature = (fourier_coeffs, positions, dx, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type2_1d_fast_py<'py>(
    py: Python<'py>,
    fourier_coeffs: PyReadonlyArray1<Complex64>,
    positions: PyReadonlyArray1<f64>,
    dx: f64,
    kernel_width: usize,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d_fast fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d_fast positions")?;
    let coeffs = fourier_coeffs.as_array().to_owned();
    let positions = positions.as_array().to_owned();
    let domain = UniformDomain1D::new(coeffs.len(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(nufft_type2_1d_fast(
        &coeffs,
        positions
            .as_slice()
            .expect("owned positions are contiguous"),
        domain,
        kernel_width,
    )
    .into_pyarray(py))
}

/// Fast 3D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_3d_fast")]
#[pyo3(signature = (positions, values, nx, ny, nz, dx, dy, dz, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type1_3d_fast_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<Complex64>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    kernel_width: usize,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_2d(&positions, "nufft_type1_3d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d_fast values")?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions must have shape (n_samples, 3)",
        ));
    }
    if positions.nrows() != values.as_array().len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let owned_values = values.as_array().to_owned();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray3::from_owned_array(
        py,
        nufft_type1_3d_fast(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
            kernel_width,
        ),
    ))
}

/// Return the backend names that are genuinely usable from Python on this host.
#[pyfunction]
fn available_backends() -> Vec<String> {
    let mut backends = vec!["cpu".to_string()];
    if wgpu_backend_usable() {
        backends.push("wgpu".to_string());
    }
    backends
}

/// Return backend capability metadata for Python callers.
#[pyfunction]
fn backend_capabilities(py: Python<'_>) -> PyResult<PyObject> {
    let backends = PyDict::new(py);

    let cpu_caps = CpuBackend.capabilities();
    let cpu = PyDict::new(py);
    cpu.set_item("available", true)?;
    cpu.set_item("supports_1d", cpu_caps.supports_1d)?;
    cpu.set_item("supports_2d", cpu_caps.supports_2d)?;
    cpu.set_item("supports_3d", cpu_caps.supports_3d)?;
    cpu.set_item(
        "supports_real_to_complex",
        cpu_caps.supports_real_to_complex,
    )?;
    cpu.set_item(
        "supports_mixed_precision",
        cpu_caps.supports_mixed_precision,
    )?;
    cpu.set_item(
        "default_precision_profile",
        precision_name(cpu_caps.default_precision_profile),
    )?;
    cpu.set_item(
        "supported_precision_profiles",
        cpu_caps
            .supported_precision_profiles
            .iter()
            .map(|profile| precision_name(*profile))
            .collect::<Vec<_>>(),
    )?;
    backends.set_item("cpu", cpu)?;

    let wgpu = PyDict::new(py);
    if let Ok(backend) = apollo_fft_wgpu::WgpuBackend::try_default() {
        let caps = backend.capabilities();
        wgpu.set_item("available", true)?;
        wgpu.set_item("supports_1d", caps.supports_1d)?;
        wgpu.set_item("supports_2d", caps.supports_2d)?;
        wgpu.set_item("supports_3d", caps.supports_3d)?;
        wgpu.set_item("supports_real_to_complex", caps.supports_real_to_complex)?;
        wgpu.set_item("supports_mixed_precision", caps.supports_mixed_precision)?;
        wgpu.set_item(
            "default_precision_profile",
            precision_name(caps.default_precision_profile),
        )?;
        wgpu.set_item(
            "supported_precision_profiles",
            caps.supported_precision_profiles
                .iter()
                .map(|profile| precision_name(*profile))
                .collect::<Vec<_>>(),
        )?;
    } else {
        wgpu.set_item("available", false)?;
        wgpu.set_item("supports_1d", false)?;
        wgpu.set_item("supports_2d", false)?;
        wgpu.set_item("supports_3d", false)?;
        wgpu.set_item("supports_real_to_complex", false)?;
        wgpu.set_item("supports_mixed_precision", false)?;
        wgpu.set_item("default_precision_profile", "low_precision")?;
        wgpu.set_item("supported_precision_profiles", vec!["low_precision"])?;
    }
    backends.set_item("wgpu", wgpu)?;

    let cudatile = PyDict::new(py);
    cudatile.set_item("available", false)?;
    cudatile.set_item("supports_1d", false)?;
    cudatile.set_item("supports_2d", false)?;
    cudatile.set_item("supports_3d", false)?;
    cudatile.set_item("supports_real_to_complex", false)?;
    cudatile.set_item("supports_mixed_precision", false)?;
    cudatile.set_item("default_precision_profile", "high_accuracy")?;
    cudatile.set_item("supported_precision_profiles", Vec::<&str>::new())?;
    backends.set_item("cudatile", cudatile)?;

    Ok(backends.into_any().unbind())
}

/// Python module entry point.
#[pymodule]
fn _pyapollofft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFftPlan1D>()?;
    m.add_class::<PyFftPlan2D>()?;
    m.add_class::<PyFftPlan3D>()?;
    m.add_function(wrap_pyfunction!(fft1, m)?)?;
    m.add_function(wrap_pyfunction!(ifft1, m)?)?;
    m.add_function(wrap_pyfunction!(fft2, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2, m)?)?;
    m.add_function(wrap_pyfunction!(fft3, m)?)?;
    m.add_function(wrap_pyfunction!(ifft3, m)?)?;
    m.add_function(wrap_pyfunction!(rfft3, m)?)?;
    m.add_function(wrap_pyfunction!(irfft3, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_1d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type2_1d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_1d_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type2_1d_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_3d_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(backend_capabilities, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
