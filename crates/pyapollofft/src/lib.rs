#![warn(missing_docs)]
//! Python bindings for Apollo FFT and NUFFT.

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollofft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, Complex64, FftPlan1D, FftPlan2D, FftPlan3D, UniformDomain1D,
    UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

fn require_contiguous_1d<T: Element>(input: &PyReadonlyArray1<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be C-contiguous")))
    }
}

fn require_contiguous_2d<T: Element>(input: &PyReadonlyArray2<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be C-contiguous")))
    }
}

fn require_contiguous_3d<T: Element>(input: &PyReadonlyArray3<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be C-contiguous")))
    }
}

fn wgpu_backend_usable() -> bool {
    apollofft_wgpu::WgpuBackend::try_default().is_ok()
}

/// Python wrapper for a reusable 1D FFT plan.
#[pyclass(name = "FftPlan1D")]
struct PyFftPlan1D {
    inner: FftPlan1D,
}

#[pymethods]
impl PyFftPlan1D {
    #[new]
    fn new(n: usize) -> Self {
        Self {
            inner: FftPlan1D::new(n),
        }
    }

    fn fft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        require_contiguous_1d(&input, "fft input")?;
        Ok(PyArray1::from_owned_array(
            py,
            self.inner.forward(&input.as_array().to_owned()),
        ))
    }

    fn ifft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        require_contiguous_1d(&input, "ifft input")?;
        Ok(PyArray1::from_owned_array(
            py,
            self.inner.inverse(&input.as_array().to_owned()),
        ))
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
    fn new(nx: usize, ny: usize) -> Self {
        Self {
            inner: FftPlan2D::new(nx, ny),
        }
    }

    fn fft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        require_contiguous_2d(&input, "fft input")?;
        Ok(PyArray2::from_owned_array(
            py,
            self.inner.forward(&input.as_array().to_owned()),
        ))
    }

    fn ifft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        require_contiguous_2d(&input, "ifft input")?;
        Ok(PyArray2::from_owned_array(
            py,
            self.inner.inverse(&input.as_array().to_owned()),
        ))
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
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            inner: FftPlan3D::new(nx, ny, nz),
        }
    }

    fn fft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        require_contiguous_3d(&input, "fft input")?;
        Ok(PyArray3::from_owned_array(
            py,
            self.inner.forward(&input.as_array().to_owned()),
        ))
    }

    fn ifft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        require_contiguous_3d(&input, "ifft input")?;
        Ok(PyArray3::from_owned_array(
            py,
            self.inner.inverse(&input.as_array().to_owned()),
        ))
    }

    fn rfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        require_contiguous_3d(&input, "rfft input")?;
        let input = input.as_array().to_owned();
        let mut output =
            ndarray::Array3::<Complex64>::zeros((input.dim().0, input.dim().1, self.inner.nz_c()));
        self.inner.forward_r2c_into(&input, &mut output);
        Ok(PyArray3::from_owned_array(py, output))
    }

    fn irfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        require_contiguous_3d(&input, "irfft input")?;
        let mut spectrum = input.as_array().to_owned();
        if spectrum.dim().2 != self.inner.nz_c() {
            return Err(PyValueError::new_err("irfft input shape does not match plan nz_c"));
        }
        let (nx, ny, nz) = self.inner.dimensions();
        let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
        self.inner.inverse_c2r_inplace(&mut spectrum, &mut output);
        Ok(PyArray3::from_owned_array(py, output))
    }
}

/// Forward 1D FFT of a real signal.
#[pyfunction]
fn fft1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&input, "fft1 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray1::from_owned_array(py, apollofft::fft_1d_array(&owned)))
}

/// Inverse 1D FFT of a complex spectrum.
#[pyfunction]
fn ifft1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<Complex64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "ifft1 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray1::from_owned_array(py, apollofft::ifft_1d_array(&owned)))
}

/// Forward 2D FFT of a real array.
#[pyfunction]
fn fft2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    require_contiguous_2d(&input, "fft2 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray2::from_owned_array(py, apollofft::fft_2d_array(&owned)))
}

/// Inverse 2D FFT of a complex spectrum.
#[pyfunction]
fn ifft2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<Complex64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "ifft2 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray2::from_owned_array(py, apollofft::ifft_2d_array(&owned)))
}

/// Forward 3D FFT of a real array.
#[pyfunction]
fn fft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "fft3 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray3::from_owned_array(py, apollofft::fft_3d_array(&owned)))
}

/// Inverse 3D FFT of a complex spectrum.
#[pyfunction]
fn ifft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "ifft3 input")?;
    let owned = input.as_array().to_owned();
    Ok(PyArray3::from_owned_array(py, apollofft::ifft_3d_array(&owned)))
}

/// Forward 3D real-to-complex half-spectrum FFT.
#[pyfunction]
fn rfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "rfft3 input")?;
    let owned = input.as_array().to_owned();
    let plan = FftPlan3D::new(owned.dim().0, owned.dim().1, owned.dim().2);
    let mut output =
        ndarray::Array3::<Complex64>::zeros((owned.dim().0, owned.dim().1, plan.nz_c()));
    plan.forward_r2c_into(&owned, &mut output);
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
    let mut owned = input.as_array().to_owned();
    let (nx, ny, nz_c) = owned.dim();
    if nz_c != nz / 2 + 1 {
        return Err(PyValueError::new_err("irfft3 input shape and nz are inconsistent"));
    }
    let plan = FftPlan3D::new(nx, ny, nz);
    let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
    plan.inverse_c2r_inplace(&mut owned, &mut output);
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
            positions.as_slice().expect("owned positions are contiguous"),
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
        positions.as_slice().expect("owned positions are contiguous"),
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
            owned_values.as_slice().expect("owned values are contiguous"),
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
            positions.as_slice().expect("owned positions are contiguous"),
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
        positions.as_slice().expect("owned positions are contiguous"),
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
            owned_values.as_slice().expect("owned values are contiguous"),
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
fn backend_capabilities() -> BTreeMap<String, BTreeMap<String, bool>> {
    let mut backends = BTreeMap::new();

    let mut cpu = BTreeMap::new();
    cpu.insert("available".to_string(), true);
    cpu.insert("supports_1d".to_string(), true);
    cpu.insert("supports_2d".to_string(), true);
    cpu.insert("supports_3d".to_string(), true);
    cpu.insert("supports_real_to_complex".to_string(), true);
    backends.insert("cpu".to_string(), cpu);

    let wgpu_usable = wgpu_backend_usable();
    let mut wgpu = BTreeMap::new();
    wgpu.insert("available".to_string(), wgpu_usable);
    wgpu.insert("supports_1d".to_string(), false);
    wgpu.insert("supports_2d".to_string(), false);
    wgpu.insert("supports_3d".to_string(), wgpu_usable);
    wgpu.insert("supports_real_to_complex".to_string(), false);
    backends.insert("wgpu".to_string(), wgpu);

    let mut cudatile = BTreeMap::new();
    cudatile.insert("available".to_string(), false);
    cudatile.insert("supports_1d".to_string(), false);
    cudatile.insert("supports_2d".to_string(), false);
    cudatile.insert("supports_3d".to_string(), false);
    cudatile.insert("supports_real_to_complex".to_string(), false);
    backends.insert("cudatile".to_string(), cudatile);

    backends
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
