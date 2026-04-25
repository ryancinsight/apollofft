//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_sht::infrastructure::kernel::spherical_harmonic::{
    gauss_legendre_nodes_weights, spherical_harmonic,
};
use apollo_sht::{SphericalGridSpec, SphericalHarmonicCoefficients};
use ndarray::Array2;
use num_complex::{Complex32, Complex64};

use crate::application::plan::ShtWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::ShtGpuKernel;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    ShtWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct ShtWgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernel: Arc<ShtGpuKernel>,
}

impl ShtWgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            kernel: Arc::new(ShtGpuKernel::new(device.as_ref())),
            device,
            queue,
        }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| WgpuError::AdapterUnavailable {
                message: error.to_string(),
            })?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("apollo-sht-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                WgpuError::DeviceUnavailable {
                    message: error.to_string(),
                }
            })?;
        Ok(Self::new(Arc::new(device), Arc::new(queue)))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_complex(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Create a WGPU plan descriptor.
    #[must_use]
    pub const fn plan(
        &self,
        latitudes: usize,
        longitudes: usize,
        max_degree: usize,
    ) -> ShtWgpuPlan {
        ShtWgpuPlan::new(latitudes, longitudes, max_degree)
    }

    /// Execute forward complex SHT by direct quadrature matrix summation.
    pub fn execute_forward(
        &self,
        plan: &ShtWgpuPlan,
        samples: &Array2<Complex32>,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_plan(plan)?;
        if samples.dim() != (plan.latitudes(), plan.longitudes()) {
            let (actual_latitudes, actual_longitudes) = samples.dim();
            return Err(WgpuError::SampleShapeMismatch {
                expected_latitudes: plan.latitudes(),
                expected_longitudes: plan.longitudes(),
                actual_latitudes,
                actual_longitudes,
            });
        }
        let basis = forward_basis(plan);
        let input: Vec<Complex32> = samples.iter().copied().collect();
        let raw = self.kernel.execute_forward(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan.mode_count(),
            plan.sample_count(),
            &input,
            &basis,
        )?;
        Ok(coefficients_from_modes(plan.max_degree(), &raw))
    }

    /// Execute inverse complex SHT by direct synthesis matrix summation.
    pub fn execute_inverse(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<Array2<Complex64>> {
        validate_plan(plan)?;
        if coefficients.max_degree() != plan.max_degree() {
            return Err(WgpuError::CoefficientShapeMismatch {
                expected: plan.max_degree(),
                actual: coefficients.max_degree(),
            });
        }
        let basis = inverse_basis(plan);
        let input = modes_from_coefficients(coefficients);
        let raw = self.kernel.execute_inverse(
            self.device.as_ref(),
            self.queue.as_ref(),
            plan.sample_count(),
            plan.mode_count(),
            &input,
            &basis,
        )?;
        Array2::from_shape_vec(
            (plan.latitudes(), plan.longitudes()),
            raw.into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        )
        .map_err(|_| WgpuError::InvalidPlan {
            latitudes: plan.latitudes(),
            longitudes: plan.longitudes(),
            max_degree: plan.max_degree(),
            message: "inverse output shape does not match plan",
        })
    }
}

fn validate_plan(plan: &ShtWgpuPlan) -> WgpuResult<()> {
    if SphericalGridSpec::new(plan.latitudes(), plan.longitudes(), plan.max_degree()).is_err() {
        return Err(WgpuError::InvalidPlan {
            latitudes: plan.latitudes(),
            longitudes: plan.longitudes(),
            max_degree: plan.max_degree(),
            message: "grid must be non-empty and satisfy max_degree < latitudes and 2*max_degree+1 <= longitudes",
        });
    }
    if plan.sample_count() > u32::MAX as usize || plan.mode_count() > u32::MAX as usize {
        return Err(WgpuError::InvalidPlan {
            latitudes: plan.latitudes(),
            longitudes: plan.longitudes(),
            max_degree: plan.max_degree(),
            message: "WGPU dispatch dimensions must fit in u32",
        });
    }
    Ok(())
}

fn mode_pairs(max_degree: usize) -> impl Iterator<Item = (usize, isize)> {
    (0..=max_degree).flat_map(|degree| {
        (-(degree as isize)..=(degree as isize)).map(move |order| (degree, order))
    })
}

fn grid_angles(plan: &ShtWgpuPlan) -> Vec<(f64, f64, f64)> {
    let (cos_theta_nodes, theta_weights) = gauss_legendre_nodes_weights(plan.latitudes());
    let longitude_weight = std::f64::consts::TAU / plan.longitudes() as f64;
    (0..plan.latitudes())
        .flat_map(|lat| {
            let theta = cos_theta_nodes[lat].acos();
            let weight = theta_weights[lat] * longitude_weight;
            (0..plan.longitudes()).map(move |lon| {
                (
                    theta,
                    std::f64::consts::TAU * lon as f64 / plan.longitudes() as f64,
                    weight,
                )
            })
        })
        .collect()
}

fn forward_basis(plan: &ShtWgpuPlan) -> Vec<Complex32> {
    let grid = grid_angles(plan);
    mode_pairs(plan.max_degree())
        .flat_map(|(degree, order)| {
            grid.iter().map(move |(theta, phi, weight)| {
                let value = spherical_harmonic(degree, order, *theta, *phi).conj() * *weight;
                Complex32::new(value.re as f32, value.im as f32)
            })
        })
        .collect()
}

fn inverse_basis(plan: &ShtWgpuPlan) -> Vec<Complex32> {
    let grid = grid_angles(plan);
    (0..plan.sample_count())
        .flat_map(|sample| {
            let (theta, phi, _) = grid[sample];
            mode_pairs(plan.max_degree()).map(move |(degree, order)| {
                let value = spherical_harmonic(degree, order, theta, phi);
                Complex32::new(value.re as f32, value.im as f32)
            })
        })
        .collect()
}

fn coefficients_from_modes(max_degree: usize, raw: &[Complex32]) -> SphericalHarmonicCoefficients {
    let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
    for ((degree, order), value) in mode_pairs(max_degree).zip(raw.iter().copied()) {
        coefficients.set(
            degree,
            order,
            Complex64::new(value.re as f64, value.im as f64),
        );
    }
    coefficients
}

fn modes_from_coefficients(coefficients: &SphericalHarmonicCoefficients) -> Vec<Complex32> {
    mode_pairs(coefficients.max_degree())
        .map(|(degree, order)| {
            let value = coefficients.get(degree, order);
            Complex32::new(value.re as f32, value.im as f32)
        })
        .collect()
}
