import numpy as np
import pytest

import pyapollofft as afft


def test_fft1_roundtrip():
    signal = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    spectrum = afft.fft1(signal)
    reconstructed = afft.ifft1(spectrum)
    assert np.allclose(signal, reconstructed, atol=1e-12)


def test_fft3_roundtrip():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = afft.fft3(field)
    reconstructed = afft.ifft3(spectrum)
    assert np.allclose(field, reconstructed, atol=1e-12)


def test_rfft3_roundtrip():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = afft.rfft3(field)
    reconstructed = afft.irfft3(spectrum, field.shape[-1])
    assert np.allclose(field, reconstructed, atol=1e-12)


def test_nufft_1d_fast_tracks_exact():
    positions = np.array([0.01, 0.09, 0.23, 0.51], dtype=np.float64)
    values = np.array([1.0 + 0.0j, 0.5 + 0.2j, -0.2 + 0.3j, 0.1 - 0.4j], dtype=np.complex128)
    exact = afft.nufft_type1_1d(positions, values, 0.05, n_out=16)
    fast = afft.nufft_type1_1d_fast(positions, values, 0.05, n_out=16)
    assert np.allclose(exact, fast, atol=1e-5)


def test_nufft_3d_fast_tracks_exact():
    positions = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.12, 0.11, 0.05],
            [0.21, 0.18, 0.09],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0 + 0.0j, 0.5 + 0.1j, -0.4 + 0.2j], dtype=np.complex128)
    exact = afft.nufft_type1_3d(positions, values, 4, 4, 4, 0.1, 0.1, 0.1)
    fast = afft.nufft_type1_3d_fast(positions, values, 4, 4, 4, 0.1, 0.1, 0.1)
    assert np.allclose(exact, fast, atol=1e-5)


def test_backend_capabilities_are_reported():
    capabilities = afft.backend_capabilities()
    assert capabilities["cpu"]["available"] is True
    assert capabilities["cpu"]["supports_3d"] is True
    assert "wgpu" in capabilities


def test_non_contiguous_input_is_rejected():
    signal = np.linspace(0.0, 1.0, 16, dtype=np.float64)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        afft.fft1(signal)


def test_dtype_validation_rejects_wrong_real_dtype():
    signal = np.arange(8, dtype=np.float32)
    with pytest.raises(TypeError):
        afft.fft1(signal)
