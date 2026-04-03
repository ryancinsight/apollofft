import json
import sys
import time

import numpy as np

try:
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    import pyfftw
except ImportError:
    pyfftw_fft = None
    pyfftw = None


def complex_pairs(array):
    flat = np.asarray(array, dtype=np.complex128).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in flat]


def stability_delta(factory, repeats):
    baseline = factory()
    max_delta = 0.0
    for _ in range(max(repeats - 1, 0)):
        candidate = factory()
        delta = np.max(np.abs(candidate - baseline))
        max_delta = max(max_delta, float(delta))
    return max_delta


def benchmark(factory, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        factory()
    return (time.perf_counter() - start) * 1000.0 / max(iterations, 1)


def main():
    payload = json.load(sys.stdin)
    mode = payload["mode"]

    if mode == "environment":
        json.dump(
            {
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pyfftw_version": getattr(pyfftw, "__version__", None),
            },
            sys.stdout,
        )
        return

    shape = tuple(int(value) for value in payload["shape"])
    real_input = np.array(payload["real_input"], dtype=np.float64).reshape(shape)

    if mode == "compare_fft":
        repeats = int(payload.get("repeats") or 3)
        numpy_factory = lambda: np.fft.fftn(real_input)
        result = {
            "numpy_available": True,
            "pyfftw_available": pyfftw_fft is not None,
            "numpy_pairs": complex_pairs(numpy_factory()),
            "pyfftw_pairs": None,
            "numpy_stability_max_abs_delta": stability_delta(numpy_factory, repeats),
            "pyfftw_stability_max_abs_delta": None,
            "numpy_version": np.__version__,
            "pyfftw_version": getattr(pyfftw, "__version__", None),
        }
        if pyfftw_fft is not None:
            pyfftw_factory = lambda: pyfftw_fft.fftn(real_input)
            result["pyfftw_pairs"] = complex_pairs(pyfftw_factory())
            result["pyfftw_stability_max_abs_delta"] = stability_delta(pyfftw_factory, repeats)
        json.dump(result, sys.stdout)
        return

    if mode == "benchmark_fft":
        iterations = int(payload.get("iterations") or 16)
        numpy_factory = lambda: np.fft.fftn(real_input)
        result = {
            "numpy_available": True,
            "pyfftw_available": pyfftw_fft is not None,
            "numpy_ms": benchmark(numpy_factory, iterations),
            "pyfftw_ms": None,
        }
        if pyfftw_fft is not None:
            pyfftw_factory = lambda: pyfftw_fft.fftn(real_input)
            result["pyfftw_ms"] = benchmark(pyfftw_factory, iterations)
        json.dump(result, sys.stdout)
        return

    raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    main()
