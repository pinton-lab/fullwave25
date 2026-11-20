"""Tests for the multi-GPU array upload helper (skipped without CuPy and a CUDA device)."""

import numpy as np
import pytest

from fullwave.medium import _upload_arrays_multi_gpu

try:
    import cupy as cp

    _N_GPUS = cp.cuda.runtime.getDeviceCount()
except Exception:
    _N_GPUS = 0

N_ARRAYS = 5
SHAPE = (64, 48)


def _make_named_arrays(seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(f"array_{i}", rng.standard_normal(SHAPE)) for i in range(N_ARRAYS)]


def _assert_uploaded_exactly(named_arrays: list[tuple[str, np.ndarray]], results: dict) -> None:
    assert set(results) == {name for name, _ in named_arrays}
    for name, arr_np in named_arrays:
        result = results[name]
        assert result.device.id == 0
        assert result.dtype == np.float32
        np.testing.assert_array_equal(cp.asnumpy(result), arr_np.astype(np.float32))


@pytest.mark.skipif(_N_GPUS < 1, reason="CuPy or CUDA device not available")
def test_upload_arrays_multi_gpu_returns_exact_float32_copies_on_device_0():
    named_arrays = _make_named_arrays(seed=0)

    results = _upload_arrays_multi_gpu(named_arrays, np.dtype(np.float32))

    _assert_uploaded_exactly(named_arrays, results)


@pytest.mark.skipif(_N_GPUS < 2, reason="At least 2 GPUs required")
def test_upload_arrays_multi_gpu_stays_exact_over_repeated_cross_device_uploads():
    for seed in range(10):
        named_arrays = _make_named_arrays(seed=seed)

        results = _upload_arrays_multi_gpu(named_arrays, np.dtype(np.float32))

        _assert_uploaded_exactly(named_arrays, results)
