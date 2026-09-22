"""Tests for the best effort release of the CuPy memory pools."""

from unittest.mock import patch

import pytest

from fullwave.solver.solver import Solver

cp = pytest.importorskip("cupy")

INSUFFICIENT_DRIVER = 35


def test_release_is_quiet_when_the_driver_is_too_old():
    """A machine without a usable CUDA driver must not fail the release."""
    error = cp.cuda.runtime.CUDARuntimeError(INSUFFICIENT_DRIVER)
    with patch.object(cp, "get_default_memory_pool", side_effect=error):
        Solver._release_gpu_memory_pools()


def test_release_is_quiet_when_the_pinned_pool_fails():
    """The pinned pool raises on the same machines, and it must not fail either."""
    error = cp.cuda.runtime.CUDARuntimeError(INSUFFICIENT_DRIVER)
    with patch.object(cp, "get_default_pinned_memory_pool", side_effect=error):
        Solver._release_gpu_memory_pools()


def test_release_drains_both_pools_when_the_device_works():
    """On a working device both pools are drained."""
    with (
        patch.object(cp, "get_default_memory_pool") as device_pool,
        patch.object(cp, "get_default_pinned_memory_pool") as pinned_pool,
    ):
        Solver._release_gpu_memory_pools()

    device_pool.return_value.free_all_blocks.assert_called_once_with()
    pinned_pool.return_value.free_all_blocks.assert_called_once_with()
