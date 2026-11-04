"""Read the free GPU memory and release the CuPy memory pools."""

from __future__ import annotations

import ctypes
import gc
import logging
from typing import TYPE_CHECKING

from fullwave.solver import cuda_utils

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("__main__." + __name__)

GPU_MEMORY_FRACTION = 0.9
GPU_MEMORY_RESERVE_BYTES = 512 * 1024**2


def members_that_fit(*, free_bytes: int, shared_bytes: int, per_member_bytes: int) -> int:
    """Return how many batch members fit into the free memory of a GPU.

    Only ``GPU_MEMORY_FRACTION`` of the free memory is used, less ``GPU_MEMORY_RESERVE_BYTES``
    for the CUDA context and for what the estimate does not count.

    Parameters
    ----------
    free_bytes : int
        The free memory of the GPU.
    shared_bytes : int
        The memory used once, whatever the number of members.
    per_member_bytes : int
        The memory each member adds.

    Returns
    -------
    int
        The count, which is zero or negative when not even one member fits.

    """
    usable_bytes = GPU_MEMORY_FRACTION * free_bytes - GPU_MEMORY_RESERVE_BYTES - shared_bytes
    return int(usable_bytes // per_member_bytes)


def free_gpu_memory_bytes(device_ids: Sequence[str]) -> dict[str, int] | None:
    """Return the free memory of each GPU in bytes, as the CUDA driver reports it.

    Parameters
    ----------
    device_ids : Sequence[str]
        CUDA device ordinals as this process sees them, the same ids as ``cuda_device_id``.

    Returns
    -------
    dict[str, int] | None
        The free bytes of each GPU, or None when the CUDA driver cannot report every GPU.

    """
    wanted = [device.strip() for device in device_ids]
    try:
        return {
            device: cuda_utils.free_memory_bytes_of_cuda_ordinal(ordinal=int(device))
            for device in wanted
        }
    except (
        AttributeError,
        NameError,
        OSError,
        RuntimeError,
        ValueError,
        ctypes.ArgumentError,
    ) as error:
        logger.debug("Free GPU memory unknown: %s", error)
        return None


def release_gpu_memory_pools() -> None:
    """Release all CuPy GPU memory pool blocks back to CUDA.

    Call ``gc.collect()`` first so that Python releases references to
    CuPy arrays, then drain both the device and pinned memory pools.
    This prevents stale allocations from causing memory pressure when
    subsequent operations allocate large GPU arrays.

    The release is best effort. It does nothing when CuPy is absent, and
    it does nothing when CuPy is present without a usable CUDA driver,
    which is the case on a machine with no GPU.
    """
    gc.collect()
    try:
        import cupy as cp  # noqa: PLC0415
    except ImportError:
        return

    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except cp.cuda.runtime.CUDARuntimeError as error:
        logger.debug("GPU memory pools were not released: %s", error)
