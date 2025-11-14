"""CUDA utilities for device information and API management.

This module provides:
- CUDA device specification retrieval
- CUDA API call wrappers with error handling
- Device architecture and compute capability mappings
- Memory and performance information extraction

based on: https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549?permalink_comment_id=5043495#gistcomment-5043495
"""

import ctypes
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger("__main__." + __name__)

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

# Conversions from semantic version numbers
# Borrowed from original gist
# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
# and updated from the "GPUs supported" section of this Wikipedia article
# https://en.wikipedia.org/wiki/CUDA
SEMVER_TO_CORES = {
    (1, 0): 8,  # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,  # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,  # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,  # Volta
    (7, 2): 64,
    (7, 5): 64,  # Turing
    (8, 0): 64,  # Ampere
    (8, 6): 64,
    (8, 7): 64,
    (8, 9): 128,  # Ada Lovelace
    (9, 0): 128,  # Hopper
    (10, 0): 128,  # Blackwell
    (10, 1): 128,  # Blackwell
    (12, 0): 128,  # Blackwell
}
SEMVER_TO_ARCH = {
    (1, 0): "tesla",
    (1, 1): "tesla",
    (1, 2): "tesla",
    (1, 3): "tesla",
    (2, 0): "fermi",
    (2, 1): "fermi",
    (3, 0): "kepler",
    (3, 2): "kepler",
    (3, 5): "kepler",
    (3, 7): "kepler",
    (5, 0): "maxwell",
    (5, 2): "maxwell",
    (5, 3): "maxwell",
    (6, 0): "pascal",
    (6, 1): "pascal",
    (6, 2): "pascal",
    (7, 0): "volta",
    (7, 2): "volta",
    (7, 5): "turing",
    (8, 0): "ampere",
    (8, 6): "ampere",
    (8, 7): "ampere",
    (8, 9): "ada lovelace",
    (9, 0): "hopper",
    (10, 0): "blackwell",
    (10, 1): "blackwell",
    (12, 0): "blackwell",
}


# Attempt to load the CUDA library
libnames = ("libcuda.so", "libcuda.so.1", "libcuda.dylib", "cuda.dll")
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    message = (
        f"Critical: Could not load any of: {', '.join(libnames)}\n"
        "Please ensure that the CUDA toolkit is installed and accessible."
    )
    logger.critical(message)


def cuda_api_call(func: Callable) -> Callable:
    """Wrap CUDA API calls and check their results.

    Decorator for CUDA API calls
    Raises RuntimeError if the CUDA call does not return CUDA_SUCCESS.

    Returns:
        Callable: The wrapped function that checks CUDA API call results.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            error_msg = error_str.value.decode() if error_str.value else "Unknown error"
            msg = f"{func.__name__} failed with error code {result}: {error_msg}"
            logger.error(error_msg)
            raise RuntimeError(msg)
        return result

    return wrapper


def cuda_api_call_warn(func: Callable) -> Callable:
    """Wrap CUDA API calls and check their results.

    Prints a warning message if the CUDA call does not return CUDA_SUCCESS.

    Returns:
        Callable: The wrapped function that checks CUDA API call results.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            error_msg = error_str.value.decode() if error_str.value else "Unknown error"
            msg = f"Warning: {func.__name__} failed with error code {result}: {error_msg}"
            logger.warning(msg, stacklevel=2)
        return result

    return wrapper


# CUDA API calls wrapped with the decorator
@cuda_api_call
def cuInit(flags):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuInit(flags)


@cuda_api_call
def cuDeviceGetCount(count):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuDeviceGetCount(count)


@cuda_api_call
def cuDeviceGet(device, ordinal):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuDeviceGet(device, ordinal)


@cuda_api_call
def cuDeviceGetName(name, length, dev):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuDeviceGetName(name, length, dev)


@cuda_api_call
def cuDeviceComputeCapability(major, minor, dev):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuDeviceComputeCapability(major, minor, dev)


@cuda_api_call
def cuDeviceGetAttribute(pi, attrib, dev):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuDeviceGetAttribute(pi, attrib, dev)


@cuda_api_call_warn
def cuCtxCreate(pctx, flags, dev):  # noqa: ANN001, ANN201, D103, N802
    try:
        result = cuda.cuCtxCreate_v2(pctx, flags, dev)
    except AttributeError:
        result = cuda.cuCtxCreate(pctx, flags, dev)
    return result


@cuda_api_call_warn
def cuMemGetInfo(free, total):  # noqa: ANN001, ANN201, D103, N802
    try:
        result = cuda.cuMemGetInfo_v2(free, total)
    except AttributeError:
        result = cuda.cuMemGetInfo(free, total)
    return result


@cuda_api_call
def cuCtxDetach(ctx):  # noqa: ANN001, ANN201, D103, N802
    return cuda.cuCtxDetach(ctx)


# Main function to get CUDA device specs
def get_cuda_device_specs() -> list[dict[str, Any]]:
    """Generate spec for each GPU device with format.

    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'cuda_cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float,
        'architecture': str,
        'cuda_cores': int
    }

    Returns:
        A list of dictionaries containing specifications for each CUDA device.

    """
    # Initialize CUDA
    cuInit(0)

    num_gpus = ctypes.c_int()
    cuDeviceGetCount(ctypes.byref(num_gpus))

    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        device = ctypes.c_int()
        cuDeviceGet(ctypes.byref(device), i)

        name = b" " * 100
        cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
        spec["name"] = name.split(b"\0", 1)[0].decode()

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        cuDeviceComputeCapability(
            ctypes.byref(cc_major),
            ctypes.byref(cc_minor),
            device,
        )
        compute_capability = (cc_major.value, cc_minor.value)
        spec["compute_capability"] = compute_capability

        cores = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(cores),
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        )
        spec["cores"] = cores.value

        threads_per_core = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(threads_per_core),
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            device,
        )
        spec["concurrent_threads"] = cores.value * threads_per_core.value

        clockrate = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(clockrate),
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            device,
        )
        spec["gpu_clock_mhz"] = clockrate.value / 1000.0

        cuDeviceGetAttribute(
            ctypes.byref(clockrate),
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            device,
        )
        spec["mem_clock_mhz"] = clockrate.value / 1000.0

        context = ctypes.c_void_p()
        if cuCtxCreate(ctypes.byref(context), 0, device) == CUDA_SUCCESS:
            free_mem = ctypes.c_size_t()
            total_mem = ctypes.c_size_t()

            cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))

            spec["total_mem_mb"] = total_mem.value / 1024**2
            spec["free_mem_mb"] = free_mem.value / 1024**2

            spec["architecture"] = SEMVER_TO_ARCH.get(compute_capability, "unknown")
            spec["cuda_cores"] = SEMVER_TO_CORES.get(
                compute_capability,
                "unknown",
            )
            spec["cuda_cores"] = (
                spec["cuda_cores"] * cores.value if spec["cuda_cores"] != "unknown" else "unknown"
            )
            cuCtxDetach(context)

        device_specs.append(spec)
    return device_specs


def get_cuda_architecture() -> list[dict[str, Any]]:
    """Get CUDA architecture information for each GPU device.

    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'architecture': str,
    }

    Returns:
        A list of dictionaries containing architecture information for each CUDA device.

    """
    # Initialize CUDA
    cuInit(0)

    num_gpus = ctypes.c_int()
    cuDeviceGetCount(ctypes.byref(num_gpus))

    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        device = ctypes.c_int()
        cuDeviceGet(ctypes.byref(device), i)

        name = b" " * 100
        cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
        spec["name"] = name.split(b"\0", 1)[0].decode()

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        cuDeviceComputeCapability(
            ctypes.byref(cc_major),
            ctypes.byref(cc_minor),
            device,
        )
        compute_capability = (cc_major.value, cc_minor.value)
        spec["compute_capability"] = compute_capability
        context = ctypes.c_void_p()
        if cuCtxCreate(ctypes.byref(context), 0, device) == CUDA_SUCCESS:
            spec["architecture"] = SEMVER_TO_ARCH.get(compute_capability, "unknown")
            cuCtxDetach(context)

        device_specs.append(spec)
    return device_specs


def retrieve_cuda_version() -> float:
    """Retrieve the CUDA driver version.

    Returns:
        str: CUDA version in the format "major.minor" or "unknown" if retrieval fails.

    """
    try:
        version = ctypes.c_int()
        cuda.cuDriverGetVersion(ctypes.byref(version))
        major = version.value // 1000
        minor = (version.value % 1000) // 10
    except (AttributeError, OSError, ctypes.ArgumentError) as e:
        logger.warning("Could not retrieve CUDA version: %s", e)
        return -1
    else:
        return float(f"{major}.{minor}")


if __name__ == "__main__":
    print(json.dumps(get_cuda_device_specs(), indent=2))
    print(json.dumps(get_cuda_architecture(), indent=2))
    cuda_archtecture_dict = get_cuda_architecture()[0]  # Get the first device's architecture
    arch_option = (
        "sm_"
        + str(cuda_archtecture_dict["compute_capability"][0])
        + str(cuda_archtecture_dict["compute_capability"][1])
    )
    print(arch_option)

    cuda_version = retrieve_cuda_version()
    print(f"CUDA Version: {cuda_version}")
