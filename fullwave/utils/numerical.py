"""Utility functions for numerical operations."""

import os
import random

import numpy as np
import scipy


def seed_everything(seed: int) -> None:
    """Apply seed."""
    random.seed(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002


def normalize_255(image: np.ndarray) -> np.ndarray:
    """Normalize image data to the 0-255 range."""
    return (image - image.min()) / (image.max() - image.min() + 1e-6) * 255


def matlab_round(value: float) -> int:
    """MATLAB style rounding of a value."""
    return np.round(value + 1e-9).astype(int)


def matlab_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter similar to MATLAB's implementation."""
    return scipy.ndimage.gaussian_filter(
        image.astype(float),
        sigma=sigma,
        radius=np.ceil(2 * sigma).astype(int),
    )


def matlab_interp2easy(
    image: np.ndarray,
    interpolation_x: float,
    interpolation_y: float,
) -> np.ndarray:
    """Perform interpolation similar to MATLAB's interp2 function."""
    dxi = 1 / matlab_round(image.shape[0] * interpolation_x - 1)
    xvec = np.arange(0, 1 + dxi, dxi)
    xvec = xvec * (image.shape[0] - 1)

    dyi = 1 / matlab_round(image.shape[1] * interpolation_y - 1)
    yvec = np.arange(0, 1 + dyi, dyi)
    yvec = yvec * (image.shape[1] - 1)
    xi, yi = np.meshgrid(xvec, yvec)

    return (
        scipy.ndimage.map_coordinates(image, [xi.ravel(), yi.ravel()], order=0, mode="nearest")
        .reshape(yvec.shape[0], xvec.shape[0])
        .T
    )


def matlab_interp2easy_3d(
    image: np.ndarray,
    interpolation_x: float,
    interpolation_y: float,
    interpolation_z: float,
) -> np.ndarray:
    """Perform interpolation similar to MATLAB's interp2 function."""
    dxi = 1 / matlab_round(image.shape[0] * interpolation_x - 1)
    xvec = np.arange(0, 1 + dxi, dxi)
    xvec = xvec * (image.shape[0] - 1)

    dyi = 1 / matlab_round(image.shape[1] * interpolation_y - 1)
    yvec = np.arange(0, 1 + dyi, dyi)
    yvec = yvec * (image.shape[1] - 1)

    dzi = 1 / matlab_round(image.shape[2] * interpolation_z - 1)
    zvec = np.arange(0, 1 + dzi, dzi)
    zvec = zvec * (image.shape[2] - 1)

    xi, yi, zi = np.meshgrid(xvec, yvec, zvec, indexing="ij")

    return (
        scipy.ndimage.map_coordinates(
            image,
            [xi.ravel(), yi.ravel(), zi.ravel()],
            order=1,
            mode="nearest",
        ).reshape(xvec.shape[0], yvec.shape[0], zvec.shape[0])
        # .reshape(yvec.shape[0], xvec.shape[0], zvec.shape[0])
        # .T
    )


def detect_envelope(image: np.ndarray) -> np.ndarray:
    """Compute the envelope of a signal using the Hilbert transform."""
    return np.abs(scipy.signal.hilbert(image, axis=0))


def cut_domain(
    map_data: np.ndarray,
    i_event: int,
    num_x: int,
    num_y: int,
    n_events: int,
    beam_spacing: float,
) -> np.ndarray:
    """Extract a subdomain from map_data based on event parameters."""
    orig = int(
        np.round(map_data.shape[0] / 2)
        - np.round(num_x / 2)
        - np.round(((i_event + 1) - (n_events + 1) / 2) * beam_spacing)
        - 6,
    )
    return map_data[orig : orig + num_x, 0:num_y]
