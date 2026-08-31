"""Drive the source by addition rather than by assignment.

A hard source sets the pressure of one grid row on every step, while the
staggered derivative spans `2 * m_spatial_order` pressure points. One hard source row
is therefore not a consistent plane-wave condition, and the plane launches a low
amplitude. Measured in a uniform lossless medium, where the maximum intensity
projection must equal the source pressure squared, a hard source reads 0.989886
at 16 points per wavelength, 0.954260 at 8 and 0.997504 at 32.

An additive source does not radiate the value it is given. Adding `s` to the
pressure of one plane of nodes on every step is a volume rate of `s / dt` over a
thickness `dx`. A source of that rate radiates `s * dx / (2 * c * dt)` in
each direction. A caller who wants to radiate `p` must inject `p * 2 * c * dt / dx`.
With that scale an additive source reads 0.999877 of the same reference.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fullwave.source import Source

if TYPE_CHECKING:
    import fullwave

CLAMPED = "clamped"
ADDITIVE = "additive"
SOURCE_TYPES = (CLAMPED, ADDITIVE)


def _is_on_the_device(array: object) -> bool:
    """Return whether an accelerator holds the array rather than the host."""
    return not isinstance(array, np.ndarray) and hasattr(array, "get")


def _at_the_source(array: NDArray, index: tuple[NDArray, ...]) -> NDArray:
    """Return the values at the source positions, as a host array.

    The array is indexed where it lives, so a map on an accelerator never moves
    whole.

    Parameters
    ----------
    array : NDArray
        A map over the grid, on the host or on an accelerator.
    index : tuple[NDArray, ...]
        One index array for each axis of the grid.

    Returns
    -------
    NDArray
        The sampled values, on the host.
    """
    if _is_on_the_device(array):
        return array[index].get().astype(float, copy=False)
    return np.asarray(array, dtype=float)[index]


def is_additive(source_type: str) -> bool:
    """Return whether this source type adds to the field rather than assigning it."""
    return source_type == ADDITIVE


def source_row(coords: NDArray[np.int64], grid_shape: tuple[int, ...]) -> int:
    """Return the row the source occupies, and check that it fills one whole row.

    Parameters
    ----------
    coords : NDArray[np.int64]
        Source positions, shape [n_sources, ndim].
    grid_shape : tuple[int, ...]
        Shape of the grid the coordinates index.

    Returns
    -------
    int
        The index along the first axis that the source occupies.

    Raises
    ------
    ValueError
        If the source does not fill exactly one whole row. The scale below is
        the whole-row relation, so it does not describe any other geometry.

    Notes
    -----
    This reads the coordinates and never builds a mask, because a mask over a
    3D grid allocates the whole volume.
    """
    coords = np.asarray(coords)
    rows = np.unique(coords[:, 0])
    across = int(np.prod(grid_shape[1:]))
    if rows.size != 1 or len(coords) != across:
        error_msg = (
            f"source_type='{ADDITIVE}' needs a source that fills one whole row of "
            f"the grid, and this source has {len(coords)} positions on "
            f"{rows.size} rows, against {across} positions in one row"
        )
        raise ValueError(error_msg)
    return int(rows[0])


def additive_signal_scale(
    sound_speed: NDArray[np.float64] | float,
    dt: float,
    dx: float,
) -> NDArray[np.float64] | float:
    """Return the factor that makes an additive source radiate its nominal pressure."""
    return 2.0 * sound_speed * dt / dx


def node_sound_speeds(
    sound_speed: NDArray[np.float64] | float,
    coords: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return the sound speed at each source position.

    The relation is local, so a source that crosses a sound-speed contrast
    carries one scale per node rather than one scale for the whole row.

    `sound_speed` is a scalar, a map over the grid, or one value per source
    position. A grid map has the dimension of the grid, so only the last form is
    one dimensional with as many entries as there are positions.
    """
    if _is_on_the_device(sound_speed):
        return _at_the_source(sound_speed, tuple(np.asarray(coords).T))
    values = np.asarray(sound_speed, dtype=float)
    if values.ndim == 0:
        return np.full(len(coords), float(values))
    if values.ndim == 1 and len(values) == len(coords):
        return values
    return values[tuple(np.asarray(coords).T)]


def _mechanism_response(
    relaxation: dict,
    side: int,
    kappa: NDArray[np.float64],
    angular_frequency: float,
) -> NDArray[np.complex128]:
    """Return gamma for one side of the dispersion relation, at each position."""
    total = np.zeros(len(kappa), dtype=complex)
    mechanism = 1
    while f"d_x{side}_nu{mechanism}" in relaxation:
        strength = np.asarray(relaxation[f"d_x{side}_nu{mechanism}"], dtype=float)
        rate = np.asarray(relaxation[f"alpha_x{side}_nu{mechanism}"], dtype=float)
        total += (strength / kappa**2) / (strength / kappa + rate + 1j * angular_frequency)
        mechanism += 1
    return total


def relaxation_phase_speed(
    relaxation: dict,
    sound_speed: NDArray[np.float64] | float,
    frequency: float,
) -> NDArray[np.float64]:
    """Return the phase speed at each position, in m/s.

    This is the dispersion relation of the multiple relaxation model, derived in
    the appendix of the Fullwave 2 paper. With
    `G_i = 1 / kappa_x_i - gamma_i`, the wavenumber is
    `k = (omega / c) * (G_1 * G_2) ** -0.5`, and the phase speed is
    `omega / k.real`.

    The relaxation mechanisms move the wave off the stored sound speed, and an
    additive source radiates in proportion to the speed the wave actually travels
    at. A signal scaled by the stored speed therefore radiates the wrong
    amplitude by the ratio of the two speeds.

    Parameters
    ----------
    relaxation : dict
        One value for each relaxation parameter at each position.
    sound_speed : NDArray[np.float64] | float
        The stored sound speed at each position, or one value for all [m/s].
    frequency : float
        The frequency to read the phase speed at [Hz].

    Returns
    -------
    NDArray[np.float64]
        The phase speed at each position [m/s].

    """
    angular_frequency = 2 * np.pi * frequency
    kappa_1 = np.asarray(relaxation["kappa_x1"], dtype=float)
    kappa_2 = np.asarray(relaxation["kappa_x2"], dtype=float)
    side_1 = 1 / kappa_1 - _mechanism_response(relaxation, 1, kappa_1, angular_frequency)
    side_2 = 1 / kappa_2 - _mechanism_response(relaxation, 2, kappa_2, angular_frequency)
    stored = np.asarray(sound_speed, dtype=float)
    wavenumber = (angular_frequency / stored) * (side_1 * side_2) ** -0.5
    return angular_frequency / wavenumber.real


def _relaxation_at_the_source(
    medium: "fullwave.Medium",
    coords: NDArray[np.int64],
) -> tuple[dict | None, NDArray[np.float64]]:
    """Return the relaxation parameters and the sound speed at the source positions.

    A medium with no relaxation model returns no parameters.

    Parameters
    ----------
    medium : fullwave.Medium
        The medium the source sits in.
    coords : NDArray[np.int64]
        Source positions, shape [n_sources, ndim].

    Returns
    -------
    tuple[dict | None, NDArray[np.float64]]
        The parameters at each source position, or None, and the sound speed at
        each source position [m/s].

    """
    relaxation = getattr(medium, "relaxation_param_dict", None)
    if relaxation is not None:
        index = tuple(np.asarray(coords).T)
        parameters = {name: _at_the_source(value, index) for name, value in relaxation.items()}
        return parameters, node_sound_speeds(medium.sound_speed, coords)
    if hasattr(medium, "relaxation_parameters_at"):
        return medium.relaxation_parameters_at(coords)
    return None, node_sound_speeds(medium.sound_speed, coords)


def additive_source(
    p0: NDArray[np.float64],
    coords: NDArray[np.int64],
    grid_shape: tuple[int, ...],
    sound_speed: NDArray[np.float64] | float,
    dt: float,
    dx: float,
) -> Source:
    """Return an additive source that radiates the pressure `p0` carries.

    Parameters
    ----------
    p0 : NDArray[np.float64]
        Pressure at each source position, shape [n_sources, nt]. Its row order
        must match `coords`.
    coords : NDArray[np.int64]
        Source positions, shape [n_sources, ndim], filling one whole row.
    grid_shape : tuple[int, ...]
        Shape of the grid the coordinates index.
    sound_speed : NDArray[np.float64] | float
        Sound speed, as a scalar or as a map over the grid.
    dt : float
        Time step, in s.
    dx : float
        Grid spacing, in m.

    Returns
    -------
    Source
        A source with no assigned positions and a scaled additive signal.
    """
    coords = np.asarray(coords, dtype=np.int64)
    source_row(coords, grid_shape)
    scale = additive_signal_scale(node_sound_speeds(sound_speed, coords), dt, dx)
    source = Source(
        coords=np.empty((0, len(grid_shape)), dtype=np.int64),
        grid_shape=grid_shape,
        p0_additive=np.asarray(p0, dtype=float) * scale[:, None],
        coords_additive=coords,
    )
    source.additive_signal_is_scaled = True
    return source


def as_additive_source(
    source: Source,
    grid: "fullwave.Grid",
    medium: "fullwave.Medium",
    *,
    use_exponential_attenuation: bool = False,
) -> Source:
    """Return an additive copy of a hard source, scaled to radiate the same wave.

    Parameters
    ----------
    source : Source
        A source whose pressure positions fill one whole row of the grid.
    grid : fullwave.Grid
        The grid the source is defined on, read for `dt` and `dx`.
    medium : fullwave.Medium
        The medium. Its relaxation parameters give the phase speed at each source
        position. A medium built with `use_gpu=True` holds its maps on the
        accelerator, and each map is read at the source positions where it lies.
    use_exponential_attenuation : bool, optional
        True scales the signal by the stored sound speed, because the exponential
        attenuation model does not carry the relaxation dispersion relation.
        False, the default, reads the phase speed of the relaxation model.

    Returns
    -------
    Source
        A source with no assigned positions and a scaled additive signal.
    """
    coords = np.asarray(source.incoords, dtype=np.int64)
    if use_exponential_attenuation:
        speed = node_sound_speeds(medium.sound_speed, coords)
    else:
        parameters, sound_speed = _relaxation_at_the_source(medium, coords)
        speed = (
            relaxation_phase_speed(parameters, sound_speed, float(grid.f0))
            if parameters
            else sound_speed
        )
    return additive_source(
        source.p0,
        coords,
        source.grid_shape,
        speed,
        float(grid.dt),
        float(grid.dx),
    )
