"""Power-law attenuation computed by a compatible fullwave2-ultra binary from a parameter file."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from fullwave.solver.binary_markers import binary_has_marker

from . import dat_io
from .binary import binary_name
from .errors import UltraUnsupportedError

if TYPE_CHECKING:
    from numpy.typing import NDArray

MARKER = b"alpha_power.dat"
UNLIMITED_MARKER = b"--unlimited-materials"
SETTINGS_FILES = {
    "centered": "attenuation_parameters_centered_v1.dat",
    "author": "attenuation_parameters_author_v1.dat",
}
PARAMETER_FILE_NAME = "attenuation_parameters.dat"


def check_unlimited_materials(
    *,
    unlimited_materials: bool,
    parameters: str | Path | None,
    use_ultra: bool,
) -> None:
    """Raise unless ``ultra_unlimited_materials`` comes with ``use_ultra`` and a parameter file.

    Parameters
    ----------
    unlimited_materials : bool
        The ``ultra_unlimited_materials`` argument of ``Solver``.
    parameters : str | Path | None
        The ``ultra_attenuation_parameters`` argument of ``Solver``.
    use_ultra : bool
        The ``Solver`` argument of the same name.

    Raises
    ------
    UltraUnsupportedError
        If the request cannot run, with the fix in the message.
    """
    if not unlimited_materials:
        return
    if not use_ultra:
        msg = (
            "ultra_unlimited_materials needs use_ultra=True; without it the argument would be "
            "ignored. Set use_ultra=True or remove ultra_unlimited_materials."
        )
        raise UltraUnsupportedError(msg)
    if parameters is None:
        msg = (
            "ultra_unlimited_materials needs ultra_attenuation_parameters. Pass a parameter "
            "file or remove ultra_unlimited_materials."
        )
        raise UltraUnsupportedError(msg)


class PowerLawRequest(NamedTuple):
    """The parameter file and the binary that a power-law attenuation request resolved to."""

    parameter_file: Path
    binary: Path


def check_request(
    *,
    parameters: str | Path,
    use_ultra: bool,
    use_exponential_attenuation: bool,
    binary: str | Path | None,
    is_3d: bool,
    unlimited_materials: bool = False,
) -> PowerLawRequest:
    """Check a request for the parameter-file attenuation and resolve its files.

    Parameters
    ----------
    parameters : str | Path
        The ``ultra_attenuation_parameters`` argument: "centered", "author" or a file path.
    use_ultra : bool
        The ``Solver`` argument of the same name.
    use_exponential_attenuation : bool
        The ``Solver`` argument of the same name.
    binary : str | Path | None
        The ``path_fullwave_simulation_bin`` argument: a binary file, or a folder that holds
        the power-law build for the grid dimension.
    is_3d : bool
        True for a 3D grid.
    unlimited_materials : bool, optional
        The ``ultra_unlimited_materials`` argument.

    Returns
    -------
    PowerLawRequest
        The parameter file and the binary file.

    Raises
    ------
    UltraUnsupportedError
        If the request cannot run, with the fix in the message.
    """
    if not use_ultra:
        msg = (
            "ultra_attenuation_parameters needs use_ultra=True; without it the file would be "
            "ignored and the attenuation would change silently. Set use_ultra=True or remove "
            "ultra_attenuation_parameters."
        )
        raise UltraUnsupportedError(msg)
    if use_exponential_attenuation:
        msg = (
            "ultra_attenuation_parameters computes a power-law attenuation, so it cannot be "
            "combined with use_exponential_attenuation=True. Set use_exponential_attenuation="
            "False or remove ultra_attenuation_parameters."
        )
        raise UltraUnsupportedError(msg)
    binary_file = _find_binary(binary, is_3d=is_3d)
    if not binary_has_marker(binary_file, marker=MARKER):
        msg = (
            f"{binary_file} does not read {MARKER.decode()}, so it would run without the "
            "power-law attenuation. Pass a fullwave2-ultra binary that supports "
            "ultra_attenuation_parameters, or remove ultra_attenuation_parameters."
        )
        raise UltraUnsupportedError(msg)
    if unlimited_materials and not binary_has_marker(binary_file, marker=UNLIMITED_MARKER):
        msg = (
            f"{binary_file} does not support ultra_unlimited_materials, so the option would "
            "be ignored. Pass a newer fullwave2-ultra binary, or remove "
            "ultra_unlimited_materials."
        )
        raise UltraUnsupportedError(msg)
    return PowerLawRequest(
        parameter_file=_find_parameter_file(parameters, binary_file=binary_file),
        binary=binary_file,
    )


def _find_binary(binary: str | Path | None, *, is_3d: bool) -> Path:
    if binary is None:
        msg = (
            "ultra_attenuation_parameters needs path_fullwave_simulation_bin, the binary file "
            "or the folder that holds it; the FW2U_BATCH_BIN fallback is not used."
        )
        raise UltraUnsupportedError(msg)
    binary = Path(binary)
    if binary.is_dir():
        binary = binary / binary_name(is_3d=is_3d, power_law=True)
    if not binary.is_file():
        msg = (
            f"The binary {str(binary)!r} for ultra_attenuation_parameters does not exist. "
            "Pass the binary file, or a folder that holds the power-law build "
            f"{binary_name(is_3d=is_3d, power_law=True)!r}."
        )
        raise UltraUnsupportedError(msg)
    return binary


def _find_parameter_file(parameters: str | Path, *, binary_file: Path) -> Path:
    if isinstance(parameters, str) and parameters in SETTINGS_FILES:
        file_name = SETTINGS_FILES[parameters]
        folders = (binary_file.parent, binary_file.parent.parent / "params")
        for folder in folders:
            if (folder / file_name).is_file():
                return folder / file_name
        searched = ", ".join(str(folder / file_name) for folder in folders)
        msg = (
            f"ultra_attenuation_parameters {parameters!r} is not a file; the settings file "
            f"{file_name} was not found in: {searched}. Put the file next to the binary, "
            "or pass its path."
        )
        raise UltraUnsupportedError(msg)
    parameter_file = Path(parameters)
    if not parameter_file.is_file():
        msg = (
            f"ultra_attenuation_parameters {str(parameter_file)!r} is not a file. Pass one of "
            f"{sorted(SETTINGS_FILES)} or the path of an existing parameter file."
        )
        raise UltraUnsupportedError(msg)
    return parameter_file


def write_files(
    *,
    outdir: str | Path,
    parameters: str | Path,
    alpha_coeff: NDArray,
    alpha_power: NDArray,
    f0: float,
) -> None:
    """Write the inputs of the power-law attenuation into a run directory.

    Parameters
    ----------
    outdir : str | Path
        The run directory.
    parameters : str | Path
        The parameter file; it is copied byte for byte.
    alpha_coeff : NDArray
        Attenuation coefficient [dB/(MHz^y cm)] on the extended grid.
    alpha_power : NDArray
        Attenuation exponent y on the extended grid.
    f0 : float
        Center frequency of the grid [Hz].
    """
    outdir = Path(outdir)
    dat_io.write_map(outdir / "alpha_coeff.dat", alpha_coeff)
    dat_io.write_map(outdir / "alpha_power.dat", alpha_power)
    dat_io.write_float(outdir / "f0.dat", f0)
    shutil.copyfile(parameters, outdir / PARAMETER_FILE_NAME)
