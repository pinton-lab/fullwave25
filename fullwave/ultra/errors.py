"""Exceptions and the warning of the experimental ``use_ultra`` backend."""

NO_BINARY_MESSAGE = (
    "Solver(use_ultra=True) is experimental and needs a fullwave2-ultra binary. "
    "fullwave25 does not ship the fullwave2-ultra binary (its own solver binaries are "
    "unchanged and serve use_ultra=False). Pass path_fullwave_simulation_bin= or set "
    "FW2U_BATCH_BIN to the binary (bench_2d_batch for a 2D grid, bench_3d_opt for a 3D grid). "
    "To get the binary, contact the Pinton lab, "
    "or find it at https://github.com/pinton-lab/fullwave2-ultra."
)


class UltraBinaryError(RuntimeError):
    """Raised when no usable fullwave2-ultra binary is given or found."""


class UltraUnsupportedError(NotImplementedError):
    """Raised when an option or input is not supported by ``use_ultra=True``."""


class FullwaveUltraExperimentalWarning(UserWarning):
    """Warns on the first use that ``use_ultra=True`` is experimental and may change."""
