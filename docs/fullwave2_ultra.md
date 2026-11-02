# fullwave2-ultra backend (experimental)

`Solver(..., use_ultra=True)` runs the simulation with the fullwave2-ultra binary, which is optimized for speed at 6 points per wavelength (`Grid.ppw`). It is experimental and its API may change. fullwave25 does not ship the binaries; obtain them from the Pinton lab.

Point the solver to the binary, or to the folder that holds it, with `path_fullwave_simulation_bin` or with the environment variable `FW2U_BATCH_BIN`.

```python
from pathlib import Path

import fullwave

solver = fullwave.Solver(
    work_dir=Path("./outputs/ultra"),
    grid=grid,  # 2D or 3D fullwave.Grid
    medium=medium,  # fullwave.Medium
    source=source,
    sensor=sensor,
    path_fullwave_simulation_bin=Path("/path/to/fullwave2-ultra/bin"),
    use_ultra=True,
)
sensor_output = solver.run()
# 2D only: sensor_output = solver.run_batch(sources)
```

[simple_plane_wave_ultra.py](../examples/simple_plane_wave/simple_plane_wave_ultra.py) is a complete example.

## Supported

- 2D grids with `run` and `run_batch`, 3D grids with `run`.
- Air cells, additive sources and the three source types (`"hard"`, `"hard_released_after"`, `"soft"`).
- Power-law attenuation with the power-law binaries, `use_exponential_attenuation=False` and `ultra_attenuation_parameters="centered"`. Without it, the backend uses frequency-independent attenuation: `alpha_coeff` is used as given and `alpha_power` is ignored.

Anything else, such as velocity sources or `run_batch` in 3D, raises `UltraUnsupportedError`.
