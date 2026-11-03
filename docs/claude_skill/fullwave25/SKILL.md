---
name: fullwave25
description: How to set up, run and check a Fullwave 2.5 (fullwave25) ultrasound simulation in the standard way - the basic grid, medium, source, sensor and solver flow first, then transducers, transmit batches, beamforming, multi-GPU runs and the container. Use when writing, reviewing or debugging any code that builds a fullwave.Grid, fullwave.Medium, fullwave.Source, fullwave.Sensor, fullwave.Transducer or fullwave.Solver.
---

# Fullwave 2.5 simulations

Follow this workflow and these conventions, so every simulation built with fullwave25 reads the same way and avoids the known pitfalls. Start from the basic flow below, or from the closest example in `examples/` (the README lists them), not from an empty file.

## Basic flow: grid, medium, source, sensor, solver

Every simulation has these five steps. This one sends a plane wave from the top row into a medium with an inclusion and records the pressure in every cell (`examples/simple_plane_wave/simple_plane_wave.py` is the complete version).

```python
import numpy as np

import fullwave
from fullwave.utils.coordinates import map_to_coords

# 1. Grid: (depth, lateral) in metres; depth is the first axis.
f0 = 3e6  # Hz
c0 = 1540  # m/s
domain_size = (3e-2, 2e-2)
duration = domain_size[0] / c0 * 2  # s
grid = fullwave.Grid(domain_size=domain_size, f0=f0, duration=duration, c0=c0)

# 2. Medium: one map per property, each with the grid shape.
shape = (grid.nx, grid.ny)
sound_speed = np.full(shape, 1540.0)  # m/s
density = np.full(shape, 1000.0)  # kg/m^3
alpha_coeff = np.full(shape, 0.5)  # dB/(MHz^y cm)
alpha_power = np.full(shape, 1.0)  # -
beta = np.zeros(shape)  # -
inclusion = (slice(grid.nx // 3, 2 * grid.nx // 3), slice(grid.ny // 3, 2 * grid.ny // 3))
sound_speed[inclusion] = 1600.0
alpha_coeff[inclusion] = 0.75
alpha_power[inclusion] = 1.1
medium = fullwave.Medium(
    grid=grid,
    sound_speed=sound_speed,
    density=density,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power,
    beta=beta,
)

# 3. Source: the cells that emit, and one signal per cell.
source_mask = np.zeros(shape, dtype=bool)
source_mask[0, :] = True
source_coords = map_to_coords(source_mask)
pulse = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
    nt=grid.nt,
    duration=duration,
    ncycles=2,
    drop_off=2,
    f0=f0,
    p0=1e5,
)
source = fullwave.Source(
    p0=np.tile(pulse, (source_coords.shape[0], 1)),
    coords=source_coords,
    grid_shape=grid.shape,
)

# 4. Sensor: where and how often the pressure is recorded.
sensor = fullwave.Sensor(mod_x=1, mod_y=1, sampling_modulus_time=7)

# 5. Solver.
solver = fullwave.Solver(
    work_dir="./outputs/simple_plane_wave",
    grid=grid,
    medium=medium,
    source=source,
    sensor=sensor,
    source_type="soft",  # the signal is added to the field
)
sensor_output = solver.run()
```

Plot the medium (`medium.plot(export_path=...)`) before the run, and the pressure field after it (`fullwave.utils.plot_utils`, with `**plot_utils.mm_axes_kwargs(grid)` for mm axes).

## The output: shapes and sensor order

- `solver.run()` returns `sensor_output` with shape `[n_sensors, n_samples]`. `solver.run_batch(sources)` returns `[n_transmits, n_sensors, n_samples]`.
- **Time axis.** One sample every `sampling_modulus_time` solver steps: the sample interval is `grid.dt * sampling_modulus_time`, and `n_samples` is `grid.nt / sampling_modulus_time`, rounded.
- **Sensor axis: the pitfall.** Row `i` is the grid cell `sensor.outcoords[i]`, not the i-th element and not the i-th lateral position. A sensor built from a mask lists its cells in C (row-major) order: all lateral cells of the first depth row, then the next depth row, and in 3D the elevation index changes fastest. A mask that covers more than one depth row therefore interleaves the rows. Use `sensor.outcoords` to place each row, or keep the sensor to one row per depth.
- **Whole domain.** With `Sensor(mod_x=1, mod_y=1, ...)` every cell is recorded; `fullwave.utils.signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)` turns `sensor_output` into `[n_samples, nx, ny]` (`reshape_whole_sensor_to_nt_nx_ny_nz` in 3D).
- **Transducer.** The rows of `sensor_output` are the grid cells of the receiving element surfaces, not the elements. `rf_data = transducer.post_process_sensor_output(sensor_output)` sorts and averages them into the RF data, `[n_elements, n_samples]`, in element order; for a batch, call it once per transmit. The beamformers take `rf_data` in this element order.

## Scatterers for speckle

Tissue speckle comes from sub-resolution scatterers: random variations of the density. `fullwave.utils.generate_scatterer` returns a map of values centred on 1, which multiplies the density map before the `Medium` is built:

```python
rng = np.random.default_rng(seed=42)
scatterer, _ = fullwave.utils.generate_scatterer(
    grid=grid,
    ratio_scatterer_to_total_grid=0.38,  # fraction of the cells that carry a scatterer
    scatter_value_std=0.02,  # standard deviation of the density variation
    rng=rng,
)
density *= scatterer
```

The density standard deviation sets the echo strength of a tissue, and so the speckle brightness of the B-mode image. The values above are a starting point only: calibrate the fraction and the standard deviation of each tissue to your imaging system or target, for example by matching the speckle brightness of a measured phantom or tissue B-mode image. Leave the scatterers out for an anechoic region such as a cyst or a vessel. A `MediumBuilder` domain carries its own scatterers (`ScattererDomain`, or `domain.density *= scatterer` on a preset domain). Fix the seed of `rng` to repeat a speckle pattern.

## Debug at a low frequency first

Build and check a new simulation at a low `f0`, then raise it. The grid spacing follows the wavelength, so halving `f0` halves the number of points along each axis and the number of time steps: the grid has 1/4 of the cells in 2D and 1/8 in 3D, and the run does 1/8 (2D) or 1/16 (3D) of the cell updates. Check the geometry, the medium and the source at the low frequency, then run the target frequency once. Build a simulation in 2D before its 3D version for the same reason.

## Rules

- **Grid resolution.** Use `ppw=12`, the default and the recommended value. `ppw=10` also works. `ppw=8` is the minimum: the attenuation and the dispersion are validated at 8; do not go below it. Keep `cfl` at 0.2 unless you have checked another value.
- **Units.** SI everywhere (m, s, Hz, Pa, kg/m³, m/s), except `alpha_coeff` in dB/(MHz^y cm).
- **Axis order.** Depth first: (depth, lateral) in 2D and (depth, lateral, elevation) in 3D. Source and sensor coordinates are in C (row-major) order.
- **Attenuation range.** `alpha_coeff` up to 1.0 dB/(MHz^y cm) and `alpha_power` from 0.4 to 2.0, from 0.1 to 20 MHz.
- **Attenuation values are discretized.** The relaxation parameters come from a table, not from a fit per voxel. The table holds `alpha_coeff` = 0.0022, then 0.01 to 1.00 in steps of 0.01, and `alpha_power` = 0.4 to 2.0 in steps of 0.1. Each voxel is served the first table level at or above its request, without interpolation: a request of 0.505 runs as 0.51, and an exponent of 1.05 runs as 1.1. Values outside the table are clipped to its ends, with a warning. Pick values on the table levels when the exact value matters, and expect a smooth map to give an attenuation in steps. `Medium(..., scale_to_requested_alpha_coeff=True)` scales the attenuation back to the requested coefficient; the exponent stays on its level.
- **Source type.** `"soft"` adds the signal to the field; `"hard"` assigns it for the whole run, so the source cells reflect after the drive ends; `"hard_released_after"` assigns it while the drive plays.
- **Signal processing is a choice.** `run` returns the unfiltered `sensor_output`. Filtering (`highpass_cutoff_mhz`, `bandpass_cutoff_mhz`) and the dynamic range of a B-mode image are decisions to state, not defaults to add without saying so.
- **Duration and pulse.** Set `duration` to cover the round trip: `2 * depth / c0` for echoes from the bottom of the domain, plus the pulse length. The pulse is set by its center frequency, its number of cycles and its peak pressure (`ncycles` and `p0` of `gaussian_modulated_sinusoidal_signal`, or `fullwave.Pulse(pressure=..., cycles=...)` for a transducer). With `beta = 0` the simulation is linear and the peak pressure only scales the output; with `beta > 0` the peak pressure sets the strength of the nonlinear effects, so give it a physical value.
- **Reproducibility.** Pin the fullwave25 version in your environment. The solver binaries come from the release pinned in the package (`BINARY_RELEASE_TAG`); `FULLWAVE25_BINARY_TAG` selects another release. Fix the seeds of every random generator (scatterers, phantoms).

## Air

Cells marked as air hold zero pressure, so they reflect the wave as a pressure-release boundary, like a gas pocket or a probe backed by air. Pass a boolean map with the grid shape. The solver ignores the sound speed and the density of the air cells, so you do not need to set air values in `sound_speed` or `density`:

```python
air_map = np.zeros(grid.shape, dtype=bool)
air_map[air_region] = True
medium = fullwave.Medium(
    grid=grid,
    sound_speed=sound_speed,
    density=density,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power,
    beta=beta,
    air_map=air_map,
)
```

`air_coords=` takes the coordinates instead of a map. For the curved `c5_2v`, `transducer.make_surface_reflective_with_air()` returns the air map behind the array face. `examples/simple_plane_wave/simple_plane_wave_air.py` and `examples/wave_3d/simple_plane_wave_3d_with_air.py` scatter single air cells inside an inclusion.

## Run in memory for speed

`Solver(..., run_on_memory=True)` keeps the input files and the recorded field in a RAM directory (`/run/user/<uid>`) instead of on the disk, which makes the run faster. It needs RAM for the whole record. Before a run, check that the free space of `/run/user/<uid>` and the free memory of the machine exceed the size of the record (next section). With `run_on_memory=False`, the run writes its files, including the record `genout.dat`, under `work_dir`; point `work_dir` at a disk with room for them. A static map (`is_static_map=True`) needs the files on the disk.

## Size of the record

Estimate the record before a run: `n_sensors * n_samples * 4` bytes, and Python needs one or two more copies of it while it loads and plots. A transducer sensor records one or a few rows of cells. A whole-domain record grows with every cell: the 2D C5-2V abdominal wall example (`examples/convex_transducer/convex_transducer_abdominal_wall.py`) at 1 MHz peaks at 39.6 GB of RAM. To reduce the record:

- Record every second or fourth cell: `Sensor(mod_x=2, mod_y=2, sampling_modulus_time=...)` (a sparse grid), or a mask with a stride.
- Record fewer time samples: a larger `sampling_modulus_time`. A movie needs 50 to 100 frames, not every solver step.
- Check the setup at a lower frequency first (see above); the record shrinks with the grid.

## Advanced: transducers and imaging

1. **Transducer.** Take a named array, `fullwave.Transducer.l7_4(grid)`, `p4_1c(grid)` or `c5_2v(grid)`, or build one with `fullwave.TransducerGeometry`. Set the transmit with `plane_wave(angle_deg=...)`, `focus(focus_m=(depth_m, lateral_m))`, `diverging(virtual_source_m=...)` or `synthetic_aperture(element)`. Transmit positions are in metres from the top and the left edge of the grid.
2. **Medium with a linear array** (`l7_4`, `p4_1c`, `LinearTransducer`): make the grid wider than the array, start the tissue at `transducer.face_depth_m` (for example `start_depth=transducer.face_depth_m` in an `AbdominalWallDomain`), and pass the maps through `transducer.fill_probe_above_face(...)` (or register `transducer.probe_fill_domain()` last in a `MediumBuilder`). Measure target and focus depths from the face. The named arrays use `source_type="hard"`.
3. **Solver.** `fullwave.Solver(work_dir=..., grid=grid, medium=medium, transducer=transducer)`; the transducer supplies the source and the sensor.
4. **Transmit sequences** on one medium go through `solver.run_batch(sources)`, not a loop of `run`; it sizes each simulation to the free GPU memory. Convert each transmit's `sensor_output` to `rf_data` with `transducer.post_process_sensor_output(...)`.
5. **B-mode image.** `fullwave.beamformer.MachBeamformer` on a GPU (`pip install "fullwave25[mach]"`), or `fullwave.beamformer.Beamformer` on the CPU for teaching. Beamformer depths are measured from the probe face.

Check `transducer.plot_source_mask(...)` and `plot_sensor_mask(...)` before the first run.

### Building the beamformer

Build `MachBeamformer` with every argument named, so each choice is visible:

```python
from fullwave.beamformer import MachBeamformer, face_crossing_time_s, receiver_positions_m

beamformer = MachBeamformer(
    lateral_position_m=lateral_position_m,  # B-mode columns, 0 at the array center
    axial_position_m=axial_position_m,  # B-mode rows, depth below the probe face
    receiver_positions_m=receiver_positions_m(transducer),
    sound_speed=grid.c0,
    center_frequency_hz=grid.f0,
    sampling_frequency_hz=1 / (grid.dt * transducer.sampling_modulus_time),
    f_number=1.0,  # depth / receive aperture width
    time_zero_s=face_crossing_time_s(transducer),
    tukey_alpha=0.5,  # taper of the receive aperture
)
rf_data = transducer.post_process_sensor_output(sensor_output)
tx_arrival_s = beamformer.plane_wave_arrival_s(angle_deg=0.0)
# complex, [n_depth, n_lateral]
b_mode_image = beamformer.beamform(rf_data, tx_arrival_s=tx_arrival_s)
```

`focused_arrival_s(focus_m=...)` and `synthetic_aperture_arrival_s(element=...)` give the transmit arrival for the other transmits. `time_zero_s` is the time the pulse crosses the probe face, so set the transmit before calling `face_crossing_time_s`. Display `20 * log10(abs(b_mode_image) / abs(b_mode_image).max())` over a dynamic range you state.

## Media from presets

`fullwave.MediumBuilder` combines domains into one medium; a domain registered later overrides an earlier one where they overlap.

```python
from fullwave import MediumBuilder, presets

background = presets.BackgroundDomain(grid=grid, background_property_name="liver")
abdominal_wall = presets.AbdominalWallDomain(grid=grid, start_depth=transducer.face_depth_m)
mb = MediumBuilder(grid=grid)
mb.register_domain(background)
mb.register_domain(abdominal_wall)
mb.register_domain(transducer.probe_fill_domain())  # last, so the probe fill is not overwritten
mb.plot_current_map(export_path="medium.png")
medium = mb.run()
```

The presets are `BackgroundDomain` (one tissue everywhere), `AbdominalWallDomain` (a layered wall), `CombinedPhantomDomain` (point targets and hypoechoic and hyperechoic cylinders) and `ScattererDomain`, next to `SimpleDomain` for your own maps.

## Multiple GPUs

- Pass the devices to the solver: `fullwave.Solver(..., cuda_device_id=[0, 1, 2, 3])` or `cuda_device_id="0,1,2,3"`. One integer or string selects one GPU; the default is GPU 0.
- As in CuPy and PyTorch, the IDs count within `CUDA_VISIBLE_DEVICES` when it is set: with `CUDA_VISIBLE_DEVICES=2,3`, `cuda_device_id=[0, 1]` runs on GPUs 2 and 3, and `cuda_device_id=2` is an error.
- The solver divides the depth rows of the domain equally among the GPUs (to within one row) and exchanges 8 rows at each boundary between neighbouring GPUs. The memory per GPU therefore falls with the number of GPUs, and the run time scales linearly with the number of GPUs.
- Every GPU holds the whole source signal matrix, so each source point and each time step of the signal costs memory on every GPU.

### Use the GPU memory estimate

Before each run, `run` and `run_batch` write an estimate of the memory of each GPU to the log (`gpu_memory_estimate=True`, the default), for example:

```text
GPU memory estimate [GPU 0] (relaxation mode): 0.52 GiB  (depth=2189 +0 halo, lateral=1986)
```

Compare it with the free memory that `nvidia-smi` or `nvtop` shows before a long run. If it does not fit, reduce the grid (a lower `f0` or a smaller domain), add GPUs, or for a batch lower `max_members_per_run`. `run_batch` already lowers the number of transmits per simulation to fit the free GPU memory.

## Container

The released solver binaries need glibc 2.34 or newer (for example Ubuntu 22.04). On an older system, run fullwave25 in the container described in `docker/README.md`:

```sh
docker build -f docker/Dockerfile -t fullwave25 .
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD":/work fullwave25 python my_simulation.py
```

The host needs an NVIDIA driver and Docker with the NVIDIA Container Toolkit. Select GPUs inside the container with `cuda_device_id`, as outside it. Where Docker is not available, convert the image for Apptainer as `docker/README.md` describes.

## Troubleshooting

- **`GLIBC_2.34' not found`** at the start of a run: the system's glibc is older than the released binaries need; use the container.
- **A warning that the driver's CUDA version has no binary for the GPU architecture** and that the closest compatible version is used: the run continues with that binary.
- **NaN or exploding values** in `sensor_output`: check `cfl` (keep 0.2) and `ppw` (at least 8), and that the maps hold no NaN and no zero sound speed or density outside air cells.
- **Out of GPU memory:** read the GPU memory estimate in the log (see "Multiple GPUs").
- **Out of RAM, or a killed process:** see "Size of the record". A killed in-memory run can leave its directory in `/run/user/<uid>`; check the free space there and remove the leftover directories of your own runs.

## Checks before trusting a result

- The medium plot shows the intended maps (with a linear array, the tissue starts at the probe face).
- A short run (low frequency, short `duration`, one transmit) finishes before the full run.
- For a new attenuation setting, `examples/attenuation_dispersion/attenuation_dispersion.py` measures the attenuation and the phase velocity against the power law.

## Citation

Cite Fullwave 2.5 as given in the README, and also cite mach when you use `MachBeamformer`.
