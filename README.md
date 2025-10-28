# Fullwave 2.5

## Prerequisite

- This system only works in a Linux environment.
- This simulation requires NVIDIA GPU to execute.
- You might need multiple GPUs for 3D simulation.
- We recommend setting up an SSH key for GitHub, if you haven't done already. The repository changes over time to fix bugs and add new features.
  - for ssh key generation
    - please see: [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
  - for ssh key registration to your github account
    - please see: [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- after that, you can clone the repository through

  ```sh
  git clone git@github.com:pinton-lab/fullwave-python.git
  ```

## Resources

- If you are not familiar with the tools below, please refer to the provided links.
  - VSCode
    - [Official Visual Studio Code documentation](https://code.visualstudio.com/docs)
    - [Visual Studio Code Tutorial for Beginners by Udacity](https://www.udacity.com/blog/2025/09/visual-studio-code-tutorial-for-beginners-productivity-tips-and-extensions.html)
  - Git
    - [Git Tutorial by GeeksForGeeks](https://www.geeksforgeeks.org/git/git-tutorial/)
    - [Git Tutorial by W3 schools](https://www.w3schools.com/git/default.asp)
    - [Using Git source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)
  - UV
    - [Python UV: The Ultimate Guide to the Fastest Python Package Manager](https://www.datacamp.com/tutorial/python-uv)

## Install

We use [uv](https://docs.astral.sh/uv/) for package project and virtual environment management.

If uv is not installed, run below.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run below to install the development environment.

```sh
git clone git@github.com:pinton-lab/fullwave-python.git
cd fullwave-python
make install-all-extras # for running examples
# or
make install # for the core library installation
```

To test the installation, run

```sh
make test
```

## Example

Please see

- 2D plane wave
  - `examples/simple_plane_wave/simple_plane_wave_demo.py`
- 3D plane wave
  - `examples/3d_wave/simple_plane_wave_3d.py`

## Attention

- Note that 3D utilities are under development.
- The simulation grid is defined as (x, y, z) = (depth, lateral, elevational)
  - This order is due to the multiple-GPU development efficiency.
  - Multi-GPU domain decomposition is processed in the depth dimension.
  - The index of the input coordinates (acoustic source location) is defined in C-array order (row-major) within the simulation, regardless of your setup. This is for the efficiency of multi-GPU development.

## Usage 2D

Here are the main steps to run the Fullwave simulation

1. Define the computational grid.
2. Define the properties of the acoustic medium.
3. Define the acoustic source.
4. Define the sensor.
5. Execute the simulation.

### Import libraries

```py
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process
```

### Define the working directory

```py
work_dir = Path("./outputs/") / "simple_plane_wave"
work_dir.mkdir(parents=True, exist_ok=True)
```

### Define the computational grid

```py
domain_size = (3e-2, 3e-2)  # meters
f0 = 3e6  # Hz
c0 = 1540  # m/s
duration = domain_size[0] / c0 * 2  # seconds

# setup the Grid instance
grid = fullwave.Grid(
  domain_size=domain_size,
  f0=f0,
  duration=duration,
  c0=c0,
)
```

### Define the properties of the acoustic medium

```py
sound_speed = 1540  # m/s
density = 1000  # kg/m^3
alpha_coeff = 0.5  # dB/(MHz^gamma * cm)
alpha_power = 1.0  # [-]
beta = 0.0

sound_speed_map = sound_speed * np.ones((grid.nx, grid.ny))
density_map = density * np.ones((grid.nx, grid.ny))
alpha_coeff_map = alpha_coeff * np.ones((grid.nx, grid.ny))
alpha_power_map = alpha_power * np.ones((grid.nx, grid.ny))
beta_map = beta * np.ones((grid.nx, grid.ny))

# setup the Medium instance
medium = fullwave.Medium(
  grid=grid,
  sound_speed=sound_speed_map,
  density=density_map,
  alpha_coeff=alpha_coeff_map,
  alpha_power=alpha_power_map,
  beta=beta_map,
  # air_map=air_map,
)
```

### Define the acoustic source

```py
# define where to put the pressure source [nx, ny]
p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
p_mask[0:1, :] = True

# define the pressure source [n_sources, nt]
p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
  nt=grid.nt,
  f0=f0,
  duration=duration,
  ncycles=2,
  drop_off=2,
  p0=1e5,
)
p0 = np.zeros((p_mask.sum(), grid.nt))
p0[:] = p0_vec

# setup the Source instance
source = fullwave.Source(p0, p_mask)
```

### Define the sensor

```py
sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
sensor_mask[:, :] = True

# setup the Sensor instance
sensor = fullwave.Sensor(mask=sensor_mask, sampling_interval=7)
```

### Execute the simulation

```py
# setup the Solver instance
fw_solver = fullwave.Solver(
  work_dir=work_dir,
  grid=grid,
  medium=medium,
  source=source,
  sensor=sensor,
)

# execute the solver
sensor_output = fw_solver.run()
```

### Visualization

```py
propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(
  sensor_output,
  grid,
)

p_max_plot = np.abs(propagation_map).max().item() / 4
plot_utils.plot_wave_propagation_with_map(
  propagation_map=propagation_map,
  c_map=medium.sound_speed,
  rho_map=medium.density,
  export_name=work_dir / "wave_propagation_animation.mp4",
  vmax=p_max_plot,
  vmin=-p_max_plot,
)
```

## New simulation development instruction

- after the [installation](#install)
- make a directory below `experiments` such as `my_cool_wave_simulation`
- make a `.py` file or copy the example files below to use the boilerplate.
  - 2D plane wave
    - `examples/simple_plane_wave/simple_plane_wave_demo.py`
  - 3D plane wave
    - `examples/3d_wave/simple_plane_wave_3d.py`
- after that follow [Usage 2D](#usage-2d) to define the simulation code.

## Note for developers

- When developing something new, please create a new branch such as "YOURNAME/dev".
  - we use GitHub Flow for Git branching
    - ref: [Git branching strategies](https://www.geeksforgeeks.org/git/branching-strategies-in-git/)
- Please make a pull request if you want to add a new feature to the main branch.
- Please use the pre-commit tool to keep the code clean. Pre-commit is installed when you use the make command to install `fullwave-python`.
  ```sh
  pre-commit install
  ```
- [Ruff](https://docs.astral.sh/ruff/) will check your code and suggest improvements before you commit.
  - Sometimes, however, the fix is unnecessary and cumbersome. Let Masashi know if you want to remove some coding rules.
