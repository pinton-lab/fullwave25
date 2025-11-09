# Fullwave 2.5: Ultrasound wave propagation simulation with heterogeneous power law attenuation modelling capabilities

<a href="https://doi.org/10.5281/zenodo.17497689"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17497689.svg" alt="DOI"></a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/153Sx9D_5zTlF7UtlKHlyz_CTJnLIAI6s)

Fullwave 2.5 is a Python package for high-fidelity ultrasound wave propagation simulation with the following features:

- State-of-the-art attenuation modelling capabilities for ultrasound wave propagation in complex biological tissues
  - Heterogeneous power law attenuation ($\alpha=\alpha_0 f^\gamma$) modeling, where **both the attenuation coefficient $\alpha_0$ and exponent $\gamma$ can vary** spatially.
- High-performance simulation engine
  - High accuracy staggered-grid finite-difference time-domain (FDTD) scheme (8th-order in space and 4th-order in time).
  - 2D and 3D ultrasound wave propagation simulation.
  - Multiple GPU execution support.
- Easy-to-use Python interface with CUDA/C backend
  - Python wrapper for easy usability and extensibility, with the core simulation engine implemented in CUDA/C for high performance on NVIDIA GPUs.
  - It offers a user experience similar to [k-Wave](http://www.k-wave.org/) and [k-wave-python](https://github.com/waltsims/k-wave-python), while providing advanced attenuation modeling capabilities and multi-GPU support in FDTD simulations.

| Computational medium                     | Wave propagation                                                           |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| <img src="figs/medium.png" width="600"/> | <img src="figs/linear_transducer_focused_abdominal_wall.gif" width="200"/> |

Building upon the original Fullwave 2 simulator, Fullwave 2.5 enhances its capabilities to model ultrasound wave propagation in media where **both the attenuation coefficient and exponent can vary spatially**. This enables more accurate simulations of biological tissues, which often exhibit complex attenuation behaviours that cannot be captured by uniform exponent models.

The library is designed with a Python wrapper for ease of use and extensibility, while the core simulation engine is implemented in CUDA/C to leverage high-performance computing on NVIDIA GPUs. Fullwave 2.5 supports 2D and 3D simulations, including multi-GPU execution for enhanced performance.

## Special Thanks

This repository design was inspired by [k-wave-python](https://github.com/waltsims/k-wave-python). We appreciate the great work of the k-wave-python development team.
This repository would not have been possible without them.

Please check their repository for additional ultrasound simulation tools and resources.
Their comprehensive tools have significantly contributed to the ultrasound research community.

## Theoretical Background

Fullwave 2.5 models multiple relaxation processes to approximate frequency-dependent power-law attenuation in heterogeneous media.
It solves the stretched-coordinate pressure-velocity formulation using a staggered-grid finite-difference schemes with 8th-order accuracy in space and 4th-order accuracy in time.
The formulation is expressed as follows:

$$\nabla_1 p + \rho \cfrac{\partial {\bf{v}}}{\partial t} = 0$$
$$\nabla_2 \cdot {\bf{v}} + \kappa \cfrac{\partial p}{\partial t} = 0$$

The stretched-coordinate derivatives, denoted by $\nabla_1$ and $\nabla_2$, control frequency-dependent power-law attenuation and dispersion by selecting the optimal relaxation parameters.

The following figure illustrates the performance of the attenuation modeling in Fullwave 2.5.
The graph shows a comparison of the target power-law attenuation $\alpha=\alpha_0 f^\gamma$ (red line) and the simulated attenuation (black dots) for various spatially varying attenuation coefficients ($\alpha_0 =$ 0.25, 0.5, and 0.75) and exponents ($\gamma =$ 0.4, 0.7, 1.0, 1.3, and 1.6).

![attenuation modeling performance](./figs/attenuation_modeling.svg)

## Citation

Fullwave 2.5 is developed and maintained by [Pinton Lab](https://github.com/pinton-lab) at the University of North Carolina at Chapel Hill.

If you use Fullwave 2.5 in your research, please cite this repository as:

```bibtex
@software{Sode2025-fullwave25,
  author = {Sode, Masashi and Pinton, Gianmarco},
  title = {{Fullwave 2.5: Ultrasound wave propagation simulation with heterogeneous power law attenuation modelling capabilities}},
  year = {2025},
  month = oct,
  doi = {10.5281/zenodo.17497689},
  url = {https://github.com/pinton-lab/fullwave25},
}

@ARTICLE{Pinton2021-fullwave2,
  title = "A fullwave model of the nonlinear wave equation with multiple relaxations and relaxing perfectly matched layers for high-order numerical finite-difference solutions",
  author = "Pinton, Gianmarco",
  month = jun,
  year = 2021,
  copyright = "http://creativecommons.org/licenses/by/4.0/",
  archivePrefix = "arXiv",
  primaryClass = "physics.med-ph",
  eprint = "2106.11476"
}
```

---

## Hardware prerequisites

- This system operates in a Linux environment.
  - If you need a Windows environment, please consider using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux 2).
- This simulation requires an NVIDIA GPU to execute.
- You may need multiple GPUs for 3D simulation.

## Technical recommendations

- We recommend setting up an SSH key for GitHub, if you haven't done already. The repository changes over time to fix bugs and add new features. You can keep your local repository up to date by pulling the latest changes from GitHub. Cloning through SSH is more convenient than HTTPS in the long run.
  - for ssh key generation
    - please see: [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
  - for ssh key registration to your github account
    - please see: [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- after that, you can clone the repository through

  ```sh
  git clone git@github.com:pinton-lab/fullwave-python.git
  ```

## Technical references

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

---

## installation for users

```sh
pip install fullwave25
```

## installation for development

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

---

## Tutorial: Basic Usage

Please start from [example_simple_plane_wave.ipynb](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/example_simple_plane_wave.ipynb).

or try the Google Colab tutorial.
You don't need to install or set up a GPU environment on your local machine in order to run the simulation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/153Sx9D_5zTlF7UtlKHlyz_CTJnLIAI6s)

Here are the main steps to run the Fullwave simulation

1. Define the computational grid.
2. Define the properties of the acoustic medium.
3. Define the acoustic source.
4. Define the sensor.
5. Execute the simulation.

## New simulation development instruction

- after the [installation](#installation)
- make a directory for your simulation under your favorite path.
  - e.g. `examples/my_simulation/`
- make a `.py` file or copy the example files below to use the boilerplate.
  - 2D plane wave
    - [examples/simple_plane_wave/simple_plane_wave_demo.py](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/simple_plane_wave.py)
  - 3D plane wave
    - [examples/wave_3d/simple_plane_wave_3d.py](https://github.com/pinton-lab/fullwave25/blob/main/examples/wave_3d/simple_plane_wave_3d.py)
- after that follow [Usage 2D](#tutorial-basic-usage) to define the simulation code.

## Tutorial: Advanced Usages

Please see the following examples for more advanced usage.

- 2D plane wave
  - Basic usage
    - [Simple plane wave](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/simple_plane_wave.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/153Sx9D_5zTlF7UtlKHlyz_CTJnLIAI6s)
      - <img src="figs/simple_plane_wave.gif" width="200"/>
    - [Simple plane wave with air](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/simple_plane_wave.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AnTaT6ZtwyIEcOvGgpn5srB4LWPfYKh0)
      <!-- - <img src="figs/simple_plane_wave_with_air.gif" width="200"/> -->
  - Linear transducer
    - [Linear transducer](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer.py)
    - [Linear transducer (plane wave transmit) with animation settings](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer_animation.py)
      <!-- - <img src="figs/linear_transducer.gif" width="200"/> -->
    - [Linear transducer (focused transmit) with animation settings](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer_animation.py)
      - <img src="figs/linear_transducer_focused.gif" width="200"/>
    - [Linear transducer (focused transmit) with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zlviYe0qBy0JLifFuA2MqUkavJKUQrNb)
      - <img src="figs/linear_transducer_focused_abdominal_wall.gif" width="200"/>
  - Convex transducer
    - [Convex transducer with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/convex_transducer/convex_transducer_abdominal_wall.py)
      - <img src="figs/convex_transducer_abdominal_wall.gif" width="200"/>
- 3D plane wave

  - Basic usage

    - [Simple plane wave in 3D](https://github.com/pinton-lab/fullwave25/blob/main/examples/wave_3d/simple_plane_wave_3d.py)
      <table>
        <tr>
          <td style="text-align: center;" colspan="2">Computational medium</td>
        </tr>
        <tr>
          <td style="text-align: center;" colspan="2"><img src="figs/medium_3d.png" width="600"/></td>
        </tr>
        <tr>
          <td style="text-align: center;" >x-y slice propagation</td>
          <td style="text-align: center;" >x-z slice propagation</td>
        </tr>
        <tr>
          <td style="text-align: center;" ><img src="figs/wave_propagation_x-y.gif" width="300"/></td>
          <td style="text-align: center;" ><img src="figs/wave_propagation_x-z.gif" width="300"/></td>
        </tr>
      </table>
    - [Simple plane wave in 3D with air inclusion](https://github.com/pinton-lab/fullwave25/blob/main/examples/wave_3d/simple_plane_wave_3d_with_air.py)
      <table>
        <tr>
          <td style="text-align: center;"  colspan="2">Computational medium with air inclusion</td>
        </tr>
        <tr>
          <td style="text-align: center;"  colspan="2"><img src="figs/medium_3d_air.png" width="600"/></td>
        </tr>
        <tr>
          <td style="text-align: center;" >x-y slice propagation</td>
          <td style="text-align: center;" >x-z slice propagation</td>
        </tr>
        <tr>
          <td style="text-align: center;" ><img src="figs/wave_propagation_x-y_air.gif" width="300"/></td>
          <td style="text-align: center;" ><img src="figs/wave_propagation_x-z_air.gif" width="300"/></td>
        </tr>
      </table>

- Medium builder usage
  - Medium builder is a utility to create computational medium from simple geometric operations. This is especially useful when you want to create complex heterogeneous media.
  - [simple medium builder usage](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_example.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/169aHbXRt_-aXFEdf2pfPkwH4-niJvrfL)
  - [simple medium builder usage with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_abdominal_example.py)
  - [medium builder in 3D](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_example_3d.py)

---

## Attention

- The simulation grid is defined as follows:
  - (x, y, z) = (depth, lateral, elevational).
    - This order is due to the efficiency of the multiple-GPU execution.
    - Multi-GPU domain decomposition is processed in the depth dimension.
  - The index of the input coordinates (i.e. the acoustic source location) is defined in C-array order (i.e. row-major) within the simulation, regardless of your setup. This is to improve the efficiency of multi-GPU development.
  - This might be confusing, so please be careful when you define the source and source signal definition.
- GPU memory requirement
  - A 3D simulation requires a lot of GPU memory.
    - Please reduce the grid size or use multiple GPUs if you run out of memory.
    - You can check GPU memory usage with the 'nvidia-smi' or 'nvtop' commands.
- Multi-GPU execution
  - The current implementation supports multiple GPU execution in 2D and 3D simulations.
  - Our implementation demonstrates linear performance scaling with the number of GPUs.
- Before 3D simulation:
  - If you want to run a 3D simulation, it is recommended that you start with a 2D simulation first to understand the basic usage.
  - The 3D simulation code is similar to the 2D code, but some plot functions are unavailable in 3D.
  - The 3D simulation takes longer to run, so starting with 2D will help you debug your code faster.

## Note for developers

- Contributions are welcome!
- When developing something new, please create a new branch such as `TYPE/BRANCH_NAME`.
  - TYPE can be `feature`, `bugfix`, `hotfix`, `docs`, `refactor`, `release`, `test`, or `experiment`.
  - `BRANCH_NAME` should be descriptive of the feature or fix you are working on.
  - see also: [GitHub Branching Name Best Practices](https://dev.to/jps27cse/github-branching-name-best-practices-49ei)
- Please write clear and concise commit messages.
- please see [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## Authors

- Masashi Sode (GitHub: [MasashiSode](https://github.com/MasashiSode))
- Gianmarco Pinton (GitHub: [gfpinton](https://github.com/gfpinton))

## Maintainers

- Masashi Sode (GitHub: [MasashiSode](https://github.com/MasashiSode))
