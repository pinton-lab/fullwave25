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

## Examples

Start from [example_simple_plane_wave.ipynb](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/example_simple_plane_wave.ipynb).

After that, please see the following examples for more advanced usage.

- 2D plane wave
  - Basic usage
    - [Simple plane wave](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/simple_plane_wave.py)
      - ![alt text](figs/simple_plane_wave.gif)
    - [Simple plane wave with air](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/simple_plane_wave.py)
      - ![alt text](figs/simple_plane_wave_with_air.gif)
  - Linear transducer
    - [Linear transducer](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer.py)
    - [Linear transducer with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/linear_transducer/linear_transducer.py)
  - Convex transducer
    - [Convex transducer with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/convex_transducer/convex_transducer_abdominal_wall.py)
- 3D plane wave
  - Basic usage
    - [Simple plane wave in 3D](https://github.com/pinton-lab/fullwave25/blob/main/examples/wave_3d/simple_plane_wave_3d.py)
- Medium builder usage
  - [simple medium builder usage](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_example.py)
  - [simple medium builder usage with abdominal wall](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_abdominal_example.py)
  - [medium builder in 3D](https://github.com/pinton-lab/fullwave25/blob/main/examples/medium_builder/medium_builder_example_3d.py)

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
- Before 3D simulation...
  - If you want to run a 3D simulation, it is recommended that you start with a 2D simulation first to understand the basic usage.
  - The 3D simulation code is similar to the 2D code, but some plot functions are unavailable in 3D.
  - The 3D simulation takes longer to run, so starting with 2D will help you debug your code faster.

## Usage 2D

Here are the main steps to run the Fullwave simulation

1. Define the computational grid.
2. Define the properties of the acoustic medium.
3. Define the acoustic source.
4. Define the sensor.
5. Execute the simulation.

Please check [example_simple_plane_wave.ipynb](https://github.com/pinton-lab/fullwave25/blob/main/examples/simple_plane_wave/example_simple_plane_wave.ipynb) for the detailed code example.

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
