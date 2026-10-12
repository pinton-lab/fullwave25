# fullwave25 in a container

The released solver binaries need glibc 2.34 or newer (for example Ubuntu 22.04, Debian 12 or RHEL 9). On an older system, such as Ubuntu 20.04 or RHEL 8, they stop at start with an error like:

```text
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
```

This image runs fullwave25 and the released solver binaries on such a system. The host needs only an NVIDIA driver for CUDA 12.0 or newer, and Docker with the NVIDIA Container Toolkit, or Apptainer.

The image contains:

- Ubuntu 22.04
- Python 3.12 and fullwave25, installed from this repository checkout, with the CUDA 12 libraries that CuPy needs
- the solver binaries of the release `fullwave_bin_v1.5.2`, for CUDA 11.8, 12.4, 12.9 and 13.0
- `ffmpeg`, for the video export of the plotting functions

## Build

Run from the repository root:

```sh
docker build -f docker/Dockerfile -t fullwave25 .
```

To use another release of the solver binaries:

```sh
docker build -f docker/Dockerfile -t fullwave25 --build-arg FULLWAVE_BIN_TAG=<release tag> .
```

## Run with Docker

Mount your working directory at `/work` and run your script there. The outputs are written to your working directory, owned by your user.

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD":/work fullwave25 python my_simulation.py
```

If `--gpus all` is not available, use `--device nvidia.com/gpu=all` (CDI).

To use selected GPUs, pass them to fullwave25 as usual with `cuda_device_id` in `fullwave.Solver`.

## Run with Apptainer

Apptainer runs the same image without Docker. Build the image on a machine with Docker, and save it to a file:

```sh
docker save fullwave25 -o fullwave25.tar
```

Copy `fullwave25.tar` to the target machine, and convert it there:

```sh
apptainer build fullwave25.sif docker-archive://fullwave25.tar
```

The saved image is about 4.3 GB and the `.sif` file about 2.3 GB, and the conversion needs temporary space of several GB in addition. If `/tmp` is small, set `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` to a folder on a larger disk.

Then run your script. `--nv` gives the container the GPUs and the driver of the host. Apptainer mounts your home and the current directory by default.

```sh
apptainer exec --nv fullwave25.sif python my_simulation.py
```

If Apptainer prints `Error changing the container working directory`, your working directory is reached through a link to another disk. Run from its real path:

```sh
cd "$(pwd -P)"
```

## Check the installation

```sh
apptainer exec --nv fullwave25.sif python -c "import fullwave; print(fullwave.__version__)"
docker run --rm --gpus all fullwave25 nvidia-smi -L
```
