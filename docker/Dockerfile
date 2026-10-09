# fullwave25 with the released solver binaries, for Linux systems whose glibc is older than
# the binaries need (2.34, for example Ubuntu 20.04 or RHEL 8).
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t fullwave25 .
#
# See docker/README.md for how to run it with Docker or Apptainer.

FROM ubuntu:22.04

# The release tag of the solver binaries, and the CUDA versions to include. The wrapper
# picks the binary that matches the CUDA version of the host driver.
ARG FULLWAVE_BIN_TAG=fullwave_bin_v1.5.2
ARG FULLWAVE_BIN_CUDA_VERSIONS="118 124 129 130"
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}
RUN uv venv --python ${PYTHON_VERSION} /opt/venv

COPY . /opt/fullwave25-src
# The [ctk] extra installs the CUDA libraries and headers that CuPy needs at run time.
RUN uv pip install --no-cache /opt/fullwave25-src \
    && uv pip install --no-cache "cupy-cuda12x[ctk]==$(python -c "import importlib.metadata as m; print(m.version('cupy-cuda12x'))")"

# Put the solver binaries where the installed package looks first, so a run needs no
# download and works when the home directory is mounted from the host (Apptainer).
RUN set -eu; \
    bins="$(python -c "import importlib.util; print(importlib.util.find_spec('fullwave').submodule_search_locations[0])")/solver/bins"; \
    url="https://github.com/pinton-lab/fullwave25/releases/download/${FULLWAVE_BIN_TAG}"; \
    for cuda in ${FULLWAVE_BIN_CUDA_VERSIONS}; do \
        for dim in 2d 3d; do \
            mkdir -p "${bins}/gpu/${dim}" "${bins}/exponential_attenuation/gpu/${dim}"; \
            curl -fsSL -o "${bins}/gpu/${dim}/fullwave2_${dim}_n_relax_multi_gpu_cuda${cuda}" \
                "${url}/fullwave2_${dim}_n_relax_multi_gpu_cuda${cuda}"; \
            curl -fsSL -o "${bins}/exponential_attenuation/gpu/${dim}/fullwave2_${dim}_exponential_attenuation_multi_gpu_cuda${cuda}" \
                "${url}/fullwave2_${dim}_exponential_attenuation_multi_gpu_cuda${cuda}"; \
        done; \
    done; \
    chmod 755 "${bins}"/gpu/*/* "${bins}"/exponential_attenuation/gpu/*/*; \
    echo "${FULLWAVE_BIN_TAG}" > "${bins}/RELEASE_TAG"

# A container started with --user has no home of its own, so the caches of CuPy,
# matplotlib and numba go to /tmp. Apptainer sets HOME to the host home instead.
ENV HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /work
CMD ["python"]
