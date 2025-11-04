# Start from a CUDA-enabled base image with Python
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

# Avoid running as root user for security
RUN groupadd -r fullwave && useradd -m -g fullwave fullwave

# Set working directory
WORKDIR /app

# Install system dependencies (build essentials, CUDA dev libs, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential make curl ca-certificates \
    libgl1-mesa-dev ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*


# ---- install uv package manager ----
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV PATH="/app/.venv/bin:$PATH"

# ----
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./
COPY uv.lock ./
COPY Makefile ./

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --all-extras

# Install Python dependencies efficiently
RUN uv run pre-commit install

# Set permissions for non-root user
RUN chown -R fullwave:fullwave /app

# Switch to user, preserve CUDA access
USER fullwave

# Specify default runtime arguments for simulation (override as needed)
CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && pytest"]
