# This dockefile will setup a container with pytorch lightning and rapids
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=20.04

FROM nvcr.io/nvidia/rapidsai/rapidsai:23.02-cuda${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION}

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Prague \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

# Install dependencies
RUN \
    apt-get update && apt-get install -y wget && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
        openmpi-bin \
        llvm \
        ssh \
    && apt-get install -y python${PYTHON_VERSION} \
    && apt-get autoremove -y  \
    && apt-get clean  \
    && rm -rf /root/.cache \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./requirements/*.txt /requirements/

# Install requirements
RUN pip install --upgrade pip \
    && pip install -r /requirements/base.txt --no-cache-dir \
    && pip install -r /requirements/linting.txt --no-cache-dir

# Copy the rest of the code
COPY . /app

# Set the correct workdir
WORKDIR /app

# Install pre-commit hooks
RUN pre-commit install
