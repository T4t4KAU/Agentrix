# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG MAX_JOBS=2
ARG NVCC_THREADS=1
ARG TORCH_CUDA_ARCH_LIST=12.0

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_HTTP_TIMEOUT=500 \
    UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache \
    UV_LINK_MODE=copy \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev0+pat \
    PATH=/opt/uv/bin:/root/.cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        g++-11 \
        gcc-11 \
        git \
        libibverbs-dev \
        libnuma-dev \
        make \
        pkg-config \
        sudo \
        unzip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${UV_INSTALL_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version

WORKDIR /workspace/agentrix

COPY vllm/ vllm/

WORKDIR /workspace/agentrix/vllm

RUN uv venv --python "${PYTHON_VERSION}" --seed .venv \
    && uv pip install --python .venv/bin/python \
        -r requirements/build/cuda.txt \
        --torch-backend=auto

RUN tools/install_protoc.sh \
    && PATH="${PWD}/.venv/bin:${PATH}" ./build_rust.sh

RUN uv pip install --python .venv/bin/python \
    --no-build-isolation \
    -e .

WORKDIR /workspace/agentrix

COPY benchmark/ benchmark/

WORKDIR /workspace/agentrix/benchmark

RUN uv venv --python "${PYTHON_VERSION}" --seed .venv \
    && uv pip install --python .venv/bin/python -e ".[data,test]"

ENV PATH=/workspace/agentrix/benchmark/.venv/bin:/workspace/agentrix/vllm/.venv/bin:${PATH}

WORKDIR /workspace/agentrix

CMD ["/bin/bash"]
