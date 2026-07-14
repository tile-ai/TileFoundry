FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System toolchain: Python 3.12 (deadsnakes), build tools, clang-format-18 from LLVM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        gnupg \
        wget \
        git \
        ninja-build \
        g++ \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-format-18 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-18 100 \
    && rm -rf /var/lib/apt/lists/*

# Make python/pip resolve to 3.12.
RUN wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.12 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip

# cmake >=3.18 via pip (apt cmake on 22.04 is too old); ninja-build/g++ from apt above.
RUN pip install --no-cache-dir "cmake>=3.18"

# torch 2.9.1 (cu128): the runtime maps DType.f8e8m0 -> torch.float8_e8m0fnu,
# which torch 2.5 lacks. cu128 bundles its own CUDA runtime; the devel base's
# nvcc 12.9 is what JIT-builds the kernels, so the two coexist.
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# tilefoundry runtime deps (mirrors pyproject.toml [project.dependencies]) plus CI tooling.
# The tilefoundry package itself is NOT installed here; editable install happens at CI runtime.
RUN pip install --no-cache-dir \
        "isl-python>=0.1.8" \
        "jinja2>=3.0" \
        apache-tvm-ffi \
        "graphviz>=0.20" \
        "transformers>=4.57" \
        "pytest>=8" \
        "ruff==0.14.13" \
        "pre-commit>=3"

# isl-python bundles a newer libisl under /usr/local/lib; put it ahead of the
# older apt libisl23 so `import isl` finds the symbols it needs (mirrors how the
# conda dev env resolves it). Prepend to keep the CUDA lib paths from the base.
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# cutlass headers only, pinned to the submodule SHA (passed as a build-arg).
ARG CUTLASS_SHA
RUN git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git /tmp/cutlass && cd /tmp/cutlass && git sparse-checkout init --cone && git sparse-checkout set include && git checkout ${CUTLASS_SHA} && mkdir -p /opt/cutlass && mv /tmp/cutlass/include /opt/cutlass/include && rm -rf /tmp/cutlass
