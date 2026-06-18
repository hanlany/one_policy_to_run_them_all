FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libgles2 \
        libglvnd0 \
        libopenmpi-dev \
        mesa-utils \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        vim \
        tmux && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m pip install --upgrade pip uv

WORKDIR /app

# Install Lava-DL for the Sigma-Delta conversion flow.
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/bamsumit/lava-dl.git
WORKDIR /app/lava-dl
RUN git checkout ann_sdnn && \
    uv pip install -e . --system

# Install RL-X.
WORKDIR /app
RUN git clone -b urma_snn https://github.com/hanlany/RL-X.git
WORKDIR /app/RL-X
RUN uv pip install -e .[all] --system

# Install the CUDA-enabled PyTorch stack first so later editable installs can
# reuse it instead of trying to resolve a mismatched CPU build from PyPI.
RUN pip install --upgrade \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --upgrade \
    nvidia-cublas-cu12==12.4.2.65 \
    nvidia-cuda-cupti-cu12==12.4.99 \
    nvidia-cuda-nvcc-cu12==12.4.99 \
    nvidia-cuda-nvrtc-cu12==12.4.99 \
    nvidia-cuda-runtime-cu12==12.4.99 \
    nvidia-cudnn-cu12==8.9.7.29 \
    nvidia-cufft-cu12==11.2.0.44 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-cusolver-cu12==11.6.0.99 \
    nvidia-cusparse-cu12==12.3.0.142 \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvjitlink-cu12==12.4.99 \
    nvidia-nvtx-cu12==12.4.99

RUN pip install -U \
    "jax[cuda12_pip]==0.4.25" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install one_policy_to_run_them_all.
WORKDIR /app
RUN git clone https://github.com/hanlany/one_policy_to_run_them_all.git
WORKDIR /app/one_policy_to_run_them_all
RUN grep -Ev '^(torch==|jax==|jaxlib==|nvidia-|triton==)' docker/requirements.txt > /tmp/urma-requirements.txt && \
    pip install -r /tmp/urma-requirements.txt && \
    uv pip install -e . --system --no-deps

# Add the extra runtime bits used by genloco-loihi without downgrading the
# newer gym stack required by one_policy_to_run_them_all.
RUN mkdir -p /genloco-loihi && \
    pip install \
        absl-py \
        attrs \
        imageio \
        imageio[ffmpeg] \
        pybullet

WORKDIR /app

CMD ["bash"]
