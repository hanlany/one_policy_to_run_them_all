# Start from cuda image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git
RUN apt-get install -y python3-pip

# Install system dependencies (MPI and others you may need)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 libglib2.0-0 \
        vim \
        ca-certificates \
        libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/hanlany/RL-X.git
WORKDIR /app/RL-X
# RUN git checkout 70e220ac7eaf7b14ff4e0660227185a4628791b4
RUN python3 -m pip install pip -U
RUN pip3 install -e .[all]
# RUN pip3 install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu118 --upgrade
RUN pip3 install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
RUN pip3 uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
RUN pip3 install -U "jax[cuda12]"
# RUN pip3 install -U "jax[cuda12_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install URMA
WORKDIR /app
RUN git clone https://github.com/hanlany/one_policy_to_run_them_all.git
WORKDIR /app/one_policy_to_run_them_all
RUN pip install -e .[all] --upgrade-strategy only-if-needed

# Make shell script executable
RUN chmod +x ./experiments/experiment.sh
RUN chmod +x ./experiments/test.sh

CMD ["bash"]
