FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libopenslide0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY histoseg-plugin /app
COPY pathseg-benchmark /opt/pathseg-benchmark

RUN python -m pip install --upgrade pip && \
    pip install torch==2.2.2 torchvision==0.17.2 \
      --extra-index-url https://download.pytorch.org/whl/cu123 && \
    pip install debugpy && \
    pip install -e /opt/pathseg-benchmark && \
    pip install -e /app

CMD ["bash"]