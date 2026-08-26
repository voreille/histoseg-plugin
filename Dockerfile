FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libopenslide0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml /app/
COPY src/ /app/src/
COPY config/ /app/config/

RUN uv pip install \
      torch==2.2.2 \
      torchvision==0.17.2 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install debugpy && \
    uv pip install -e /app

CMD ["uvicorn", "histoseg_plugin.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
