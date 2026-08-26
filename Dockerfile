# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON=python3.10 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.10 \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 \
    /uv /usr/local/bin/uv

WORKDIR /app

# Install locked third-party runtime dependencies.
COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-default-groups \
        --extra cu121 \
        --no-install-project

COPY src/ /app/src/
COPY config/ /app/config/


# Development image: application + debug and test dependencies.
FROM base AS development

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --extra cu121

CMD ["uvicorn", "histoseg_plugin.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


# Production image: application and runtime dependencies only.
# Kept as the final stage so it is the default build target.
FROM base AS production

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-default-groups \
        --extra cu121

CMD ["uvicorn", "histoseg_plugin.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]