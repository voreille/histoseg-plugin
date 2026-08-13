# Histoseg Plugin

Histoseg Plugin is a Docker-first backend for whole-slide image (WSI) segmentation with:

* a FastAPI API service,
* a database-backed asynchronous queue,
* a worker process for model inference,
* a small web UI to inspect and control the queue,
* a QuPath extension to submit jobs and import results.

The current project is centered on API + worker orchestration and QuPath integration.

The repository provides a **CPU-only Docker configuration** intended for initial integration and deployment testing without requiring an NVIDIA GPU. GPU inference can be enabled separately for production or performance testing.

## Current Architecture

```mermaid
flowchart LR
    QP[QuPath extension] -->|POST /jobs| API[FastAPI service]
    WEB[Queue web UI] --> API
    API --> DB[(SQLite queue DB)]
    WORKER[Worker process] --> DB
    WORKER --> CORE[Segmentation pipeline]
    CORE --> OUT[Result files: GeoJSON + stats]
    API -->|GET /results/{id}| OUT
```

## What Is Included

* Asynchronous job submission and tracking (`/jobs`)
* Queue controls (`/queue` pause/resume)
* Result lookup and retrieval (`/results/lookup`, `/results/{id}`)
* Debug synchronous segmentation endpoints (`/segment/tissue`, `/segment/wsi`)
* Browser queue dashboard (`/web/queue`) with:

  * live status counters,
  * filtering/sorting/pagination,
  * cancel, retry, and delete actions.
* QuPath extension under `extensions/qupath` with commands to:

  * submit current slide,
  * submit all project slides,
  * check queue state,
  * import existing results.

## Repository Layout

```text
src/histoseg_plugin/
  api/        FastAPI app and routes
  core/       Segmentation + inference pipeline
  db/         SQLAlchemy models and DB setup
  jobs/       Queue operations, service, worker loop
  results/    Result registration and file IO
  web/        Jinja templates + static queue UI

extensions/qupath/
  QuPath plugin (Gradle project)

Dockerfile
  GPU development image

Dockerfile.cpu
  CPU-only image

docker-compose.yml
  Base Compose configuration

docker-compose.override.yml
  Local development volume mounts

docker-compose.cpu.yml
  CPU-specific Compose overrides

compose-cpu.sh
  Helper script to build and start the CPU configuration
```

## Prerequisites

For CPU-only deployment:

* Docker Engine
* Docker Compose plugin
* Access to slide files under the configured allowed roots
* Model weights available under the configured models directory

No NVIDIA GPU, CUDA runtime, or NVIDIA Container Toolkit is required for the CPU configuration.

The CPU image uses ONNX Runtime for CPU inference and includes the required OpenSlide runtime dependencies.

## Quick Start — CPU-only Docker

### 1. Configure host directories

The Compose configuration expects host directories for configuration, input data, model weights, results, and logs.

For example:

```bash
export UID="$(id -u)"
export GID="$(id -g)"

export HOST_CONFIG_DIR="$PWD/config"
export HOST_DATA_DIR="$PWD/data"
export HOST_RESULTS_DIR="$PWD/results"
export HOST_MODELS_DIR="$PWD/models"
export HOST_LOGS_DIR="$PWD/logs"
```

Create the directories if needed:

```bash
mkdir -p data results models logs
```

Place the model directory and weights under:

```text
models/
```

The actual model selected by the worker is configured through `default_model_id` in the application settings.

### 2. Build and start the CPU services

A helper script is provided:

```bash
./compose-cpu.sh
```

It starts the application using the base Compose configuration together with the CPU-specific overrides.

Equivalent command:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.cpu.yml \
  up --build
```

To run in detached mode:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.cpu.yml \
  up --build -d
```

### 3. Verify

Check the API:

```bash
curl http://localhost:8090/health
```

Follow worker logs:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.cpu.yml \
  logs -f worker
```

### Stop the CPU stack

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.cpu.yml \
  down
```

## CPU vs GPU Configuration

The project currently provides separate Docker images for CPU and GPU execution.

### CPU

The CPU configuration uses:

```text
Dockerfile.cpu
docker-compose.cpu.yml
```

and does not request NVIDIA devices.

This is the recommended configuration for initial integration and infrastructure testing.

### GPU

The standard development configuration uses:

```text
Dockerfile
docker-compose.yml
```

and can request NVIDIA GPU resources.

GPU execution requires the NVIDIA Container Toolkit and a compatible host GPU.

For local GPU development, the standard workflow remains:

```bash
docker compose up
```

or:

```bash
docker compose up --build
```

## Service Ports

* API: `http://localhost:8090`
* API docs: `http://localhost:8090/docs`
* Queue UI: `http://localhost:8090/web/queue`
* Debug ports:

  * API debugpy: host `5679` -> container `5678`
  * Worker debugpy: host `5678` -> container `5678`

## Configuration

Runtime settings are loaded from YAML using `HISTOSEG_CONFIG`.

In Docker:

```text
/app/config/settings-dev.yaml
```

Local fallback:

```text
config/settings.yaml
```

Important settings include:

* `database_url`: queue database path, SQLite by default
* `allowed_roots`: slide path allowlist for API requests
* `results_root`: output directory for job artifacts
* `models_root`: model directory for worker loading
* `default_model_id`: default model subdirectory name
* `preferred_device`: inference device (`cpu` for the CPU deployment)

For CPU deployment, configure:

```yaml
preferred_device: cpu
```

Model-specific runtime information is defined by the model manifest. For an ONNX model, for example:

```yaml
inference:
  runtime: onnx
  weights: model.onnx
  preferred_device: cpu
```

## Models

Models are expected under the configured `models_root`.

A model directory typically contains:

```text
models/
  <model_id>/
    manifest.yaml
    model.onnx
```

For CPU inference, ONNX models are executed using the CPU ONNX Runtime backend.

The model manifest defines, among other things:

* model runtime,
* weights file,
* expected tile size and resolution,
* input layout and datatype,
* output heads,
* labels,
* output geometry.

## API Overview

### Health

```http
GET /health
```

### Submit asynchronous jobs

```http
POST /jobs
Content-Type: application/json
```

Example request:

```json
{
  "items": [
    {
      "slide_uri": "file:///path/to/slide.svs",
      "model_id": "default"
    }
  ]
}
```

Response:

```json
{
  "job_id": 42,
  "status": "pending"
}
```

### Poll a job

```http
GET /jobs/{job_id}
```

### Queue controls

```http
GET  /queue
POST /queue/pause
POST /queue/resume
```

### Result lookup and retrieval

```http
POST /results/lookup
GET  /results/{result_id}
```

`/results/lookup` accepts a `JobItem` payload and resolves whether an equivalent task already exists using the same parameter hash.

### Debug synchronous routes

```http
POST /segment/tissue
POST /segment/wsi
```

These routes are intended for development and debugging workflows.

## Queue Web App

Open:

```text
http://localhost:8090/web/queue
```

Features:

* auto-refreshing queue summary and task table,
* status filters and sortable columns,
* pause/resume queue,
* task operations:

  * stop running,
  * cancel pending,
  * retry failed/cancelled/interrupted,
  * delete task and associated result if unreferenced.

## Result Storage

Each task writes files into:

```text
results/<task_hash>/
  predictions.geojson
  stats.json
  result_metadata.json
```

The database stores metadata and file paths.

`/results/{result_id}` returns the GeoJSON payload together with the associated statistics.

## QuPath Integration

The QuPath plugin lives in:

```text
extensions/qupath
```

### Build extension

```bash
cd extensions/qupath
./gradlew build
```

Output JAR files are created in:

```text
extensions/qupath/build/libs
```

### Install in QuPath

Drag the built JAR into QuPath to install it.

### Configure plugin

In QuPath:

* Open `Extensions > HistoSeg > Settings...`
* Set:

  * Server URL, default: `http://localhost:8090`
  * Model ID, default: `default`

### Available commands

* `Submit current slide...`
* `Submit all project slides...`
* `Import existing result for current slide...`
* `Queue status...`
* `Settings...`

## Local Development Without Docker

Docker is the reference workflow, but local execution is possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export HISTOSEG_CONFIG="config/settings.yaml"

# API
uvicorn histoseg_plugin.api.main:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8000

# Worker, in a separate shell
python -m histoseg_plugin.jobs.worker_main
```

When using local mode, make sure:

* OpenSlide system libraries are installed,
* required Python runtime dependencies are installed,
* configured `allowed_roots` are valid on the machine.

## Testing

```bash
pytest
```

## Troubleshooting

### `403 Slide path not under allowed roots`

Add the slide parent directory to `allowed_roots` in the settings YAML.

### Queue appears stuck

Check worker logs:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.cpu.yml \
  logs -f worker
```

Also confirm that the queue is not paused in:

```text
http://localhost:8090/web/queue
```

### Permission issues writing results or database files

Ensure the exported `UID` and `GID` match the host user before starting the containers:

```bash
export UID="$(id -u)"
export GID="$(id -g)"
```

Also verify that the configured host directories are writable.

### Model cannot be found

Verify that:

* the model directory is mounted under `models_root`,
* `default_model_id` matches the model directory name,
* the model manifest references the correct weights filename.

### ONNX Runtime errors

For the CPU image, the model manifest should specify:

```yaml
inference:
  runtime: onnx
  preferred_device: cpu
```

The CPU image uses the `onnxruntime` package and does not require CUDA.

### `slide_uri` validation errors

Use absolute paths or `file://` URIs pointing to files that exist and are readable from inside the container.

Remember that the path visible to the container may differ from the corresponding host path depending on the configured volume mounts.

## Notes

* Coordinates and GeoJSON outputs are in level-0 slide space.
* CPU inference is primarily intended for integration and functional testing and may be significantly slower than GPU inference on large WSIs.
* The GPU deployment can be enabled later without changing the API or queue architecture.
* The old tiling-focused README and roadmap are obsolete for the current architecture.
