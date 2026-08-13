# Histoseg Plugin

Histoseg Plugin is a Docker-first backend for whole-slide image (WSI) segmentation with:

- a FastAPI API service,
- a database-backed asynchronous queue,
- a worker process for GPU inference,
- a small web UI to inspect and control the queue,
- a QuPath extension to submit jobs and import results.

The current project is centered on API + worker orchestration and QuPath integration.

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

- Asynchronous job submission and tracking (`/jobs`)
- Queue controls (`/queue` pause/resume)
- Result lookup and retrieval (`/results/lookup`, `/results/{id}`)
- Debug synchronous segmentation endpoints (`/segment/tissue`, `/segment/wsi`)
- Browser queue dashboard (`/web/queue`) with:
  - live status counters,
  - filtering/sorting/pagination,
  - cancel, retry, and delete actions.
- QuPath extension under `extensions/qupath` with commands to:
  - submit current slide,
  - submit all project slides,
  - check queue state,
  - import existing results.

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
```

## Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA Container Toolkit (if running on GPU)
- Access to slide files under configured allowed roots

The default Docker image is CUDA-based and includes OpenSlide runtime dependencies.

## Quick Start (Docker)

The compose files expect host path environment variables.

1. Export host variables

```bash
export UID="$(id -u)"
export GID="$(id -g)"

# Workspace parent folder (contains histoseg-plugin/)
export HOST_WORKSPACE_DIR="/home/valentin/workspaces"

# Host directories to mount
export HOST_CONFIG_DIR="/home/valentin/workspaces/histoseg-plugin/config"
export HOST_DATA_DIR="/home/valentin/workspaces/histoseg-plugin/data"
export HOST_RESULTS_DIR="/home/valentin/workspaces/histoseg-plugin/results"
export HOST_MODELS_DIR="/home/valentin/workspaces/histoseg-plugin/models"
export HOST_LOGS_DIR="/home/valentin/workspaces/histoseg-plugin/logs"
```

2. Build and run services

```bash
docker compose build api
docker compose up -d api worker
```

3. Verify

```bash
curl http://localhost:8090/health
docker compose logs -f worker
```

### Service Ports

- API: `http://localhost:8090`
- API docs: `http://localhost:8090/docs`
- Queue UI: `http://localhost:8090/web/queue`
- Debug ports:
  - API debugpy: host `5679` -> container `5678`
  - Worker debugpy: host `5678` -> container `5678`

## Configuration

Runtime settings are loaded from YAML using `HISTOSEG_CONFIG`.

- In Docker, default is `/app/config/settings-dev.yaml`
- Local fallback is `config/settings.yaml`

Important settings:

- `database_url`: queue database path (SQLite by default)
- `allowed_roots`: slide path allowlist for API requests
- `results_root`: output directory for job artifacts
- `models_root`: model directory for worker loading
- `default_model_id`: default model subdirectory name
- `preferred_device`: `cuda` or `cpu`

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
      "slide_uri": "file:///mnt/nas7/path/to/slide.svs",
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

`/results/lookup` accepts a `JobItem` payload and resolves whether an equivalent task already exists (same parameter hash).

### Debug synchronous routes

```http
POST /segment/tissue
POST /segment/wsi
```

These are intended for development/debug workflows.

## Queue Web App

Open `http://localhost:8090/web/queue`.

Features:

- auto-refreshing queue summary and task table,
- status filters and sortable columns,
- pause/resume queue,
- task operations:
  - stop running,
  - cancel pending,
  - retry failed/cancelled/interrupted,
  - delete task (and associated result if unreferenced).

## Result Storage

Each task writes files into:

```text
results/<task_hash>/
  predictions.geojson
  stats.json
  result_metadata.json
```

The DB stores metadata and paths; `/results/{result_id}` returns GeoJSON payload and attached statistics.

## QuPath Integration

The QuPath plugin lives in `extensions/qupath`.

### Build extension

```bash
cd extensions/qupath
./gradlew build
```

Output JAR files are created in `extensions/qupath/build/libs`.

### Install in QuPath

Drag the built JAR into QuPath to install it.

### Configure plugin

In QuPath:

- Open `Extensions > HistoSeg > Settings...`
- Set:
  - Server URL (default: `http://localhost:8090`)
  - Model ID (default: `default`)

### Available commands

- `Submit current slide...`
- `Submit all project slides...`
- `Import existing result for current slide...`
- `Queue status...`
- `Settings...`

## Local Development (Without Docker)

Docker is the reference workflow, but local execution is possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export HISTOSEG_CONFIG="config/settings.yaml"

# API
uvicorn histoseg_plugin.api.main:create_app --factory --host 0.0.0.0 --port 8000

# Worker (separate shell)
python -m histoseg_plugin.jobs.worker_main
```

If using local mode, make sure OpenSlide system libs are installed and the configured `allowed_roots` are valid on your machine.

## Testing

```bash
pytest
```

## Troubleshooting

- `403 Slide path not under allowed roots`
  - Add the slide parent directory to `allowed_roots` in your settings YAML.

- Queue appears stuck
  - Check worker logs: `docker compose logs -f worker`
  - Confirm queue is not paused in `/web/queue`.

- Permission issues writing `results/` or DB files
  - Ensure `UID`/`GID` exports match your host user before `docker compose up`.

- No GPU inside container
  - Verify NVIDIA Container Toolkit setup and Docker GPU runtime availability.

- `slide_uri` validation errors
  - Use absolute paths or `file://` URIs to files that exist and are readable.

## Notes

- Coordinates and GeoJSON outputs are in level-0 slide space.
- The old tiling-focused README and roadmap are obsolete for the current architecture.