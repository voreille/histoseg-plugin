# Histoseg Plugin

Docker-first backend for whole-slide image (WSI) segmentation with:

* a FastAPI API service,
* a PostgreSQL-backed asynchronous queue,
* a worker process for model inference,
* a browser queue dashboard,
* a QuPath extension to submit jobs and import results.

## Architecture

```mermaid
flowchart LR
    QP[QuPath extension] -->|POST /jobs| API[FastAPI service]
    WEB[Queue web UI] --> API
    API --> DB[(PostgreSQL queue DB)]
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
* Browser queue dashboard (`/web/queue`) with live counters, filtering, cancel/retry/delete
* QuPath extension under `extensions/qupath`

## Repository Layout

```text
src/histoseg_plugin/
  api/        FastAPI app and routes
  core/       Segmentation + inference pipeline
  db/         SQLAlchemy models, migrations, DB setup
  jobs/       Queue operations, service, worker loop
  results/    Result registration and file IO
  web/        Jinja templates + static queue UI

extensions/qupath/     QuPath plugin (Gradle project)

Dockerfile             GPU image (CUDA 12.3, uv)
Dockerfile.cpu         CPU-only image (uv)

docker-compose.yaml          Base: shared services + PostgreSQL
docker-compose.gpu.yaml      GPU overlay: nvidia resources + GPU image
docker-compose.cpu.yaml      CPU overlay: clears GPU resources + CPU image
docker-compose.override.yaml Dev overlay: source mounts, debug ports (gitignored)
docker-compose.sqlite.yaml   Optional: switch queue DB to SQLite
docker-compose.podman.yaml   Podman rootless compatibility (userns_mode: keep-id)

compose.sh             Unified build/run entry point

config/
  settings.yaml        Base settings (used when baked into image)
  settings-dev.yaml    Dev settings (debug, NAS roots, etc.)

scripts/
  migrate_sqlite_to_pg.py  One-shot SQLite → PostgreSQL data migration
  pg_backup.sh             Dump PostgreSQL DB to a timestamped SQL file
  pg_restore.sh            Restore PostgreSQL DB from a SQL dump
```

## Prerequisites

* Docker Engine + Docker Compose plugin (v2.24+ for `!reset` tag support)
* For GPU: NVIDIA Container Toolkit + compatible GPU

## Quick Start

### 1. Configure your environment

Copy the override example and fill in host paths:

```bash
cp docker-compose.override.example.yaml docker-compose.override.yaml
```

Edit `.env` with your UID/GID and host directory paths:

```bash
UID=$(id -u)
GID=$(id -g)
HOST_CONFIG_DIR=./config
HOST_DATA_DIR=./data
HOST_RESULTS_DIR=./results
HOST_MODELS_DIR=./models
HOST_LOGS_DIR=./logs
```

Create directories:

```bash
mkdir -p data results models logs
```

### 2. Build and start

Use `compose.sh` with device (`gpu`/`cpu`) and environment (`dev`/`prod`):

```bash
# GPU dev (source code mounted, debug ports open)
./compose.sh gpu dev up --build

# CPU dev
./compose.sh cpu dev up --build

# GPU prod (code baked into image, no debug ports)
./compose.sh gpu prod up --build

# CPU prod
./compose.sh cpu prod up --build
```

What each combination loads:

| Command | Compose files loaded |
|---|---|
| `gpu dev` | base + gpu + override (auto) |
| `cpu dev` | base + cpu + override (auto) |
| `gpu prod` | base + gpu |
| `cpu prod` | base + cpu |

The PostgreSQL `db` service starts automatically as part of the base stack.

### 3. Verify

```bash
curl http://localhost:8090/health
```

## Compose File Reference

| File | Purpose |
|---|---|
| `docker-compose.yaml` | Base: shared service config, PostgreSQL `db` service |
| `docker-compose.gpu.yaml` | Adds nvidia resources and GPU image build |
| `docker-compose.cpu.yaml` | Clears GPU resources, sets CPU image build |
| `docker-compose.override.yaml` | Dev: source mounts, config mount, debug ports, `settings-dev.yaml` — **gitignored** |
| `docker-compose.sqlite.yaml` | Replaces PostgreSQL with a local SQLite file (dev only) |
| `docker-compose.podman.yaml` | Adds `userns_mode: keep-id` for Podman rootless deployments |

To use the SQLite overlay:

```bash
./compose.sh gpu dev -f docker-compose.sqlite.yaml up
```

To use the Podman overlay (production on Podman):

```bash
podman compose \
  -f docker-compose.yaml \
  -f docker-compose.gpu.yaml \
  -f docker-compose.podman.yaml \
  up --build
```

## Database

The stack starts a `postgres:16-alpine` container (`histoseg-db`).
The connection URL defaults to:

```
postgresql+psycopg2://histoseg:histoseg@db:5432/histoseg
```

Override for an external cluster (production):

```bash
export HISTOSEG_DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
./compose.sh gpu prod up
```

### Migrate from SQLite

If you have existing data in a SQLite queue database, migrate it once after starting the PostgreSQL stack:

```bash
# Start the stack (schema is initialised automatically on first boot)
./compose.sh gpu dev up -d

# Dry-run to verify
docker exec histoseg-api python scripts/migrate_sqlite_to_pg.py \
    data/histoseg_queue.db \
    "postgresql+psycopg2://histoseg:histoseg@db:5432/histoseg" \
    --dry-run

# Run for real
docker exec histoseg-api python scripts/migrate_sqlite_to_pg.py \
    data/histoseg_queue.db \
    "postgresql+psycopg2://histoseg:histoseg@db:5432/histoseg"
```

The SQLite file is left untouched; fall back with `-f docker-compose.sqlite.yaml` if needed.

### Backup and restore

```bash
# Dump to backups/ (timestamped)
./scripts/pg_backup.sh

# Restore from a dump (asks for confirmation)
./scripts/pg_restore.sh backups/histoseg_20260825_143012.sql
```

Run a backup before schema migrations and before production deploys.

## Service Ports

| Service | Host port | Container port |
|---|---|---|
| API | 8090 | 8000 |
| API debugpy (dev) | 5679 | 5678 |
| Worker debugpy (dev) | 5678 | 5678 |

* API docs: `http://localhost:8090/docs`
* Queue UI: `http://localhost:8090/web/queue`

## Configuration

Runtime settings are loaded from YAML via `HISTOSEG_CONFIG`.

| Context | Config file |
|---|---|
| Docker dev (override loaded) | `config/settings-dev.yaml` |
| Docker prod (image baked) | `config/settings.yaml` |
| Local (outside Docker) | `config/settings.yaml` or set `HISTOSEG_CONFIG` |

`database_url` is always injected via the `HISTOSEG_DATABASE_URL` environment variable.
For local development without Docker:

```bash
export HISTOSEG_DATABASE_URL=sqlite:///./histoseg_queue.db
```

Key settings:

* `allowed_roots`: slide path allowlist for API requests
* `results_root`: output directory for job artifacts
* `models_root`: model directory for worker loading
* `default_model_id`: default model subdirectory name
* `preferred_device`: `cuda` or `cpu`
* `use_amp`: enable automatic mixed precision (GPU only)

## Models

Place model directories under `models_root`:

```text
models/
  <model_id>/
    manifest.yaml
    model.onnx   (or PyTorch weights)
```

The model manifest defines runtime, weights file, tile size/resolution, input layout, output heads, labels, and output geometry.

## API Overview

### Health

```http
GET /health
```

### Submit asynchronous jobs

```http
POST /jobs
Content-Type: application/json

{
  "items": [{ "slide_uri": "file:///path/to/slide.svs", "model_id": "default" }]
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

### Debug synchronous routes

```http
POST /segment/tissue
POST /segment/wsi
```

## QuPath Integration

### Build extension

```bash
cd extensions/qupath && ./gradlew build
```

Drag the output JAR (`build/libs/`) into QuPath to install.

### Configure

`Extensions > HistoSeg > Settings...` → set server URL and model ID.

### Commands

* `Submit current slide...`
* `Submit all project slides...`
* `Import existing result for current slide...`
* `Queue status...`

## Local Development Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

export HISTOSEG_CONFIG=config/settings.yaml
export HISTOSEG_DATABASE_URL=sqlite:///./histoseg_queue.db

uvicorn histoseg_plugin.api.main:create_app --factory --host 0.0.0.0 --port 8000
# worker in a separate shell:
python -m histoseg_plugin.jobs.worker_main
```

## Testing

```bash
pytest
```

## Troubleshooting

### `403 Slide path not under allowed roots`

Add the slide parent directory to `allowed_roots` in the settings YAML.

### `RuntimeError: Database schema is version 0`

The schema was initialised but migrations were not stamped. Run:

```bash
docker exec histoseg-api python -m histoseg_plugin.db.migrate upgrade
```

### Queue appears stuck

Check worker logs:

```bash
./compose.sh gpu dev logs -f worker
```

Confirm the queue is not paused at `http://localhost:8090/web/queue`.


