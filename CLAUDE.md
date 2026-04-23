# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**histoseg-plugin** is a digital pathology backend for **whole slide image (WSI) segmentation and annotation**.

The project is now centered on a **Dockerized FastAPI + worker architecture** with a DB-backed queue. The goal is to expose WSI segmentation to external clients such as **QuPath**, while keeping the heavy processing logic reusable from both:

- synchronous API routes for debugging and development
- asynchronous background workers for real usage

There are two main layers in the codebase:

1. **API / orchestration layer**
   - FastAPI routes
   - request / response schemas
   - queue / job submission
   - job status / result retrieval
   - Docker deployment

2. **Core processing layer**
   - tissue segmentation
   - tile generation
   - model inference
   - logit stitching
   - postprocessing
   - GeoJSON generation
   - statistics / quantitative outputs

The active development target is the **containerized FastAPI + worker system**, not the older preprocessing / tiling CLI workflows.

---

## Current Priorities

Focus on these tasks first:

1. Keep the core WSI segmentation pipeline reusable from both API routes and workers
2. Finalize the DB-backed queue / job / task execution flow
3. Keep routes thin and move processing logic into `core/`
4. Persist results on disk and metadata in DB
5. Maintain a clean Docker-based dev workflow with API + worker services
6. Preserve a clear separation between:
   - API schemas
   - core runtime contracts
   - core shared schemas

---

## Architecture Overview

### Main components

#### 1. API service
- Receives HTTP requests
- Validates payloads
- Resolves slide URIs
- Creates jobs/tasks
- Returns status/results

#### 2. Worker service
- Polls DB
- Claims tasks
- Runs segmentation pipeline
- Writes results
- Updates DB

#### 3. Database
- Queue state
- Jobs
- Tasks
- Results metadata

#### 4. Filesystem
- GeoJSON outputs
- Stats
- Models
- Logs

---

## Layering Rules (IMPORTANT)

Dependencies:

    api    -> core
    worker -> core
    core   -> core

**Core must not import from API**

---

## Core Runtime Contracts

Defined in `core.pipeline.contracts`:

- `WSISegmentationInput`
- `WSISegmentationResult`

Key idea:
- API converts request → input contract
- Core processes
- Core returns result contract
- API converts → response

---

## Shared Schemas

Use `schemas.py` for Pydantic shared structures:

- `core.geojson.schemas`
- `core.postprocessing.schemas`

Rule:

- Pydantic → schemas
- Dataclasses → contracts

---

## WSI Segmentation Flow

Main function:

    run_wsi_segmentation(...)

Pipeline:

    WSI
     └── open slide
          └── tissue segmentation
               └── tile generation
                    └── model inference
                         └── logit stitching
                              └── postprocessing
                                   └── GeoJSON
                                   └── statistics

---

## API Design

Routes must be thin.

Example:

1. parse request
2. resolve slide path
3. build input contract
4. call core
5. build response

---

## Queue Architecture

### Models

- Job
- Task
- Result
- QueueState

### Workflow

Submission:
- API creates job + tasks

Worker:
- poll
- claim
- process
- store results
- update DB

---

## QuPath Integration

Two modes:

### Sync
    POST /segment/wsi

### Async
    POST /jobs
    → poll status
    → fetch results

---

## Docker Context

- API container
- Worker container
- Shared volumes
- NAS mounted
- HF token via env
- debugpy enabled

Multiprocessing:

- use `forkserver` on Linux
- increase `shm_size` if needed

---

## Critical Invariants

- All coords = level0
- Tiles = level0 top-left
- GeoJSON = level0
- Stitching = weighted accumulation
- No duplicate pipeline logic
- No core → API dependency

---

## Module Responsibilities

API:
    api/routes/
    api/schemas.py

Core:
    core/pipeline/contracts.py
    core/pipeline/wsi_segmentation.py
    core/geojson/schemas.py
    core/postprocessing/schemas.py

---

## Testing

- pytest
- mock WSI I/O
- no real slide dependency