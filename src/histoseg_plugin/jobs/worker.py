import gc
import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from matplotlib.pylab import Any

import torch
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from histoseg_plugin.core.inference.bundle import InferenceBundle
from histoseg_plugin.core.inference.loader import load_inference_bundle
from histoseg_plugin.core.pipeline.contracts import (
    InferenceParams,
    TilingParams,
    TissueSegmentationParams,
    WSISegmentationInput,
)
from histoseg_plugin.core.pipeline.wsi_segmentation import run_wsi_segmentation
from histoseg_plugin.io.slide import assert_allowed_root, slide_uri_to_path
from histoseg_plugin.settings import Settings, get_settings
from histoseg_plugin.storage.results import build_result_dir, write_geojson, write_stats

from .db import get_session
from .queue_models import Task, TaskStatus
from .queue_ops import is_queue_paused, refresh_job_status
from .recovery import reset_stale_running_tasks
from .result_service import register_result

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class WorkerRuntime:
    worker_id: str
    loaded_model_id: str | None = None
    loaded_model_bundle: InferenceBundle | None = None
    last_activity_ts: float = 0.0
    model_dir: Path = Path(settings.models_root)
    device: torch.device = torch.device(
        "cuda:0"
        if torch.cuda.is_available() and settings.preferred_device == "cuda"
        else "cpu"
    )

    def touch(self) -> None:
        self.last_activity_ts = time.time()

    def unload_model(self) -> None:
        self.loaded_model_bundle = None
        self.loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_model_loaded(self, model_id: str) -> InferenceBundle:
        if self.loaded_model_id == model_id and self.loaded_model_bundle is not None:
            return self.loaded_model_bundle

        self.unload_model()
        self.loaded_model_bundle = load_inference_bundle(
            self.model_dir / model_id, device=self.device
        )
        self.loaded_model_id = model_id
        self.touch()
        return self.loaded_model_bundle


def claim_next_task(runtime: WorkerRuntime) -> Task | None:
    with get_session() as session:
        if is_queue_paused(session):
            return None

        stmt = (
            select(Task)
            .where(Task.status == TaskStatus.PENDING)
            .order_by(Task.id.asc())
            .limit(1)
        )
        task = session.scalar(stmt)
        if task is None:
            return None

        task.status = TaskStatus.RUNNING
        task.worker_id = runtime.worker_id
        task.started_at = datetime.utcnow()
        task.heartbeat_at = datetime.utcnow()
        session.flush()
        session.expunge(task)
        return task


def update_task_progress(task_id: int, stage: str, progress: float) -> None:
    with get_session() as session:
        task = session.get(Task, task_id)
        if task is None:
            return
        task.stage = stage
        task.progress = progress
        task.heartbeat_at = datetime.utcnow()
        session.flush()


def resolve_and_check_slide(slide_uri: str) -> Path:
    slide_path = slide_uri_to_path(slide_uri)
    try:
        assert_allowed_root(slide_path, settings.allowed_roots)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return slide_path


def task_to_wsi_input(task: Task) -> WSISegmentationInput:

    return WSISegmentationInput.from_dict(task.params_json)


def process_task(runtime: WorkerRuntime, task: Task, results_dir: Path) -> None:
    runtime.ensure_model_loaded(task.model_id)

    params = json.loads(task.params_json)

    update_task_progress(task.id, "starting", 1.0)
    inference_bundle = runtime.ensure_model_loaded(task.model_id)
    result = run_wsi_segmentation(
        wsi_segmentation_input=build_wsi_segmentation_input(params),
        inference_bundle=inference_bundle,
    )

    dummy_geojson = {"type": "FeatureCollection", "features": []}
    dummy_stats = {"slide_uri": task.slide_uri, "model_id": task.model_id}

    result_dir = build_result_dir(results_dir, task.task_hash)
    geojson_path = write_geojson(result_dir, dummy_geojson)
    stats_path = write_stats(result_dir, dummy_stats)

    with get_session() as session:
        result = register_result(
            session,
            task_hash=task.task_hash,
            slide_uri=task.slide_uri,
            model_id=task.model_id,
            result_dir=str(result_dir),
            geojson_path=str(geojson_path),
            stats_path=str(stats_path),
        )

        db_task = session.get(Task, task.id)
        db_task.status = TaskStatus.COMPLETED
        db_task.progress = 100.0
        db_task.stage = "done"
        db_task.result_id = result.id
        db_task.finished_at = datetime.utcnow()
        db_task.heartbeat_at = datetime.utcnow()
        session.flush()
        refresh_job_status(session, db_task.job_id)

    runtime.touch()


def maybe_unload_idle_model(runtime: WorkerRuntime, settings: Settings) -> None:
    if runtime.loaded_model_bundle is None:
        return
    if time.time() - runtime.last_activity_ts > settings.gpu_idle_unload_seconds:
        runtime.unload_model()


def run_worker_forever(settings: Settings) -> None:
    runtime = WorkerRuntime(worker_id=f"worker-{socket.gethostname()}")
    logger.info("Worker started: %s", runtime.worker_id)
    if settings.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

    with get_session() as session:
        reset_stale_running_tasks(session)

    while True:
        task = claim_next_task(runtime)
        if task is None:
            maybe_unload_idle_model(runtime, settings)
            time.sleep(settings.worker_poll_interval_seconds)
            continue

        logger.info("Claimed task %s for slide %s", task.id, task.slide_uri)
        try:
            process_task(runtime, task, settings.results_root)
        except Exception as exc:
            with get_session() as session:
                db_task = session.get(Task, task.id)
                if db_task is not None:
                    db_task.status = TaskStatus.FAILED
                    db_task.error_message = str(exc)
                    db_task.finished_at = datetime.utcnow()
                    db_task.heartbeat_at = datetime.utcnow()
                    session.flush()
                    refresh_job_status(session, db_task.job_id)
            logger.exception("Error processing task %d: %s", task.id, exc)
            continue
