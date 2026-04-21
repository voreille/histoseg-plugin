import gc
import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from sqlalchemy import select

from histoseg_plugin.settings import Settings
from histoseg_plugin.storage.results import build_result_dir, write_geojson, write_stats

from .db import get_session
from .queue_models import Task, TaskStatus
from .queue_service import is_queue_paused, refresh_job_status
from .recovery import reset_stale_running_tasks
from .result_service import register_result

logger = logging.getLogger(__name__)


@dataclass
class WorkerRuntime:
    worker_id: str
    loaded_model_id: str | None = None
    loaded_model: object | None = None
    last_activity_ts: float = 0.0

    def touch(self) -> None:
        self.last_activity_ts = time.time()

    def unload_model(self) -> None:
        self.loaded_model = None
        self.loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_model_loaded(self, model_id: str):
        if self.loaded_model_id == model_id and self.loaded_model is not None:
            return self.loaded_model

        self.unload_model()
        # TODO replace with your real model loading entry point
        self.loaded_model = f"model:{model_id}"
        self.loaded_model_id = model_id
        self.touch()
        return self.loaded_model


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


def process_task(runtime: WorkerRuntime, task: Task, results_dir: Path) -> None:
    runtime.ensure_model_loaded(task.model_id)

    params = json.loads(task.params_json)

    update_task_progress(task.id, "starting", 1.0)

    # TODO connect this to your real pipeline.
    # Suggested shape:
    # result = run_segmentation_job(slide_uri=task.slide_uri, model_id=task.model_id, params=params, ...)
    #
    # result should contain at least:
    # - geojson
    # - stats
    # - maybe extra metadata

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
    if runtime.loaded_model is None:
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
