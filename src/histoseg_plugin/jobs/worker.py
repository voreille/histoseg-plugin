# histoseg_plugin/jobs/worker.py
from __future__ import annotations

import gc
import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from histoseg_plugin.core.inference.bundle import InferenceBundle
from histoseg_plugin.core.inference.loader import load_inference_bundle
from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput
from histoseg_plugin.core.pipeline.wsi_segmentation import run_wsi_segmentation
from histoseg_plugin.core.serialization import to_jsonable
from histoseg_plugin.db.models import Task, TaskStatus
from histoseg_plugin.jobs.queue_ops import (
    is_queue_paused,
    refresh_job_status,
)
from histoseg_plugin.jobs.recovery import reset_stale_running_tasks
from histoseg_plugin.results.ops import register_result
from histoseg_plugin.settings import Settings
from histoseg_plugin.storage.results import (
    build_result_dir,
    write_geojson,
    write_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerRuntime:
    worker_id: str
    model_root: Path
    device: torch.device
    default_model_id: str = "default"
    loaded_model_id: str | None = None
    loaded_model_bundle: InferenceBundle | None = None
    last_activity_ts: float = 0.0

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

        if model_id == "default":
            model_dir = self.model_root / self.default_model_id
        else:
            model_dir = self.model_root / model_id

        logger.info("Loading model %s from %s", model_id, model_dir)

        self.loaded_model_bundle = load_inference_bundle(
            model_dir,
            device=self.device,
        )
        self.loaded_model_id = model_id
        self.touch()

        return self.loaded_model_bundle


def build_worker_runtime(settings: Settings) -> WorkerRuntime:
    use_cuda = torch.cuda.is_available() and settings.preferred_device == "cuda"
    device = torch.device("cuda:0" if use_cuda else "cpu")

    return WorkerRuntime(
        worker_id=f"worker-{socket.gethostname()}",
        model_root=settings.models_root,
        device=device,
        default_model_id=settings.default_model_id,
    )


def claim_next_task(
    session_factory: sessionmaker[Session],
    runtime: WorkerRuntime,
) -> Task | None:
    with session_factory.begin() as session:
        if is_queue_paused(session):
            return None

        stmt = (
            select(Task)
            .where(Task.status == TaskStatus.PENDING)
            .order_by(Task.priority.desc(), Task.created_at.asc())
            .limit(1)
        )

        task = session.scalar(stmt)
        if task is None:
            return None

        task.status = TaskStatus.RUNNING
        task.worker_id = runtime.worker_id
        task.started_at = datetime.utcnow()
        task.heartbeat_at = datetime.utcnow()
        task.stage = "claimed"
        task.progress = 0.0

        session.flush()
        session.expunge(task)

        return task


def update_task_progress(
    session_factory: sessionmaker[Session],
    task_id: int,
    stage: str,
    progress: float,
) -> None:
    with session_factory.begin() as session:
        task = session.get(Task, task_id)
        if task is None:
            return

        task.stage = stage
        task.progress = progress
        task.heartbeat_at = datetime.utcnow()


def mark_task_failed(
    session_factory: sessionmaker[Session],
    task_id: int,
    error_message: str,
) -> None:
    with session_factory.begin() as session:
        task = session.get(Task, task_id)
        if task is None:
            return

        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.finished_at = datetime.utcnow()
        task.heartbeat_at = datetime.utcnow()
        task.stage = "failed"

        refresh_job_status(session, task.job_id)


def task_to_wsi_input(task: Task) -> WSISegmentationInput:
    payload = json.loads(task.params_json)
    return WSISegmentationInput.from_dict(payload)


def process_task(
    session_factory: sessionmaker[Session],
    runtime: WorkerRuntime,
    task: Task,
    results_root: Path,
) -> None:
    update_task_progress(session_factory, task.id, "loading_model", 1.0)

    inference_bundle = runtime.ensure_model_loaded(task.model_id)

    update_task_progress(session_factory, task.id, "running_segmentation", 5.0)

    wsi_input = task_to_wsi_input(task)

    result = run_wsi_segmentation(
        wsi_segmentation_input=wsi_input,
        inference_bundle=inference_bundle,
    )

    update_task_progress(session_factory, task.id, "writing_results", 95.0)

    result_dir = build_result_dir(results_root, task.task_hash)

    output_payload = {
        "coords_space": result.coords_space,
        "tissue": result.tissue,
        "outputs": result.outputs,
    }

    geojson_path = write_geojson(result_dir, to_jsonable(output_payload))
    stats_path = write_stats(result_dir, to_jsonable(result.statistics))

    with session_factory.begin() as session:
        registered_result = register_result(
            session,
            task_hash=task.task_hash,
            slide_path=task.slide_path,
            model_id=task.model_id,
            result_dir=str(result_dir),
            geojson_path=str(geojson_path),
            stats_path=str(stats_path),
        )

        db_task = session.get(Task, task.id)
        if db_task is None:
            raise RuntimeError(f"Task disappeared while processing: {task.id}")

        db_task.status = TaskStatus.COMPLETED
        db_task.progress = 100.0
        db_task.stage = "done"
        db_task.result_id = registered_result.id
        db_task.finished_at = datetime.utcnow()
        db_task.heartbeat_at = datetime.utcnow()

        refresh_job_status(session, db_task.job_id)

    runtime.touch()


def maybe_unload_idle_model(runtime: WorkerRuntime, settings: Settings) -> None:
    if runtime.loaded_model_bundle is None:
        return

    idle_seconds = time.time() - runtime.last_activity_ts
    if idle_seconds > settings.gpu_idle_unload_seconds:
        logger.info("Unloading model after %.1f idle seconds", idle_seconds)
        runtime.unload_model()


def run_worker_forever(
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    runtime = build_worker_runtime(settings)

    logger.info("Worker started: %s on %s", runtime.worker_id, runtime.device)

    if settings.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        logger.info("debugpy listening on 0.0.0.0:5678")
        debugpy.wait_for_client()

    with session_factory.begin() as session:
        reset_stale_running_tasks(session)

    while True:
        task = claim_next_task(session_factory, runtime)

        if task is None:
            maybe_unload_idle_model(runtime, settings)
            time.sleep(settings.worker_poll_interval_seconds)
            continue

        logger.info(
            "Claimed task %s for slide %s with model %s",
            task.id,
            task.slide_path,
            task.model_id,
        )

        try:
            process_task(
                session_factory=session_factory,
                runtime=runtime,
                task=task,
                results_root=settings.results_root,
            )
        except Exception as exc:
            logger.exception("Error processing task %s: %s", task.id, exc)
            mark_task_failed(
                session_factory=session_factory,
                task_id=task.id,
                error_message=str(exc),
            )
