from sqlalchemy.orm import sessionmaker

from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput
from ..db.models import Job
from .queue_ops import (
    get_job,
    is_queue_paused,
    pause_queue,
    refresh_job_status,
    resume_queue,
    submit_batch,
    get_task_by_hash,
)


class QueueService:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def submit_batch(self, items: list[WSISegmentationInput]) -> Job:
        with self.session_factory() as session:
            job = submit_batch(session, items)
            session.commit()
            return job

    def get_job(self, job_id: int) -> dict | None:
        with self.session_factory() as session:
            job = get_job(session, job_id)
            if job is None:
                return None

            return {
                "job_id": job.id,
                "status": job.status.value,
                "tasks": [
                    {
                        "task_id": t.id,
                        "status": t.status.value,
                        "slide_path": str(t.slide_path),
                        "model_id": t.model_id,
                        "progress": t.progress,
                        "stage": t.stage,
                        "error_message": t.error_message,
                        "result_id": t.result_id,
                    }
                    for t in job.tasks
                ],
            }

    def pause_queue(self) -> None:
        with self.session_factory() as session:
            pause_queue(session)
            session.commit()

    def resume_queue(self) -> None:
        with self.session_factory() as session:
            resume_queue(session)
            session.commit()

    def is_queue_paused(self) -> bool:
        with self.session_factory() as session:
            return is_queue_paused(session)

    def refresh_job_status(self, job_id: int) -> None:
        with self.session_factory() as session:
            refresh_job_status(session, job_id)
            session.commit()

    def get_task_by_hash(self, task_hash: str) -> dict | None:
        with self.session_factory() as session:
            task = get_task_by_hash(session, task_hash)
            if task is None:
                return None

            return {
                "task_id": task.id,
                "status": task.status.value,
                "slide_path": str(task.slide_path),
                "model_id": task.model_id,
                "progress": task.progress,
                "stage": task.stage,
                "error_message": task.error_message,
                "result_id": task.result_id,
            }
