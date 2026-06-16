import logging
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput
from histoseg_plugin.db.models import Task, TaskStatus

from ..db.models import Job
from ..results.io import delete_result_directory
from .contracts import (
    SortOrder,
    TaskQueryResult,
    TaskSortField,
)
from .queue_ops import (
    cancel_task_op,
    count_tasks_by_status_op,
    delete_task_op,
    get_job_op,
    get_task_by_hash_op,
    is_queue_paused_op,
    list_tasks_op,
    pause_queue_op,
    query_tasks_op,
    refresh_job_status_op,
    resume_queue_op,
    retry_task_op,
    set_task_priority_op,
    submit_batch_op,
)

logger = logging.getLogger(__name__)


class QueueService:
    def __init__(self, session_factory: sessionmaker, results_root: Path | str):
        self.session_factory = session_factory
        self.results_root = (
            Path(results_root) if isinstance(results_root, str) else results_root
        )

    def submit_batch(self, items: list[WSISegmentationInput]) -> Job:
        with self.session_factory() as session:
            job = submit_batch_op(session, items)
            session.commit()
            return job

    def get_job(self, job_id: int) -> dict | None:
        with self.session_factory() as session:
            job = get_job_op(session, job_id)
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
            pause_queue_op(session)
            session.commit()

    def resume_queue(self) -> None:
        with self.session_factory() as session:
            resume_queue_op(session)
            session.commit()

    def is_queue_paused(self) -> bool:
        with self.session_factory() as session:
            return is_queue_paused_op(session)

    def refresh_job_status(self, job_id: int) -> None:
        with self.session_factory() as session:
            refresh_job_status_op(session, job_id)
            session.commit()

    def get_task_by_hash(self, task_hash: str) -> dict | None:
        with self.session_factory() as session:
            task = get_task_by_hash_op(session, task_hash)
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

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        with self.session_factory() as session:
            return list_tasks_op(session, status=status, limit=limit, offset=offset)

    def query_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: TaskSortField = "created_at",
        sort_order: SortOrder = "desc",
    ) -> TaskQueryResult:
        with self.session_factory() as session:
            return query_tasks_op(
                session,
                status=status,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )

    def count_tasks_by_status(self) -> dict[str, int]:
        with self.session_factory() as session:
            return count_tasks_by_status_op(session)

    def set_task_priority(self, task_id: int, priority: int) -> Task:
        with self.session_factory.begin() as session:
            return set_task_priority_op(session, task_id, priority)

    def cancel_task(self, task_id: int) -> Task:
        with self.session_factory.begin() as session:
            return cancel_task_op(session, task_id)

    def retry_task(self, task_id: int) -> Task:
        with self.session_factory.begin() as session:
            return retry_task_op(session, task_id)

    def delete_task(
        self,
        task_id: int,
        *,
        delete_result: bool = True,
    ) -> None:
        with self.session_factory.begin() as session:
            deleted = delete_task_op(
                session,
                task_id,
                delete_result=delete_result,
            )

        if deleted.delete_result and deleted.result_dir is not None:
            try:
                delete_result_directory(
                    deleted.result_dir,
                    results_root=self.results_root,
                )
            except OSError:
                logger.exception(
                    "Task %s was deleted, but result directory cleanup failed: %s",
                    task_id,
                    deleted.result_dir,
                )


