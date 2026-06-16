from dataclasses import dataclass
from typing import Literal

from ..db.models import (
    Task,
    TaskStatus,
)  # TODO: decouples the contracts module to the db models

TaskSortField = Literal[
    "status",
    "priority",
    "created_at",
    "started_at",
    "finished_at",
    "progress",
    "worker_id",
    "model_id",
]

SortOrder = Literal["asc", "desc"]

TASK_SORT_COLUMNS = {
    "status": Task.status,
    "priority": Task.priority,
    "created_at": Task.created_at,
    "started_at": Task.started_at,
    "finished_at": Task.finished_at,
    "progress": Task.progress,
    "worker_id": Task.worker_id,
    "model_id": Task.model_id,
}

RETRYABLE_TASK_STATUSES = {
    TaskStatus.FAILED,
    TaskStatus.INTERRUPTED,
    TaskStatus.CANCELLED,
}


@dataclass(frozen=True)
class TaskQueryResult:
    items: list[Task]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    status: TaskStatus | None
    sort_by: TaskSortField
    sort_order: SortOrder

    @property
    def has_previous(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages


@dataclass(frozen=True)
class DeletedTask:
    task_id: int
    job_id: int
    result_id: int | None
    result_dir: str | None
    delete_result: bool
