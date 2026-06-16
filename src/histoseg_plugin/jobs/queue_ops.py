import json
from math import ceil

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput

from ..db.models import Job, JobStatus, QueueState, Task, TaskStatus, Result, utcnow
from ..results.ops import find_result_by_hash
from .hashing import sha256_dict
from .contracts import (
    DeletedTask,
    TaskQueryResult,
    TaskSortField,
    SortOrder,
    TASK_SORT_COLUMNS,
    RETRYABLE_TASK_STATUSES,
)


def query_tasks_op(
    session: Session,
    *,
    status: TaskStatus | None = None,
    page: int = 1,
    page_size: int = 50,
    sort_by: TaskSortField = "created_at",
    sort_order: SortOrder = "desc",
) -> TaskQueryResult:
    page = max(page, 1)
    page_size = min(max(page_size, 1), 200)

    filters = []

    if status is not None:
        filters.append(Task.status == status)

    total_items = (
        session.scalar(select(func.count(Task.id)).select_from(Task).where(*filters))
        or 0
    )

    total_pages = max(ceil(total_items / page_size), 1)
    page = min(page, total_pages)

    sort_column = TASK_SORT_COLUMNS[sort_by]

    if sort_order == "asc":
        order_by = (
            sort_column.asc().nulls_last(),
            Task.id.asc(),
        )
    else:
        order_by = (
            sort_column.desc().nulls_last(),
            Task.id.desc(),
        )

    statement = (
        select(Task)
        .where(*filters)
        .order_by(*order_by)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    return TaskQueryResult(
        items=list(session.scalars(statement).all()),
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        status=status,
        sort_by=sort_by,
        sort_order=sort_order,
    )


def ensure_queue_state(session: Session) -> QueueState:
    state = session.get(QueueState, 1)
    if state is None:
        state = QueueState(id=1, paused=False)
        session.add(state)
        session.flush()
    return state


def pause_queue_op(session: Session) -> None:
    ensure_queue_state(session).paused = True
    session.flush()


def resume_queue_op(session: Session) -> None:
    ensure_queue_state(session).paused = False
    session.flush()


def is_queue_paused_op(session: Session) -> bool:
    return ensure_queue_state(session).paused


def submit_batch_op(session: Session, items: list[WSISegmentationInput]) -> Job:
    batch_payload = [item.as_dict() for item in items]
    batch_hash = sha256_dict(batch_payload)

    job = Job(request_hash=batch_hash, status=JobStatus.PENDING)
    session.add(job)
    session.flush()

    for item in items:
        payload = item.as_dict()
        task_hash = sha256_dict(payload)

        existing_result = find_result_by_hash(session, task_hash)

        task = Task(
            job_id=job.id,
            status=TaskStatus.CACHED if existing_result else TaskStatus.PENDING,
            slide_path=str(item.slide_path),
            model_id=item.model_id,
            task_hash=task_hash,
            params_json=json.dumps(payload, sort_keys=True),
            stage="cached" if existing_result else None,
            progress=100.0 if existing_result else 0.0,
            result_id=existing_result.id if existing_result else None,
        )

        session.add(task)

    session.flush()
    refresh_job_status_op(session, job.id)
    return job


def refresh_job_status_op(session: Session, job_id: int) -> None:
    job = session.get(Job, job_id)
    statuses = [t.status for t in job.tasks]

    if all(s in (TaskStatus.COMPLETED, TaskStatus.CACHED) for s in statuses):
        job.status = JobStatus.COMPLETED
    elif any(s == TaskStatus.RUNNING for s in statuses):
        job.status = JobStatus.RUNNING
    elif any(s == TaskStatus.FAILED for s in statuses):
        if any(s in (TaskStatus.COMPLETED, TaskStatus.CACHED) for s in statuses):
            job.status = JobStatus.PARTIAL
        else:
            job.status = JobStatus.FAILED
    elif any(s == TaskStatus.PENDING for s in statuses):
        job.status = JobStatus.PENDING

    session.flush()


def get_job_op(session: Session, job_id: int) -> Job | None:
    return session.get(Job, job_id)


def get_task_by_hash_op(session: Session, task_hash: str) -> Task | None:
    return session.query(Task).filter(Task.task_hash == task_hash).first()


def list_tasks_op(
    session: Session,
    status: TaskStatus | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Task]:
    stmt = (
        select(Task)
        .order_by(
            Task.status.asc(),
            Task.priority.desc(),
            Task.created_at.asc(),
        )
        .limit(limit)
        .offset(offset)
    )

    if status is not None:
        stmt = stmt.where(Task.status == status)

    return list(session.scalars(stmt).all())


def count_tasks_by_status_op(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(Task.status, func.count(Task.id)).group_by(Task.status)
    ).all()

    return {status.value: count for status, count in rows}


def set_task_priority_op(session: Session, task_id: int, priority: int) -> Task:
    task = session.get(Task, task_id)

    if task is None:
        raise ValueError(f"Task {task_id} not found")

    if task.status != TaskStatus.PENDING:
        raise ValueError("Only pending tasks can be reprioritized")

    task.priority = priority
    session.flush()
    session.refresh(task)
    session.expunge(task)
    return task


def delete_task_op(
    session: Session,
    task_id: int,
    *,
    delete_result: bool = True,
) -> DeletedTask:
    task = session.get(Task, task_id)

    if task is None:
        raise ValueError(f"Task {task_id} not found")

    if task.status == TaskStatus.RUNNING:
        raise ValueError("A running task cannot be deleted")

    job_id = task.job_id
    result_id = task.result_id
    result_dir: str | None = None
    should_delete_result = False

    if result_id is not None and delete_result:
        result = session.get(Result, result_id)

        if result is not None:
            other_references = (
                session.scalar(
                    select(func.count(Task.id)).where(
                        Task.result_id == result_id,
                        Task.id != task_id,
                    )
                )
                or 0
            )

            if other_references == 0:
                result_dir = result.result_dir
                should_delete_result = True

    session.delete(task)
    session.flush()

    if should_delete_result and result_id is not None:
        result = session.get(Result, result_id)

        if result is not None:
            session.delete(result)
            session.flush()

    remaining_tasks = (
        session.scalar(select(func.count(Task.id)).where(Task.job_id == job_id)) or 0
    )

    if remaining_tasks > 0:
        refresh_job_status_op(session, job_id)

    return DeletedTask(
        task_id=task_id,
        job_id=job_id,
        result_id=result_id,
        result_dir=result_dir,
        delete_result=should_delete_result,
    )


def cancel_task_op(
    session: Session,
    task_id: int,
) -> Task:
    task = session.get(Task, task_id)

    if task is None:
        raise ValueError(f"Task {task_id} not found")

    if task.status == TaskStatus.PENDING:
        task.status = TaskStatus.CANCELLED
        task.stage = "cancelled"
        task.progress_message = "Cancelled before execution"
        task.cancel_requested = False
        task.finished_at = utcnow()

        session.flush()
        refresh_job_status_op(session, task.job_id)
        return task

    if task.status == TaskStatus.RUNNING:
        if task.cancel_requested:
            return task

        task.cancel_requested = True
        task.progress_message = "Cancellation requested"

        session.flush()
        return task

    raise ValueError("Only pending or running tasks can be cancelled")




def retry_task_op(
    session: Session,
    task_id: int,
) -> Task:
    task = session.get(Task, task_id)

    if task is None:
        raise ValueError(f"Task {task_id} not found")

    if task.status not in RETRYABLE_TASK_STATUSES:
        raise ValueError(
            "Only failed, interrupted, or cancelled tasks can be retried"
        )

    task.status = TaskStatus.PENDING
    task.stage = None
    task.progress = 0.0
    task.progress_message = None
    task.error_message = None
    task.cancel_requested = False
    task.worker_id = None
    task.heartbeat_at = None
    task.started_at = None
    task.finished_at = None
    task.result_id = None

    session.flush()
    refresh_job_status_op(session, task.job_id)

    return task