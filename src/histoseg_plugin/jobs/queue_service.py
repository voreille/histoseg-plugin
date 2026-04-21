import json
from sqlalchemy import select
from sqlalchemy.orm import Session

from .hashing import build_task_payload, sha256_dict
from .queue_models import Job, JobStatus, QueueState, Task, TaskStatus
from .result_service import find_result_by_hash


def ensure_queue_state(session: Session) -> QueueState:
    state = session.get(QueueState, 1)
    if state is None:
        state = QueueState(id=1, paused=False)
        session.add(state)
        session.flush()
    return state


def pause_queue(session: Session) -> None:
    ensure_queue_state(session).paused = True
    session.flush()


def resume_queue(session: Session) -> None:
    ensure_queue_state(session).paused = False
    session.flush()


def is_queue_paused(session: Session) -> bool:
    return ensure_queue_state(session).paused


def submit_batch(session: Session, items: list[dict]) -> Job:
    batch_hash = sha256_dict(items)
    job = Job(request_hash=batch_hash, status=JobStatus.PENDING)
    session.add(job)
    session.flush()

    for item in items:
        payload = build_task_payload(
            slide_uri=item["slide_uri"],
            model_id=item.get("model_id", "default"),
            params=item.get("params", {}),
        )
        task_hash = sha256_dict(payload)

        existing_result = find_result_by_hash(session, task_hash)
        if existing_result is not None:
            task = Task(
                job_id=job.id,
                status=TaskStatus.CACHED,
                slide_uri=item["slide_uri"],
                model_id=item.get("model_id", "default"),
                task_hash=task_hash,
                params_json=json.dumps(item.get("params", {})),
                stage="cached",
                progress=100.0,
                result_id=existing_result.id,
            )
        else:
            task = Task(
                job_id=job.id,
                status=TaskStatus.PENDING,
                slide_uri=item["slide_uri"],
                model_id=item.get("model_id", "default"),
                task_hash=task_hash,
                params_json=json.dumps(item.get("params", {})),
            )
        session.add(task)

    session.flush()
    refresh_job_status(session, job.id)
    return job


def refresh_job_status(session: Session, job_id: int) -> None:
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


def get_job(session: Session, job_id: int) -> Job | None:
    return session.get(Job, job_id)
