import json
from sqlalchemy import select
from sqlalchemy.orm import Session

from .hashing import build_task_payload, sha256_dict
from ..db.models import Job, JobStatus, QueueState, Task, TaskStatus
from .result_ops import find_result_by_hash

from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput


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


def submit_batch(session: Session, items: list[WSISegmentationInput]) -> Job:
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


def get_task_by_hash(session: Session, task_hash: str) -> Task | None:
    return session.query(Task).filter(Task.task_hash == task_hash).first()
