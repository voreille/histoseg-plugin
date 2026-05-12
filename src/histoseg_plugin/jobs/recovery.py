from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..db.models import Task, TaskStatus


def reset_stale_running_tasks(session: Session, stale_seconds: int = 60) -> int:
    cutoff = datetime.utcnow() - timedelta(seconds=stale_seconds)
    tasks = session.scalars(select(Task).where(Task.status == TaskStatus.RUNNING)).all()

    count = 0
    for task in tasks:
        if task.heartbeat_at is None or task.heartbeat_at < cutoff:
            task.status = TaskStatus.PENDING
            task.worker_id = None
            task.started_at = None
            task.heartbeat_at = None
            count += 1

    session.flush()
    return count