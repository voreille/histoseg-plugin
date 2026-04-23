from sqlalchemy.orm import sessionmaker

from .queue_models import Job 
from .queue_ops import (
    get_job,
    is_queue_paused,
    pause_queue,
    refresh_job_status,
    resume_queue,
    submit_batch,
)


class QueueService:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def submit_batch(self, items: list[dict]) -> Job:
        with self.session_factory() as session:
            job = submit_batch(session, items)
            session.commit()
            return job

    def get_job(self, job_id: int) -> Job | None:
        with self.session_factory() as session:
            return get_job(session, job_id)

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
