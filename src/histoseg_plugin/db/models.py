from datetime import datetime, timezone
from enum import Enum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    Boolean,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CACHED = "cached"
    CANCELLED = "cancelled"


def enum_values(enum_cls):
    return [item.value for item in enum_cls]


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, values_callable=enum_values),
        default=JobStatus.PENDING,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id"), index=True, nullable=False
    )

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, values_callable=enum_values),
        default=TaskStatus.PENDING,
        index=True,
        nullable=False,
    )

    # Queue ordering
    priority: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, index=True
    )

    slide_path: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(
        String(128), nullable=False, default="default"
    )
    task_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Progress / UI
    stage: Mapped[str | None] = mapped_column(String(128), default=None)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    progress_message: Mapped[str | None] = mapped_column(String(255), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    # Cancellation
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    result_id: Mapped[int | None] = mapped_column(
        ForeignKey("results.id"), nullable=True, index=True
    )

    worker_id: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
    )
    job: Mapped["Job"] = relationship(back_populates="tasks")
    result: Mapped["Result | None"] = relationship()


class QueueState(Base):
    __tablename__ = "queue_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    paused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    task_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    slide_path: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)

    result_dir: Mapped[str] = mapped_column(Text, nullable=False)
    geojson_path: Mapped[str | None] = mapped_column(Text, default=None)
    stats_path: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
