from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, Integer, String, Text
from .queue_models import Base


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    slide_uri: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)

    result_dir: Mapped[str] = mapped_column(Text, nullable=False)
    geojson_path: Mapped[str | None] = mapped_column(Text, default=None)
    stats_path: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)