from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResultMetadata:
    schema_version: int
    task_hash: str
    slide_path: str
    model_id: str
    params: dict[str, Any]
    result_dir: str
    geojson_path: str | None = None
    stats_path: str | None = None
    pipeline_name: str | None = None
    pipeline_version: str | None = None
    created_at: str = ""

    @classmethod
    def create(
        cls,
        *,
        task_hash: str,
        slide_path: str | Path,
        model_id: str,
        params: dict[str, Any],
        result_dir: str | Path,
        geojson_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        pipeline_name: str | None = None,
        pipeline_version: str | None = None,
    ) -> "ResultMetadata":
        return cls(
            schema_version=1,
            task_hash=task_hash,
            slide_path=str(slide_path),
            model_id=model_id,
            params=params,
            result_dir=str(result_dir),
            geojson_path=str(geojson_path) if geojson_path is not None else None,
            stats_path=str(stats_path) if stats_path is not None else None,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            created_at=utcnow_iso(),
        )

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "ResultMetadata":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
