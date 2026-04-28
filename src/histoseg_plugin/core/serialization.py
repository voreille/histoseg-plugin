# core/serialization.py
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    return obj
