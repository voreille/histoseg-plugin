import json
from pathlib import Path
from typing import Any


def load_json_file(path: str | Path | None) -> Any | None:
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    with path.open("r") as f:
        return json.load(f)


def load_result_payload(*, geojson_path: str | None, stats_path: str | None) -> dict:
    payload = load_json_file(geojson_path)

    if payload is None:
        payload = {}

    stats = load_json_file(stats_path)
    if stats is not None:
        payload["statistics"] = stats

    return payload
