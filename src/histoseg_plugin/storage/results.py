import json
from pathlib import Path


def build_result_dir(root: Path, task_hash: str) -> Path:
    result_dir = root / task_hash
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def write_geojson(result_dir: Path, geojson_obj: dict) -> Path:
    path = result_dir / "geojson.json"
    path.write_text(json.dumps(geojson_obj, indent=2), encoding="utf-8")
    return path


def write_stats(result_dir: Path, stats_obj: dict) -> Path:
    path = result_dir / "stats.json"
    path.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    return path