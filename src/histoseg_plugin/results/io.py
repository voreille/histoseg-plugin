import json
import shutil
from pathlib import Path
from typing import Any

from histoseg_plugin.db.models import Task
from histoseg_plugin.results.metadata import ResultMetadata

GEOJSON_FILENAME = "predictions.geojson"
STATS_FILENAME = "stats.json"
METADATA_FILENAME = "result_metadata.json"


def build_result_dir(root: Path, task_hash: str) -> Path:
    result_dir = root / task_hash
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def write_geojson(result_dir: Path, geojson_obj: dict) -> Path:
    path = result_dir / GEOJSON_FILENAME
    path.write_text(json.dumps(geojson_obj, indent=2), encoding="utf-8")
    return path


def write_stats(result_dir: Path, stats_obj: dict) -> Path:
    path = result_dir / STATS_FILENAME
    path.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    return path


def write_result_metadata(result_dir: Path, task: Task) -> Path:
    path = result_dir / METADATA_FILENAME
    geojson_path = result_dir / GEOJSON_FILENAME
    stats_path = result_dir / STATS_FILENAME
    result_metadata = ResultMetadata.create(
        task_hash=task.task_hash,
        slide_path=task.slide_path,
        model_id=task.model_id,
        params=json.loads(task.params_json),
        result_dir=str(result_dir),
        geojson_path=str(geojson_path),
        stats_path=str(stats_path),
    )
    result_metadata.write(path)

    return path


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



def delete_result_directory(
    result_dir: str | Path,
    *,
    results_root: str | Path,
) -> None:
    root = Path(results_root).resolve()
    directory = Path(result_dir).resolve()

    if directory == root:
        raise ValueError("Refusing to delete the results root itself")

    if not directory.is_relative_to(root):
        raise ValueError(
            f"Result directory is outside the configured results root: {directory}"
        )

    if directory.exists():
        shutil.rmtree(directory)