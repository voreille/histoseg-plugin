from sqlalchemy.orm import sessionmaker

from histoseg_plugin.results.ops import get_result, find_result_by_hash
from histoseg_plugin.results.store import load_result_payload


class ResultService:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def get_result_payload(self, result_id: int) -> dict | None:
        with self.session_factory() as session:
            result = get_result(session, result_id)
            if result is None:
                return None

            return load_result_payload(
                geojson_path=result.geojson_path,
                stats_path=result.stats_path,
            )

    def find_by_hash(self, task_hash: str) -> dict | None:
        with self.session_factory() as session:
            result = find_result_by_hash(session, task_hash)
            if result is None:
                return None

            return {
                "result_id": result.id,
                "task_hash": result.task_hash,
                "slide_uri": result.slide_path,
                "model_id": result.model_id,
                "result_dir": result.result_dir,
                "geojson_path": result.geojson_path,
                "stats_path": result.stats_path,
            }
