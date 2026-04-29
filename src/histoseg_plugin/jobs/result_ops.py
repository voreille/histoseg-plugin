from sqlalchemy import select
from sqlalchemy.orm import Session

from histoseg_plugin.jobs.result_models import Result


def find_result_by_hash(session: Session, task_hash: str) -> Result | None:
    return session.scalar(select(Result).where(Result.task_hash == task_hash))


def get_result(session: Session, result_id: int) -> Result | None:
    return session.get(Result, result_id)


def register_result(
    session: Session,
    *,
    task_hash: str,
    slide_path: str,
    model_id: str,
    result_dir: str,
    geojson_path: str | None,
    stats_path: str | None,
) -> Result:
    existing = find_result_by_hash(session, task_hash)
    if existing is not None:
        return existing

    result = Result(
        task_hash=task_hash,
        slide_uri=slide_path,
        model_id=model_id,
        result_dir=result_dir,
        geojson_path=geojson_path,
        stats_path=stats_path,
    )
    session.add(result)
    session.flush()
    return result
