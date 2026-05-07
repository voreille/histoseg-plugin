from fastapi import APIRouter, Depends, HTTPException

from histoseg_plugin.api.adapters.jobs import build_job_item_input
from histoseg_plugin.api.dependencies.services import (
    get_queue_service,
    get_result_service,
)
from histoseg_plugin.api.schemas import JobItem
from histoseg_plugin.jobs.hashing import sha256_dict
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.jobs.result_service import ResultService
from histoseg_plugin.settings import Settings, get_settings

router = APIRouter(prefix="/results", tags=["results"])


@router.post("/lookup")
def lookup_result(
    job_item: JobItem,
    settings: Settings = Depends(get_settings),
    queue_service: QueueService = Depends(get_queue_service),
):
    seg_input = build_job_item_input(job_item, settings.allowed_roots)
    task_hash = sha256_dict(seg_input.as_dict())

    task = queue_service.get_task_by_hash(task_hash)

    if task is None:
        return {"found": False}

    return {
        "found": True,
        "status": task["status"],
        "task_id": task["task_id"],
        "result_id": task["result_id"],
    }


@router.get("/{result_id}")
def get_result(
    result_id: int,
    result_service: ResultService = Depends(get_result_service),
):
    payload = result_service.get_result_payload(result_id)

    if payload is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return payload
