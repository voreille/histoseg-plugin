from fastapi import APIRouter, Depends, HTTPException

from histoseg_plugin.api.adapters.jobs import build_job_inputs
from histoseg_plugin.api.dependencies.queue import get_queue_service
from histoseg_plugin.api.schemas import CreateJobRequest
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.settings import Settings, get_settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
def create_job(
    req: CreateJobRequest,
    queue_service: QueueService = Depends(get_queue_service),
    settings: Settings = Depends(get_settings),
):
    try:
        inputs = build_job_inputs(req.items, settings.allowed_roots)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    job = queue_service.submit_batch(inputs)

    return {"job_id": job.id, "status": job.status.value}


@router.get("/{job_id}")
def read_job(
    job_id: int,
    queue_service: QueueService = Depends(get_queue_service),
):
    job = queue_service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return job
