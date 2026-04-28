from fastapi import APIRouter, Depends, HTTPException

from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.api.schemas import CreateJobRequest
from histoseg_plugin.api.dependencies.queue import get_queue_service


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
def create_job(
    req: CreateJobRequest,
    queue_service: QueueService = Depends(get_queue_service),
):
    job = queue_service.submit_batch(req.items)

    return {
        "job_id": job.id,
        "status": job.status.value,
    }


@router.get("/{job_id}")
def read_job(
    job_id: int,
    queue_service: QueueService = Depends(get_queue_service),
):
    job = queue_service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.id,
        "status": job.status.value,
        "tasks": [
            {
                "task_id": t.id,
                "status": t.status.value,
                "slide_path": t.slide_path,  # <- updated
                "model_id": t.model_id,
                "progress": t.progress,
                "stage": t.stage,
                "error_message": t.error_message,
                "result_id": t.result_id,
            }
            for t in job.tasks
        ],
    }
