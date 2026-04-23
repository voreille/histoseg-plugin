from fastapi import APIRouter, HTTPException
from histoseg_plugin.jobs.db import get_session
from histoseg_plugin.jobs.queue_ops import submit_batch, get_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
def create_job(payload: dict):
    items = payload["items"]
    with get_session() as session:
        job = submit_batch(session, items)
        return {"job_id": job.id, "status": job.status.value}


@router.get("/{job_id}")
def read_job(job_id: int):
    with get_session() as session:
        job = get_job(session, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.id,
            "status": job.status.value,
            "tasks": [
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "slide_uri": t.slide_uri,
                    "model_id": t.model_id,
                    "progress": t.progress,
                    "stage": t.stage,
                    "error_message": t.error_message,
                    "result_id": t.result_id,
                }
                for t in job.tasks
            ],
        }
