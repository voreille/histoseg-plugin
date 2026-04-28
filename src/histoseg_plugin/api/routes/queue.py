from fastapi import APIRouter, Depends

from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.api.dependencies.queue import get_queue_service

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("")
def get_queue_state(queue_service: QueueService = Depends(get_queue_service)):
    return {"paused": queue_service.is_queue_paused()}


@router.post("/pause")
def pause(queue_service: QueueService = Depends(get_queue_service)):
    queue_service.pause_queue()
    return {"paused": True}


@router.post("/resume")
def resume(queue_service: QueueService = Depends(get_queue_service)):
    queue_service.resume_queue()
    return {"paused": False}
