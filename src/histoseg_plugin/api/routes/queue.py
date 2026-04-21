from fastapi import APIRouter
from histoseg_plugin.jobs.db import get_session
from histoseg_plugin.jobs.queue_service import pause_queue, resume_queue, is_queue_paused

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("")
def get_queue_state():
    with get_session() as session:
        return {"paused": is_queue_paused(session)}


@router.post("/pause")
def pause():
    with get_session() as session:
        pause_queue(session)
        return {"paused": True}


@router.post("/resume")
def resume():
    with get_session() as session:
        resume_queue(session)
        return {"paused": False}