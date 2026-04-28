from fastapi import Request

from histoseg_plugin.jobs.queue_service import QueueService


def get_queue_service(request: Request) -> QueueService:
    return request.app.state.queue_service
