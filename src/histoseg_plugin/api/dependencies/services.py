from fastapi import Request

from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.results.service import ResultService


def get_queue_service(request: Request) -> QueueService:
    return request.app.state.queue_service


def get_result_service(request: Request) -> ResultService:
    return request.app.state.result_service
