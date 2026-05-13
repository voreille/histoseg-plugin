from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from histoseg_plugin.api.dependencies.services import get_queue_service
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.db.models import TaskStatus

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/web", tags=["web"])


@router.get("/queue", response_class=HTMLResponse)
def queue_page(
    request: Request,
    queue_service: QueueService = Depends(get_queue_service),
):
    summary = queue_service.count_tasks_by_status()
    tasks = queue_service.list_tasks(limit=100)

    return templates.TemplateResponse(
        request=request,
        name="queue.html",
        context={
            "summary": summary,
            "tasks": tasks,
            "task_statuses": list(TaskStatus),
        },
    )


@router.get("/partials/queue-summary", response_class=HTMLResponse)
def queue_summary_partial(
    request: Request,
    queue_service: QueueService = Depends(get_queue_service),
):
    summary = queue_service.count_tasks_by_status()

    return templates.TemplateResponse(
        request=request,
        name="partials/queue_summary.html",
        context={
            "summary": summary,
            "task_statuses": list(TaskStatus),
        },
    )


@router.get("/partials/tasks", response_class=HTMLResponse)
def task_table_partial(
    request: Request,
    status: TaskStatus | None = None,
    queue_service: QueueService = Depends(get_queue_service),
):
    tasks = queue_service.list_tasks(status=status, limit=100)

    return templates.TemplateResponse(
        request=request,
        name="partials/task_table.html",
        context={
            "tasks": tasks,
        },
    )


@router.post("/queue/pause")
def pause_queue(
    queue_service: QueueService = Depends(get_queue_service),
):
    queue_service.pause_queue()
    return RedirectResponse("/web/queue", status_code=303)


@router.post("/queue/resume")
def resume_queue(
    queue_service: QueueService = Depends(get_queue_service),
):
    queue_service.resume_queue()
    return RedirectResponse("/web/queue", status_code=303)


@router.post("/tasks/{task_id}/priority")
def set_task_priority(
    task_id: int,
    priority: int = Form(...),
    queue_service: QueueService = Depends(get_queue_service),
):
    queue_service.set_task_priority(task_id, priority)
    return RedirectResponse("/web/queue", status_code=303)
