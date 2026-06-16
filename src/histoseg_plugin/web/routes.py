from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from histoseg_plugin.api.dependencies.services import get_queue_service
from histoseg_plugin.db.models import TaskStatus
from histoseg_plugin.jobs.queue_ops import SortOrder, TaskSortField
from histoseg_plugin.jobs.queue_service import QueueService

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/web", tags=["web"])


def parse_task_status(status: str | None) -> TaskStatus | None:
    if status in (None, "", "all"):
        return None

    try:
        return TaskStatus(status)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task status: {status}",
        ) from exc


@router.get("/queue")
def queue_page(
    request: Request,
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort_by: TaskSortField = Query(default="created_at"),
    sort_order: SortOrder = Query(default="desc"),
    queue_service: QueueService = Depends(get_queue_service),
):
    selected_status = parse_task_status(status)

    result = queue_service.query_tasks(
        status=selected_status,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    summary = queue_service.count_tasks_by_status()

    return templates.TemplateResponse(
        request=request,
        name="queue.html",
        context={
            "summary": summary,
            "queue_paused": queue_service.is_queue_paused(),
            "tasks": result.items,
            "pagination": result,
            "task_statuses": list(TaskStatus),
            "status_counts": summary,
            "selected_status": selected_status,
            "sort_by": result.sort_by,
            "sort_order": result.sort_order,
        },
    )


@router.get("/partials/queue-summary")
def queue_summary_partial(
    request: Request,
    queue_service: QueueService = Depends(get_queue_service),
):
    return templates.TemplateResponse(
        request=request,
        name="partials/queue_summary.html",
        context={
            "summary": queue_service.count_tasks_by_status(),
            "queue_paused": queue_service.is_queue_paused(),
        },
    )


@router.get("/partials/tasks")
def task_table_partial(
    request: Request,
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort_by: TaskSortField = Query(default="created_at"),
    sort_order: SortOrder = Query(default="desc"),
    queue_service: QueueService = Depends(get_queue_service),
):
    selected_status = parse_task_status(status)

    result = queue_service.query_tasks(
        status=selected_status,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return templates.TemplateResponse(
        request=request,
        name="partials/task_table.html",
        context={
            "tasks": result.items,
            "pagination": result,
            "selected_status": selected_status,
            "sort_by": result.sort_by,
            "sort_order": result.sort_order,
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


@router.post("/queue/tasks/{task_id}/cancel")
def cancel_task(
    task_id: int,
    queue_service: QueueService = Depends(get_queue_service),
):
    try:
        queue_service.cancel_task(task_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return RedirectResponse(
        url="/web/queue",
        status_code=303,
    )


@router.post("/queue/tasks/{task_id}/delete")
def delete_task(
    task_id: int,
    delete_result: bool = Form(default=True),
    queue_service: QueueService = Depends(get_queue_service),
):
    try:
        queue_service.delete_task(
            task_id,
            delete_result=delete_result,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return RedirectResponse(
        url="/web/queue",
        status_code=303,
    )


@router.post("/queue/tasks/{task_id}/retry")
def retry_task(
    task_id: int,
    queue_service: QueueService = Depends(get_queue_service),
):
    try:
        queue_service.retry_task(task_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return RedirectResponse(
        url="/web/queue",
        status_code=303,
    )
