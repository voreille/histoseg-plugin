from histoseg_plugin.db.models import JobStatus, TaskStatus
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput


def test_submit_batch_creates_job_and_task(session_factory, test_slide):
    service = QueueService(session_factory)

    job = service.submit_batch(
        [
            WSISegmentationInput(
                slide_path=test_slide,
                model_id="default",
            )
        ]
    )

    assert job.id is not None
    assert job.status == JobStatus.PENDING

    job_details = service.get_job(job.id)

    assert job_details is not None
    assert job_details["job_id"] == job.id
    assert job_details["status"] == "pending"
    assert len(job_details["tasks"]) == 1

    task = job_details["tasks"][0]
    assert task["status"] == "pending"
    assert task["model_id"] == "default"
    assert task["slide_path"] == str(test_slide)


def test_pause_and_resume_queue(session_factory):
    service = QueueService(session_factory)

    assert service.is_queue_paused() is False

    service.pause_queue()
    assert service.is_queue_paused() is True

    service.resume_queue()
    assert service.is_queue_paused() is False
