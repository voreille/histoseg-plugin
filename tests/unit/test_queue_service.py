from pathlib import Path


def test_submit_job_creates_pending_task(session_factory, test_settings):
    from histoseg_plugin.core.pipeline.contracts import (
        WSISegmentationInput,
    )
    from histoseg_plugin.jobs.queue_service import QueueService

    service = QueueService(session_factory=session_factory)

    seg_input = WSISegmentationInput(
        slide_path=test_settings.allowed_roots[0] / "slide.svs",
        model_id="test_model",
    )

    job = service.submit_batch([seg_input.as_dict()])

    db_job = service.get_job(job.id)

    assert db_job is not None
