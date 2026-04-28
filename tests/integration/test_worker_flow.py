from histoseg_plugin.core.geojson.schemas import GeoJSONFeatureCollection
from histoseg_plugin.core.pipeline.contracts import WSISegmentationResult
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.jobs.worker import (
    build_worker_runtime,
    claim_next_task,
    process_task,
)
from histoseg_plugin.core.pipeline.contracts import WSISegmentationInput


def test_worker_processes_one_task(mocker, session_factory, test_settings, test_slide):
    service = QueueService(session_factory)

    job = service.submit_batch(
        [
            WSISegmentationInput(
                slide_path=test_slide,
                model_id="default",
            )
        ]
    )

    fake_bundle = object()

    mocker.patch(
        "histoseg_plugin.jobs.worker.load_inference_bundle",
        return_value=fake_bundle,
    )

    mocker.patch(
        "histoseg_plugin.jobs.worker.run_wsi_segmentation",
        return_value=WSISegmentationResult(
            coords_space="level0",
            tissue=GeoJSONFeatureCollection(type="FeatureCollection", features=[]),
            outputs={},
            statistics=None,
        ),
    )

    runtime = build_worker_runtime(test_settings)

    task = claim_next_task(session_factory, runtime)
    assert task is not None

    process_task(
        session_factory=session_factory,
        runtime=runtime,
        task=task,
        results_root=test_settings.results_root,
    )

    job_details = service.get_job(job.id)

    assert job_details["status"] == "completed"
    assert job_details["tasks"][0]["status"] == "completed"
    assert job_details["tasks"][0]["result_id"] is not None
