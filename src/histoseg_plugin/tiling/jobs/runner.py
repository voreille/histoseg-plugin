from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Union

from ...storage.specs import TilingStoresSpec
from ..parameter_models import TilingConfig
from .domain import JobStatus, TilingJobCollection, TilingResult
from .run_options import RunOptions
from .store import JobStore

# thread lock used only to guard store.save_statuses in parallel updates
_store_lock = threading.Lock()


def run_tiling_jobs(
    joblist: TilingJobCollection,
    output_dir: Union[str, Path],
    *,
    job_store: JobStore | None,
    process_single_fn: Callable[..., TilingResult],
    opts: RunOptions = RunOptions(),
    store_spec: TilingStoresSpec,
) -> TilingJobCollection:
    output_dir = Path(output_dir)
    # TODO: add or rm autoskip

    # normalize 'running' -> 'pending'
    joblist.normalize_for_resume()
    if job_store:
        job_store.save_statuses(joblist)

    # run eligible jobs
    for idx in joblist.pick_pending():
        j = joblist.jobs[idx]
        j.status = JobStatus.RUNNING
        j.error = None
        if job_store:
            job_store.save_statuses(joblist)

        try:
            res = process_single_fn(
                wsi_path=j.slide_path,
                config=j.config,
                store_spec=store_spec,
                generate_mask=opts.generate_mask,
                generate_patches=opts.generate_patches,
                generate_stitch=opts.generate_stitch,
                verbose=opts.verbose,
            )
            # MPP policy
            if opts.strict_mpp and not res.mpp_within_tolerance and j.config.resolution.level_mode == "auto":
                j.status = JobStatus.FAILED
                j.result = res
                j.error = res.mpp_reason or "Tile MPP out of tolerance"
            else:
                j.status = JobStatus.PROCESSED
                j.result = res
                j.error = None

        except Exception as e:
            if opts.verbose:
                print(f"[{j.slide_id}] error: {e}")
            j.result = None
            j.status = JobStatus.FAILED
            j.error = str(e)
        finally:
            if job_store:
                job_store.save_statuses(joblist)

    return joblist
