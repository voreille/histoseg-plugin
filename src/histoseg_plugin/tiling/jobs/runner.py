from __future__ import annotations

import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import Callable, Union

from ..parameter_models import Config
from .domain import JobStatus, TilingJobCollection, TilingResult
from .inspector import OutputInspector
from .run_options import RunOptions
from .store import JobStore

# thread lock used only to guard store.save_statuses in parallel updates
_store_lock = threading.Lock()


# Small worker that runs in each process
def _worker_direct(
        wsi_path: str,
        output_dir: str,
        cfg_json: str,
        opts: RunOptions,
        process_single_fn: Callable[
            ..., TilingResult],  # returns a dict with 'status', times, etc.
) -> TilingResult:
    cfg = Config.model_validate_json(cfg_json)
    return process_single_fn(
        wsi_path=Path(wsi_path),
        output_dir=Path(output_dir),
        config=cfg,
        generate_mask=opts.generate_mask,
        generate_patches=opts.generate_patches,
        generate_stitch=opts.generate_stitch,
        auto_skip=True,
        verbose=opts.verbose,
    )


def run_tiling_jobs_parallel(
    joblist: TilingJobCollection,
    output_dir: Union[str, Path],
    *,
    store: JobStore | None,
    process_single_fn: Callable[...,
                                TilingResult],  # pass the function directly
    opts: RunOptions = RunOptions(),
) -> TilingJobCollection:
    output_dir = Path(output_dir)
    inspector = OutputInspector(output_dir)

    # Autoskip via filesystem inspection (treat as already done)
    if opts.auto_skip:
        for j in joblist.jobs:
            if j.process and j.status in (JobStatus.PENDING, JobStatus.FAILED):
                if inspector.is_complete(j.slide_id, opts):
                    j.status = JobStatus.PROCESSED
        if store:
            with _store_lock:
                store.save_statuses(joblist)

    idxs = joblist.pick_pending()
    if opts.verbose:
        print(f"Scheduling {len(idxs)} jobs")

    t0 = time()

    with ProcessPoolExecutor(max_workers=opts.max_workers) as ex:
        fut_to_idx = {}
        for idx in idxs:
            job = joblist.jobs[idx]
            job.status = JobStatus.RUNNING
            if store:
                with _store_lock:
                    store.save_statuses(joblist)

            fut = ex.submit(
                _worker_direct,
                str(job.slide_path),
                str(output_dir),
                job.config.model_dump_json(),
                opts,
                process_single_fn,  # pass function object
            )
            fut_to_idx[fut] = idx

        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            job = joblist.jobs[idx]
            try:
                r = fut.result()
                job.status = JobStatus.coerce(r.get("status"))
                # (Optional) attach raw result if you want:
                # job.result = TilingResult(...); or job.extra = r
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                if opts.verbose:
                    print(f"[{job.slide_id}] error: {e}")
            finally:
                if store:
                    with _store_lock:
                        store.save_statuses(joblist)

    if opts.verbose:
        dt = time() - t0
        n_ok = sum(1 for j in joblist.jobs if j.status is JobStatus.PROCESSED)
        print(f"Done in {dt:.1f}s — processed {n_ok}/{len(idxs)}")

    return joblist


def run_tiling_jobs(
        joblist: TilingJobCollection,
        output_dir: Union[str, Path],
        *,
        store: JobStore | None,
        process_single_fn: Callable[..., TilingResult],
        opts: RunOptions = RunOptions(),
) -> TilingJobCollection:
    output_dir = Path(output_dir)
    inspector = OutputInspector(output_dir)

    # normalize 'running' -> 'pending'
    joblist.normalize_for_resume()
    if store:
        store.save_statuses(joblist)

    # filesystem-based autoskip
    if opts.auto_skip:
        for j in joblist.jobs:
            if j.process and j.status in (JobStatus.PENDING, JobStatus.FAILED):
                if inspector.is_complete(j.slide_id, opts):
                    j.status = JobStatus.PROCESSED
                    j.error = None
        if store:
            store.save_statuses(joblist)

    # run eligible jobs
    for idx in joblist.pick_pending():
        j = joblist.jobs[idx]
        j.status = JobStatus.RUNNING
        j.error = None
        if store:
            store.save_statuses(joblist)

        try:
            res = process_single_fn(
                wsi_path=j.slide_path,
                output_dir=output_dir,
                config=j.config,
                generate_mask=opts.generate_mask,
                generate_patches=opts.generate_patches,
                generate_stitch=opts.generate_stitch,
                verbose=opts.verbose,
            )
            # MPP policy
            if opts.strict_mpp and not res.mpp_within_tolerance and j.config.tiling.level_mode == "auto":
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
            if store:
                store.save_statuses(joblist)

    return joblist
