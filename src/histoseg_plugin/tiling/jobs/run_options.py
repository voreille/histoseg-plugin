from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RunOptions:
    generate_mask: bool = True
    generate_patches: bool = True
    generate_stitch: bool = True
    auto_skip: bool = True
    strict_outputs: bool = True
    strict_mpp: bool = True     # if True, out-of-tolerance => FAILED
    max_workers: int = 4
    verbose: bool = True