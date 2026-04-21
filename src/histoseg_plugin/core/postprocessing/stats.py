from __future__ import annotations

from typing import Any

import torch

from .schemas import (
    CompartmentPatternStats,
    DemoPatternStatistics,
    PatternBoundStats,
)


def _make_id_to_name(labels: list[Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for label in labels:
        out[int(label.id)] = str(label.name)
    return out


def _compute_argmax_prediction(
    logits: torch.Tensor,
    covered_mask: torch.Tensor | None = None,
    background_id: int = 0,
) -> torch.Tensor:
    """
    logits: (C, H, W)
    returns: (H, W) long
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (C,H,W), got {tuple(logits.shape)}")

    pred = torch.argmax(logits, dim=0).to(torch.long)

    if covered_mask is not None:
        if covered_mask.shape != pred.shape:
            raise ValueError(
                f"covered_mask shape {tuple(covered_mask.shape)} does not match "
                f"prediction shape {tuple(pred.shape)}"
            )
        pred = pred.clone()
        pred[~covered_mask.bool()] = background_id

    return pred.cpu()


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_demo_pattern_statistics(
    *,
    pixel_area_um2: float,
    outputs: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    head_a_labels: list[Any],
    head_b_labels: list[Any],
    head_a_name: str = "logits_a",
    head_b_name: str = "logits_b",
    head_b_max_name: str = "logits_b_conformal.max_possible",
    head_b_set_size_name: str = "logits_b_conformal.set_size",
    covered_mask_a_name: str = "logits_a.covered_mask",
    covered_mask_b_name: str = "logits_b.covered_mask",
    selected_compartments: tuple[str, ...] = (
        "Tumor epithelium",
        "Stroma",
    ),
    background_id_a: int = 0,
    background_id_b: int = 0,
) -> DemoPatternStatistics:
    """
    Minimal demo stats for QuPath.

    Demo interpretation:
      - argmax = standard hard prediction
      - safe   = lower/confident bound:
                 argmax == c AND conformal set is singleton containing c
      - max    = upper/possible bound:
                 (argmax == c) OR (c is in the conformal set)

    Ratios are computed relative to the non-background region of head B argmax.

    Expected inputs:
      outputs[head_a_name]           -> (Ca,H,W) logits
      outputs[head_b_name]           -> (Cb,H,W) logits
      outputs[head_b_max_name]       -> (Cb,H,W) bool / binary masks
      outputs[head_b_set_size_name]  -> (H,W) uint8

      metadata[covered_mask_a_name]  -> optional (H,W) bool
      metadata[covered_mask_b_name]  -> optional (H,W) bool
    """
    if head_a_name not in outputs:
        raise KeyError(f"Missing output '{head_a_name}'")
    if head_b_name not in outputs:
        raise KeyError(f"Missing output '{head_b_name}'")
    if head_b_max_name not in outputs:
        raise KeyError(f"Missing output '{head_b_max_name}'")
    if head_b_set_size_name not in outputs:
        raise KeyError(f"Missing output '{head_b_set_size_name}'")

    logits_a = outputs[head_a_name].cpu()
    logits_b = outputs[head_b_name].cpu()
    possible_b = outputs[head_b_max_name].bool().cpu()
    set_size_b = outputs[head_b_set_size_name].cpu()

    if logits_a.ndim != 3 or logits_b.ndim != 3:
        raise ValueError(
            f"Expected CHW logits for '{head_a_name}' and '{head_b_name}', got "
            f"{tuple(logits_a.shape)} and {tuple(logits_b.shape)}"
        )

    h, w = logits_a.shape[-2:]
    if logits_b.shape[-2:] != (h, w):
        raise ValueError(
            f"Head A and head B spatial shapes differ: "
            f"{tuple(logits_a.shape)} vs {tuple(logits_b.shape)}"
        )

    if possible_b.ndim != 3 or possible_b.shape[-2:] != (h, w):
        raise ValueError(
            f"Max-possible tensor must have shape (C,H,W), got {tuple(possible_b.shape)}"
        )

    if tuple(set_size_b.shape) != (h, w):
        raise ValueError(
            f"Set-size tensor must have shape (H,W), got {tuple(set_size_b.shape)}"
        )

    covered_mask_a = metadata.get(covered_mask_a_name)
    covered_mask_b = metadata.get(covered_mask_b_name)

    if covered_mask_a is None:
        covered_mask_a = torch.ones((h, w), dtype=torch.bool)
    else:
        covered_mask_a = covered_mask_a.bool().cpu()

    if covered_mask_b is None:
        covered_mask_b = torch.ones((h, w), dtype=torch.bool)
    else:
        covered_mask_b = covered_mask_b.bool().cpu()

    pred_a = _compute_argmax_prediction(
        logits_a,
        covered_mask=covered_mask_a,
        background_id=background_id_a,
    )
    pred_b_argmax = _compute_argmax_prediction(
        logits_b,
        covered_mask=covered_mask_b,
        background_id=background_id_b,
    )

    a_id_to_name = _make_id_to_name(head_a_labels)
    b_id_to_name = _make_id_to_name(head_b_labels)
    a_name_to_id = {name: idx for idx, name in a_id_to_name.items()}

    common_roi = covered_mask_a & covered_mask_b
    non_bg_b_mask = common_roi & (pred_b_argmax != background_id_b)
    head_b_fg_area_px = int(non_bg_b_mask.sum().item())

    patterns_summary: dict[str, PatternBoundStats] = {}
    for pat_id, pat_name in sorted(b_id_to_name.items()):
        if pat_id == background_id_b:
            continue

        if pat_id >= possible_b.shape[0]:
            raise ValueError(
                f"Pattern id {pat_id} not present in max_possible tensor with shape "
                f"{tuple(possible_b.shape)}"
            )

        argmax_mask = (pred_b_argmax == pat_id) & non_bg_b_mask

        # Lower/confident: argmax=c and singleton conformal set containing c
        safe_mask = argmax_mask & (set_size_b == 1) & possible_b[pat_id]

        # Upper/possible: argmax=c OR c in conformal set
        max_mask = argmax_mask | (possible_b[pat_id] & non_bg_b_mask)

        safe_area_px = int(safe_mask.sum().item())
        argmax_area_px = int(argmax_mask.sum().item())
        max_area_px = int(max_mask.sum().item())

        patterns_summary[pat_name] = PatternBoundStats(
            pattern_id=pat_id,
            safe_area_px=safe_area_px,
            argmax_area_px=argmax_area_px,
            max_area_px=max_area_px,
            safe_ratio=_safe_ratio(safe_area_px, head_b_fg_area_px),
            argmax_ratio=_safe_ratio(argmax_area_px, head_b_fg_area_px),
            max_ratio=_safe_ratio(max_area_px, head_b_fg_area_px),
        )

    compartments_summary: dict[str, CompartmentPatternStats] = {}
    for compartment_name in selected_compartments:
        if compartment_name not in a_name_to_id:
            continue

        comp_id = a_name_to_id[compartment_name]

        # Restrict compartment summary to head B non-background region
        compartment_mask = non_bg_b_mask & (pred_a == comp_id)
        compartment_area_px = int(compartment_mask.sum().item())

        pattern_stats: dict[str, PatternBoundStats] = {}
        for pat_id, pat_name in sorted(b_id_to_name.items()):
            if pat_id == background_id_b:
                continue

            argmax_mask = (pred_b_argmax == pat_id) & compartment_mask
            safe_mask = argmax_mask & (set_size_b == 1) & possible_b[pat_id]
            max_mask = argmax_mask | (possible_b[pat_id] & compartment_mask)

            safe_area_px = int(safe_mask.sum().item())
            argmax_area_px = int(argmax_mask.sum().item())
            max_area_px = int(max_mask.sum().item())

            pattern_stats[pat_name] = PatternBoundStats(
                pattern_id=pat_id,
                safe_area_px=safe_area_px,
                argmax_area_px=argmax_area_px,
                max_area_px=max_area_px,
                safe_ratio=_safe_ratio(safe_area_px, compartment_area_px),
                argmax_ratio=_safe_ratio(argmax_area_px, compartment_area_px),
                max_ratio=_safe_ratio(max_area_px, compartment_area_px),
            )

        compartments_summary[compartment_name] = CompartmentPatternStats(
            compartment_id=comp_id,
            area_px=compartment_area_px,
            area_um2=compartment_area_px * pixel_area_um2,
            patterns=pattern_stats,
        )

    return DemoPatternStatistics(
        head_b_foreground_area_px=head_b_fg_area_px,
        head_b_foreground_area_um2=head_b_fg_area_px * pixel_area_um2,
        patterns=patterns_summary,
        compartments=compartments_summary,
    )
