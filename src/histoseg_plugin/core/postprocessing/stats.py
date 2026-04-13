from __future__ import annotations

from typing import Any

import torch

from histoseg_plugin.api.schemas import (
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
    outputs: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    head_a_labels: list[Any],
    head_b_labels: list[Any],
    head_a_name: str = "logits_a",
    head_b_name: str = "logits_b",
    head_b_safe_name: str = "logits_b_conformal.safe",
    head_b_max_name: str = "logits_b_conformal.max_possible",
    covered_mask_a_name: str = "logits_a.covered_mask",
    covered_mask_b_name: str = "logits_b.covered_mask",
    selected_compartments: tuple[str, ...] = (
        "Tumor epithelium",
        "Stroma",
        "Reactive epithelium",
    ),
    background_id_a: int = 0,
    background_id_b: int = 0,
) -> DemoPatternStatistics:
    """
    Minimal demo stats for QuPath.

    Design choices:
      - Global pattern ratios are computed relative to the non-background region
        of head B argmax.
      - Compartment-conditioned ratios are also computed only inside the same
        non-background head B argmax region.
      - For each pattern:
          safe   = area from conformal safe labels
          argmax = area from raw head B argmax
          max    = area from conformal max_possible masks
      - max ratios are allowed to overlap across classes and therefore the sum
        across classes can exceed 1.0. This is expected.

    Expected inputs:
      outputs[head_a_name]          -> (Ca,H,W) logits
      outputs[head_b_name]          -> (Cb,H,W) logits
      outputs[head_b_safe_name]     -> (H,W) long safe labels
      outputs[head_b_max_name]      -> (Cb,H,W) bool / binary masks

      metadata[covered_mask_a_name] -> optional (H,W) bool
      metadata[covered_mask_b_name] -> optional (H,W) bool
    """
    if head_a_name not in outputs:
        raise KeyError(f"Missing output '{head_a_name}'")
    if head_b_name not in outputs:
        raise KeyError(f"Missing output '{head_b_name}'")
    if head_b_safe_name not in outputs:
        raise KeyError(f"Missing output '{head_b_safe_name}'")
    if head_b_max_name not in outputs:
        raise KeyError(f"Missing output '{head_b_max_name}'")

    logits_a = outputs[head_a_name].cpu()
    logits_b = outputs[head_b_name].cpu()
    pred_b_safe = outputs[head_b_safe_name].long().cpu()
    possible_b = outputs[head_b_max_name].bool().cpu()

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

    if pred_b_safe.shape != (h, w):
        raise ValueError(
            f"Safe prediction shape mismatch: got {tuple(pred_b_safe.shape)}, "
            f"expected {(h, w)}"
        )

    if possible_b.ndim != 3 or possible_b.shape[-2:] != (h, w):
        raise ValueError(
            f"Max-possible tensor must have shape (C,H,W), got {tuple(possible_b.shape)}"
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

    # Common ROI: only pixels covered by both heads
    common_roi = covered_mask_a & covered_mask_b

    # Denominator for all pattern ratios:
    # non-background region according to head B argmax, restricted to common ROI
    non_bg_b_mask = common_roi & (pred_b_argmax != background_id_b)
    head_b_fg_area_px = int(non_bg_b_mask.sum().item())

    patterns_summary: dict[str, PatternBoundStats] = {}
    for pat_id, pat_name in sorted(b_id_to_name.items()):
        if pat_id == background_id_b:
            continue

        safe_area_px = int(((pred_b_safe == pat_id) & non_bg_b_mask).sum().item())
        argmax_area_px = int(((pred_b_argmax == pat_id) & non_bg_b_mask).sum().item())

        if pat_id >= possible_b.shape[0]:
            raise ValueError(
                f"Pattern id {pat_id} not present in max_possible tensor with shape "
                f"{tuple(possible_b.shape)}"
            )
        max_area_px = int((possible_b[pat_id] & non_bg_b_mask).sum().item())

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

        # Still restricted to non-bg head B argmax region, as requested.
        compartment_mask = non_bg_b_mask & (pred_a == comp_id)
        compartment_area_px = int(compartment_mask.sum().item())

        pattern_stats: dict[str, PatternBoundStats] = {}
        for pat_id, pat_name in sorted(b_id_to_name.items()):
            if pat_id == background_id_b:
                continue

            safe_area_px = int(
                ((pred_b_safe == pat_id) & compartment_mask).sum().item()
            )
            argmax_area_px = int(
                ((pred_b_argmax == pat_id) & compartment_mask).sum().item()
            )
            max_area_px = int((possible_b[pat_id] & compartment_mask).sum().item())

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
            patterns=pattern_stats,
        )

    return DemoPatternStatistics(
        head_b_foreground_area_px=head_b_fg_area_px,
        patterns=patterns_summary,
        compartments=compartments_summary,
    )
