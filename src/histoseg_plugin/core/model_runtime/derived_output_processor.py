from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .contracts import DerivedOutputSpec


class DerivedOutputProcessor(ABC):
    def __init__(self, derived_name: str, spec: DerivedOutputSpec):
        self.derived_name = derived_name
        self.spec = spec

    @abstractmethod
    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class PrototypeFGMergeProcessor(DerivedOutputProcessor):
    """
    Build a final label map from:
      - foreground logits: (1, H, W)
      - pattern logits: (K, H, W)

    Modes:
      - threshold
      - probabilistic

    Output:
      - label map: (H, W), dtype=torch.long
      - background is always 0
      - foreground pattern classes are shifted by +1 relative to argmax(pattern_logits)
    """

    def __init__(self, derived_name: str, spec: DerivedOutputSpec):
        super().__init__(derived_name=derived_name, spec=spec)

        self.fg_logits_key = spec.inputs.get("fg_logits")
        self.pattern_logits_key = spec.inputs.get("pattern_logits")

        if not self.fg_logits_key:
            raise ValueError(
                f"Derived output '{derived_name}' requires inputs['fg_logits']"
            )
        if not self.pattern_logits_key:
            raise ValueError(
                f"Derived output '{derived_name}' requires inputs['pattern_logits']"
            )

        self.mode = str(spec.params.get("mode", "threshold")).lower()
        self.fg_threshold = float(spec.params.get("fg_threshold", 0.5))

        if self.mode not in {"threshold", "probabilistic"}:
            raise ValueError(
                f"Unsupported prototype_fg_merge mode='{self.mode}' "
                f"for derived output '{derived_name}'"
            )

    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.fg_logits_key not in stitched_outputs:
            raise KeyError(
                f"Missing stitched output '{self.fg_logits_key}' required by "
                f"derived output '{self.derived_name}'"
            )
        if self.pattern_logits_key not in stitched_outputs:
            raise KeyError(
                f"Missing stitched output '{self.pattern_logits_key}' required by "
                f"derived output '{self.derived_name}'"
            )

        fg_logits = stitched_outputs[self.fg_logits_key]
        pattern_logits = stitched_outputs[self.pattern_logits_key]

        _validate_single_channel_chw(fg_logits, name=self.fg_logits_key)
        _validate_chw(pattern_logits, name=self.pattern_logits_key)

        if fg_logits.shape[-2:] != pattern_logits.shape[-2:]:
            raise ValueError(
                f"Shape mismatch between '{self.fg_logits_key}' {tuple(fg_logits.shape)} "
                f"and '{self.pattern_logits_key}' {tuple(pattern_logits.shape)}"
            )

        if self.mode == "threshold":
            return self._threshold_merge(
                fg_logits=fg_logits,
                pattern_logits=pattern_logits,
                fg_threshold=self.fg_threshold,
            )

        if self.mode == "probabilistic":
            return self._probabilistic_merge(
                fg_logits=fg_logits,
                pattern_logits=pattern_logits,
            )

        raise RuntimeError(f"Unexpected mode: {self.mode}")

    @staticmethod
    def _threshold_merge(
        fg_logits: torch.Tensor,
        pattern_logits: torch.Tensor,
        fg_threshold: float,
    ) -> torch.Tensor:
        fg_prob = torch.sigmoid(fg_logits)[0]  # (H, W)
        pred_patterns = torch.argmax(pattern_logits, dim=0) + 1  # (H, W)
        pred = pred_patterns.clone()
        pred[fg_prob < fg_threshold] = 0
        return pred.to(torch.long)

    @staticmethod
    def _probabilistic_merge(
        fg_logits: torch.Tensor,
        pattern_logits: torch.Tensor,
    ) -> torch.Tensor:
        fg_prob = torch.sigmoid(fg_logits)[0]  # (H, W)
        pattern_prob = F.softmax(pattern_logits, dim=0)  # (K, H, W)

        p_bg = (1.0 - fg_prob).unsqueeze(0)  # (1, H, W)
        p_fg = fg_prob.unsqueeze(0) * pattern_prob  # (K, H, W)

        full_prob = torch.cat([p_bg, p_fg], dim=0)  # (K+1, H, W)
        pred = torch.argmax(full_prob, dim=0)  # (H, W)
        return pred.to(torch.long)


def build_derived_output_processor(
    derived_name: str,
    spec: DerivedOutputSpec,
) -> DerivedOutputProcessor:
    if spec.type == "prototype_fg_merge":
        return PrototypeFGMergeProcessor(
            derived_name=derived_name,
            spec=spec,
        )

    raise ValueError(f"Unknown derived output type='{spec.type}' for '{derived_name}'")


def _validate_chw(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor for '{name}', got {type(x)}")
    if x.ndim != 3:
        raise ValueError(
            f"Expected stitched tensor '{name}' with shape (C,H,W), got {tuple(x.shape)}"
        )
    if x.shape[0] <= 0:
        raise ValueError(f"Expected at least one channel for '{name}'")


def _validate_single_channel_chw(x: torch.Tensor, name: str) -> None:
    _validate_chw(x, name=name)
    if x.shape[0] != 1:
        raise ValueError(
            f"Expected single-channel stitched tensor '{name}', got shape {tuple(x.shape)}"
        )
