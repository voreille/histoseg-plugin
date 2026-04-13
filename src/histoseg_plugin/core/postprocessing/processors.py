from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .base import DerivedOutputProcessor, ProcessorResult
from .contracts import DerivedOutputSpec


SAFE_AMBIGUOUS_ID = 255


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

        if spec.output is None:
            raise ValueError(
                f"prototype_fg_merge requires a single 'output' spec for '{derived_name}'"
            )

    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
        aux: dict[str, Any] | None = None,
    ) -> ProcessorResult:
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
            pred = self._threshold_merge(
                fg_logits=fg_logits,
                pattern_logits=pattern_logits,
                fg_threshold=self.fg_threshold,
            )
        else:
            pred = self._probabilistic_merge(
                fg_logits=fg_logits,
                pattern_logits=pattern_logits,
            )

        return ProcessorResult(outputs={self.derived_name: pred})

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


class ConformalSetFromLogitsProcessor(DerivedOutputProcessor):
    """
    Multi-output conformal processor operating on stitched logits (C, H, W).

    Expected input:
      - stitched_outputs[self.logits_key]: Tensor (C, H, W)

    Optional aux input:
      - f"{self.logits_key}.covered_mask": Tensor (H, W) bool

    Possible derived outputs:
      - <derived_name>.safe             : (H, W) long
      - <derived_name>.max_possible     : (C, H, W) bool
      - <derived_name>.set_size         : (H, W) uint8
      - <derived_name>.empty_set_mask   : (H, W) bool
      - <derived_name>.multi_class_mask : (H, W) bool

    Metadata:
      - <derived_name>.pred_argmax
      - <derived_name>.apply_conformal_mask
    """

    def __init__(
        self,
        derived_name: str,
        spec: DerivedOutputSpec,
        *,
        base_dir: Path | None = None,
    ):
        super().__init__(derived_name=derived_name, spec=spec)

        self.logits_key = spec.inputs.get("logits")
        if not self.logits_key:
            raise ValueError(
                f"Derived output '{derived_name}' requires inputs['logits']"
            )

        if not spec.outputs:
            raise ValueError(
                f"conformal_set_from_logits requires non-empty 'outputs' for "
                f"'{derived_name}'"
            )

        self.apply_only_on_non_background = bool(
            spec.params.get("apply_only_on_non_background", True)
        )
        self.empty_set_policy = str(
            spec.params.get("empty_set_policy", "background")
        ).lower()
        self.multiclass_policy = str(
            spec.params.get("multiclass_policy", "special_label")
        ).lower()

        if self.empty_set_policy not in {"background"}:
            raise ValueError(
                f"Unsupported empty_set_policy='{self.empty_set_policy}' "
                f"for '{derived_name}'"
            )

        if self.multiclass_policy not in {"special_label", "background"}:
            raise ValueError(
                f"Unsupported multiclass_policy='{self.multiclass_policy}' "
                f"for '{derived_name}'"
            )

        if spec.artifact_ref is None:
            raise ValueError(
                f"conformal_set_from_logits requires artifact_ref for '{derived_name}'"
            )

        if base_dir is None:
            raise ValueError(
                f"conformal_set_from_logits requires base_dir to resolve artifact path "
                f"for '{derived_name}'"
            )

        self.conformal_path = (base_dir / spec.artifact_ref.path).resolve()
        self.thresholds_by_head = self._load_thresholds(self.conformal_path)

        self.requested_outputs = set(spec.outputs.keys())
        self._validate_requested_outputs()

        self.safe_output_spec = spec.outputs.get("safe")
        self.max_possible_output_spec = spec.outputs.get("max_possible")

        self.background_id = self._infer_background_id()
        self.ambiguous_label_id = self._infer_ambiguous_label_id()

    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
        aux: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        logits = self._get_logits(stitched_outputs)
        c, h, w = logits.shape

        probs = F.softmax(logits, dim=0)
        pred_argmax = torch.argmax(probs, dim=0).to(torch.long)

        thresholds = self._get_thresholds_for_head(
            head_name=self.logits_key,
            num_classes=c,
            bg_idx=self.background_id,
            device=logits.device,
        )

        covered_mask = self._get_covered_mask(
            aux=aux,
            expected_hw=(h, w),
            device=logits.device,
        )

        apply_conformal_mask = covered_mask.clone()
        if self.apply_only_on_non_background:
            apply_conformal_mask &= pred_argmax != self.background_id

        possible = self._compute_possible_sets(
            probs=probs,
            thresholds=thresholds,
            apply_conformal_mask=apply_conformal_mask,
        )
        set_size = possible.sum(dim=0).to(torch.uint8)

        singleton_mask = apply_conformal_mask & (set_size == 1)
        empty_set_mask = apply_conformal_mask & (set_size == 0)
        multi_class_mask = apply_conformal_mask & (set_size > 1)

        pred_safe = self._build_safe_prediction(
            pred_argmax=pred_argmax,
            possible=possible,
            covered_mask=covered_mask,
            apply_conformal_mask=apply_conformal_mask,
            singleton_mask=singleton_mask,
            empty_set_mask=empty_set_mask,
            multi_class_mask=multi_class_mask,
        )

        outputs: dict[str, torch.Tensor] = {}
        metadata: dict[str, Any] = {
            f"{self.derived_name}.pred_argmax": pred_argmax.cpu(),
            f"{self.derived_name}.apply_conformal_mask": apply_conformal_mask.cpu(),
        }

        if "safe" in self.requested_outputs:
            outputs[f"{self.derived_name}.safe"] = pred_safe.cpu()

        if "max_possible" in self.requested_outputs:
            outputs[f"{self.derived_name}.max_possible"] = possible.cpu()

        if "set_size" in self.requested_outputs:
            outputs[f"{self.derived_name}.set_size"] = set_size.cpu()

        if "empty_set_mask" in self.requested_outputs:
            outputs[f"{self.derived_name}.empty_set_mask"] = empty_set_mask.cpu()

        if "multi_class_mask" in self.requested_outputs:
            outputs[f"{self.derived_name}.multi_class_mask"] = multi_class_mask.cpu()

        return ProcessorResult(outputs=outputs, metadata=metadata)

    def _validate_requested_outputs(self) -> None:
        allowed = {
            "safe",
            "max_possible",
            "set_size",
            "empty_set_mask",
            "multi_class_mask",
        }
        unknown = self.requested_outputs - allowed
        if unknown:
            raise ValueError(
                f"Unknown conformal outputs requested for '{self.derived_name}': "
                f"{sorted(unknown)}"
            )

        if (
            self.multiclass_policy == "special_label"
            and "safe" not in self.requested_outputs
        ):
            return

        if (
            self.multiclass_policy == "special_label"
            and self.spec.outputs.get("safe") is None
        ):
            raise ValueError(
                f"multiclass_policy='special_label' requires a 'safe' output spec "
                f"for '{self.derived_name}'"
            )

    def _get_logits(self, stitched_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.logits_key not in stitched_outputs:
            raise KeyError(
                f"Missing stitched output '{self.logits_key}' required by "
                f"derived output '{self.derived_name}'"
            )

        logits = stitched_outputs[self.logits_key]
        _validate_chw(logits, name=self.logits_key)
        return logits

    def _get_covered_mask(
        self,
        *,
        aux: dict[str, Any] | None,
        expected_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if aux is None:
            return torch.ones(expected_hw, dtype=torch.bool, device=device)

        covered_mask = aux.get(f"{self.logits_key}.covered_mask")
        if covered_mask is None:
            return torch.ones(expected_hw, dtype=torch.bool, device=device)

        if not torch.is_tensor(covered_mask):
            raise TypeError(
                f"covered_mask for '{self.logits_key}' must be a tensor, "
                f"got {type(covered_mask)}"
            )

        if tuple(covered_mask.shape) != expected_hw:
            raise ValueError(
                f"covered_mask for '{self.logits_key}' has shape "
                f"{tuple(covered_mask.shape)} but expected {expected_hw}"
            )

        return covered_mask.to(device=device, dtype=torch.bool)

    def _compute_possible_sets(
        self,
        *,
        probs: torch.Tensor,
        thresholds: torch.Tensor,
        apply_conformal_mask: torch.Tensor,
    ) -> torch.Tensor:
        c = probs.shape[0]
        tau = 1.0 - thresholds
        valid_cls = ~torch.isnan(tau)

        possible = torch.zeros_like(probs, dtype=torch.bool)
        for cls_id in range(c):
            if not bool(valid_cls[cls_id]):
                continue
            possible[cls_id] = apply_conformal_mask & (probs[cls_id] >= tau[cls_id])

        return possible

    def _build_safe_prediction(
        self,
        *,
        pred_argmax: torch.Tensor,
        possible: torch.Tensor,
        covered_mask: torch.Tensor,
        apply_conformal_mask: torch.Tensor,
        singleton_mask: torch.Tensor,
        empty_set_mask: torch.Tensor,
        multi_class_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_safe = pred_argmax.clone()
        pred_safe[~covered_mask] = self.background_id

        default_fill_id = self.background_id
        if self.multiclass_policy == "special_label":
            if self.ambiguous_label_id is None:
                raise ValueError(
                    f"multiclass_policy='special_label' requires an ambiguous label "
                    f"in the 'safe' output spec for '{self.derived_name}'"
                )
            default_fill_id = self.ambiguous_label_id

        pred_safe[apply_conformal_mask] = default_fill_id

        if singleton_mask.any():
            pred_safe[singleton_mask] = torch.argmax(
                possible[:, singleton_mask].float(), dim=0
            ).to(torch.long)

        if self.empty_set_policy == "background":
            pred_safe[empty_set_mask] = self.background_id

        if self.multiclass_policy == "background":
            pred_safe[multi_class_mask] = self.background_id
        elif self.multiclass_policy == "special_label":
            pred_safe[multi_class_mask] = self.ambiguous_label_id

        return pred_safe

    def _infer_background_id(self) -> int:
        for key in ("safe", "max_possible"):
            out_spec = self.spec.outputs.get(key)
            if out_spec is not None and out_spec.background_id is not None:
                return int(out_spec.background_id)
        return 0

    def _infer_ambiguous_label_id(self) -> int | None:
        """
        For multiclass_policy='special_label', infer the class id to use for
        ambiguous conformal sets from the 'safe' output spec.

        Current convention:
          - look for a label named 'multi_class_conformal_set'
          - if absent, allow params['ambiguous_label_id']
        """
        if self.multiclass_policy != "special_label":
            return None

        manual_id = self.spec.params.get("ambiguous_label_id")
        if manual_id is not None:
            return int(manual_id)

        safe_spec = self.safe_output_spec
        if safe_spec is None:
            return None

        for label in safe_spec.labels:
            if label.name == "multi_class_conformal_set":
                return int(label.id)

        return None

    @staticmethod
    def _load_thresholds(path: Path) -> dict[str, list[float | None]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing conformal artifact: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        heads = raw.get("heads")
        if not isinstance(heads, dict):
            raise ValueError(f"Invalid conformal yaml at {path}: missing 'heads' dict")

        thresholds_by_head: dict[str, list[float | None]] = {}
        for head_name, head_cfg in heads.items():
            thresholds = head_cfg.get("thresholds")
            if thresholds is None:
                continue
            if not isinstance(thresholds, list):
                raise ValueError(
                    f"Invalid conformal yaml at {path}: thresholds for head "
                    f"'{head_name}' must be a list"
                )
            thresholds_by_head[head_name] = [
                None if t is None else float(t) for t in thresholds
            ]

        return thresholds_by_head

    def _get_thresholds_for_head(
        self,
        *,
        head_name: str,
        num_classes: int,
        bg_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        if head_name not in self.thresholds_by_head:
            raise KeyError(
                f"Conformal artifact does not define thresholds for head '{head_name}'"
            )

        vals = self.thresholds_by_head[head_name]
        if len(vals) != num_classes:
            raise ValueError(
                f"Threshold length mismatch for head '{head_name}': got {len(vals)}, "
                f"expected {num_classes}"
            )

        out = torch.tensor(
            [float("nan") if t is None else float(t) for t in vals],
            dtype=torch.float32,
            device=device,
        )

        if 0 <= bg_idx < num_classes:
            out[bg_idx] = float("nan")

        return out
