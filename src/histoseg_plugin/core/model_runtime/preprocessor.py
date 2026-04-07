from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch


class BasePreprocessor(ABC):
    """
    Base interface for optional model-specific preprocessing.

    Expected input/output:
        - torch.Tensor
        - shape: (B, C, H, W)
        - RGB
        - float32
        - values in [0, 1]
    """

    def __init__(self, model_dir: Path, config: Mapping[str, Any] | None = None):
        self.model_dir = Path(model_dir)
        self.config = dict(config or {})

    @abstractmethod
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReinhardPreprocessor(BasePreprocessor):
    """
    Reinhard stain normalization on BCHW float32 RGB tensors in [0, 1].

    Config:
        reference_image: str   # required, relative to model_dir or absolute path
        std_sum_threshold: float = 5.0
        eps: float = 1e-8
    """

    def __init__(self, model_dir: Path, config: Mapping[str, Any] | None = None):
        super().__init__(model_dir=model_dir, config=config)

        reference_image = self.config.get("reference_image")
        if not reference_image:
            raise ValueError("ReinhardPreprocessor requires config['reference_image']")

        self.reference_image_path = _resolve_asset_path(
            model_dir=self.model_dir,
            path_str=reference_image,
        )
        self.std_sum_threshold = float(self.config.get("std_sum_threshold", 5.0))
        self.eps = float(self.config.get("eps", 1e-8))

        self.mt, self.stdt = self._load_reference_stats(self.reference_image_path)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        _validate_batch(batch)

        batch_np = batch.detach().cpu().numpy().astype(np.float32)  # BCHW
        out = np.empty_like(batch_np, dtype=np.float32)

        for i in range(batch_np.shape[0]):
            img_chw = batch_np[i]  # CHW
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # HWC RGB

            if np.any(img_hwc):
                img_hwc = self._norm_reinhard(
                    source_image=img_hwc,
                    mt=self.mt,
                    stdt=self.stdt,
                    std_sum_threshold=self.std_sum_threshold,
                    eps=self.eps,
                )

            img_hwc = np.clip(img_hwc, 0.0, 1.0)
            out[i] = np.transpose(img_hwc, (2, 0, 1))  # back to CHW

        return torch.from_numpy(out)

    @staticmethod
    def _load_reference_stats(
        reference_image_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise FileNotFoundError(
                f"Could not read reference image: {reference_image_path}"
            )

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2Lab)

        mt = np.mean(ref_lab, axis=(0, 1)).astype(np.float32)
        stdt = np.std(ref_lab, axis=(0, 1)).astype(np.float32)
        return mt, stdt

    @staticmethod
    def _norm_reinhard(
        source_image: np.ndarray,
        mt: np.ndarray,
        stdt: np.ndarray,
        std_sum_threshold: float,
        eps: float,
    ) -> np.ndarray:
        """
        source_image: HWC RGB float32 in [0,1]
        returns: HWC RGB float32
        """
        source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2Lab)
        ms = np.mean(source_lab, axis=(0, 1)).astype(np.float32)
        stds = np.std(source_lab, axis=(0, 1)).astype(np.float32)

        if float(np.sum(stds)) <= std_sum_threshold:
            return source_image

        norm_lab = source_lab.copy()
        for c in range(3):
            norm_lab[:, :, c] = (
                (norm_lab[:, :, c] - ms[c]) * (stdt[c] / (stds[c] + eps))
            ) + mt[c]

        norm_image = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2RGB)
        return norm_image.astype(np.float32)


class NormalizePreprocessor(BasePreprocessor):
    """
    Channel-wise mean/std normalization.

    Input:
        - BCHW
        - float32
        - values in [0, 1]

    Output:
        - BCHW
        - normalized: (x - mean) / std
    """

    def __init__(self, model_dir: Path, config: Mapping[str, Any] | None = None):
        super().__init__(model_dir=model_dir, config=config)

        mean = self.config.get("mean")
        std = self.config.get("std")

        if mean is None or std is None:
            raise ValueError("NormalizePreprocessor requires 'mean' and 'std'")

        if len(mean) != len(std):
            raise ValueError("mean and std must have same length")

        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

        if torch.any(self.std <= 0):
            raise ValueError("std must be > 0")

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        _validate_batch(batch)

        mean = self.mean.to(batch.device)
        std = self.std.to(batch.device)

        return (batch - mean) / std


def build_preprocessor(model_dir: Path, spec: Any) -> BasePreprocessor:
    if spec.id == "reinhard_v1":
        return ReinhardPreprocessor(model_dir=model_dir, config=spec.config)

    if spec.id == "normalize_v1":
        return NormalizePreprocessor(model_dir=model_dir, config=spec.config)

    raise ValueError(f"Unknown preprocessor id: {spec.id}")


def _resolve_asset_path(model_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (model_dir / path).resolve()


def _validate_batch(batch: torch.Tensor) -> None:
    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(batch)}")

    if batch.ndim != 4:
        raise ValueError(f"Expected batch of shape (B,C,H,W), got {tuple(batch.shape)}")

    if batch.shape[1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got {batch.shape[1]}")

    if not batch.is_floating_point():
        raise TypeError(f"Expected floating point tensor, got {batch.dtype}")
