from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


RuntimeKind = Literal["torchscript", "onnx"]


class LabelSpec(BaseModel):
    id: int
    name: str
    color: Optional[str] = None


class OutputHeadSpec(BaseModel):
    display_name: str
    background_id: int = 0
    labels: list[LabelSpec]
    num_classes: int
    output_layout: Literal["BCHW", "BHWC"]
    task_type: Literal["semantic_segmentation", "classification", "embedding"]
    representation: Literal["logits", "probs"] = "logits"


class InputSpec(BaseModel):
    tile_size: int
    tile_mpp: float
    mpp_tolerance: Optional[float] = None
    num_channels: int
    layout: Literal["BCHW", "BHWC"]
    dtype: str
    preprocessing: Optional[PreprocessingSpec] = None


class PreprocessingSpec(BaseModel):
    id: str
    config: dict = {}


class InferenceSpec(BaseModel):
    runtime: RuntimeKind
    weights: str
    use_amp: Optional[bool] = False
    amp_dtype: Optional[str] = None
    preferred_device: Optional[str] = None


class TrainingSpec(BaseModel):
    architecture_id: Optional[str] = None
    git_commit: Optional[str] = None
    init_args: dict = {}
    checkpoint: Optional[str] = None


class ModelManifest(BaseModel):
    name: str
    version: str

    inference: InferenceSpec
    input: InputSpec
    output: dict[str, OutputHeadSpec]
    training: Optional[TrainingSpec] = None
