from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


RuntimeKind = Literal["torchscript", "onnx", "torch"]
LayoutKind = Literal["BCHW", "BHWC"]
TaskType = Literal["semantic_segmentation", "classification", "embedding"]
RepresentationKind = Literal["logits", "probs", "embeddings"]
DTypeKind = Literal["float32", "float16", "bfloat16"]


class LabelSpec(BaseModel):
    id: int
    name: str
    color: str | None = None


class OutputGeometrySpec(BaseModel):
    output_stride: int = 1
    output_size: tuple[int, int] | None = None
    valid_size: tuple[int, int] | None = None
    valid_offset: tuple[int, int] = (0, 0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "OutputGeometrySpec":
        if self.output_stride <= 0:
            raise ValueError("output_stride must be > 0")

        oy, ox = self.valid_offset
        if oy < 0 or ox < 0:
            raise ValueError("valid_offset values must be >= 0")

        if self.output_size is not None:
            oh, ow = self.output_size
            if oh <= 0 or ow <= 0:
                raise ValueError("output_size values must be > 0")

        if self.valid_size is not None:
            vh, vw = self.valid_size
            if vh <= 0 or vw <= 0:
                raise ValueError("valid_size values must be > 0")

        if self.output_size is not None and self.valid_size is not None:
            oh, ow = self.output_size
            vh, vw = self.valid_size
            oy, ox = self.valid_offset

            if vh > oh or vw > ow:
                raise ValueError("valid_size must be <= output_size")

            if oy + vh > oh or ox + vw > ow:
                raise ValueError("valid_offset + valid_size must fit inside output_size")

        return self


class OutputHeadSpec(BaseModel):
    display_name: str
    task_type: TaskType
    representation: RepresentationKind = "logits"
    output_layout: LayoutKind = "BCHW"

    num_classes: int | None = None
    background_id: int | None = None
    labels: list[LabelSpec] = Field(default_factory=list)

    geometry: OutputGeometrySpec = Field(default_factory=OutputGeometrySpec)

    @model_validator(mode="after")
    def validate_head(self) -> "OutputHeadSpec":
        if self.task_type == "semantic_segmentation":
            if self.num_classes is None:
                raise ValueError("semantic_segmentation heads require num_classes")
            if self.num_classes <= 0:
                raise ValueError("num_classes must be > 0")
            if len(self.labels) != self.num_classes:
                raise ValueError(
                    "num_classes must match number of labels for semantic_segmentation heads"
                )

        if self.background_id is not None:
            if self.num_classes is None:
                raise ValueError("background_id requires num_classes to be defined")
            if not (0 <= self.background_id < self.num_classes):
                raise ValueError("background_id must be in [0, num_classes)")

        return self


class PreprocessingSpec(BaseModel):
    id: str
    config: dict[str, Any] = Field(default_factory=dict)


class InputSpec(BaseModel):
    tile_size: int
    tile_mpp: float
    mpp_tolerance: float | None = None
    num_channels: int
    layout: LayoutKind = "BCHW"
    dtype: DTypeKind = "float32"
    preprocessing: PreprocessingSpec | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "InputSpec":
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if self.tile_mpp <= 0:
            raise ValueError("tile_mpp must be > 0")
        if self.mpp_tolerance is not None and self.mpp_tolerance < 0:
            raise ValueError("mpp_tolerance must be >= 0")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        return self


class InferenceSpec(BaseModel):
    runtime: RuntimeKind
    weights: str
    use_amp: bool = False
    amp_dtype: DTypeKind | None = None
    preferred_device: str | None = None

    @model_validator(mode="after")
    def validate_inference(self) -> "InferenceSpec":
        if self.use_amp and self.amp_dtype is None:
            # allowed, but explicit dtype is usually clearer
            return self
        return self


class TorchModelSpec(BaseModel):
    factory: str
    init_args: dict[str, Any] = Field(default_factory=dict)
    adapter_factory: str | None = None


class CapabilitySpec(BaseModel):
    context_inference: bool = False
    support_prototype_fitting: bool = False


class TrainingSpec(BaseModel):
    architecture_id: str | None = None
    git_commit: str | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)
    checkpoint: str | None = None


class ModelManifest(BaseModel):
    name: str
    version: str

    inference: InferenceSpec
    input: InputSpec
    output: dict[str, OutputHeadSpec]

    model: TorchModelSpec | None = None
    capabilities: CapabilitySpec = Field(default_factory=CapabilitySpec)
    training: TrainingSpec | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ModelManifest":
        if not self.output:
            raise ValueError("output must contain at least one head")

        if self.inference.runtime == "torch":
            if self.model is None:
                raise ValueError("runtime='torch' requires a model section")
        else:
            if self.model is not None:
                raise ValueError("model section is only valid when runtime='torch'")

        if (
            self.capabilities.context_inference
            or self.capabilities.support_prototype_fitting
        ):
            if self.inference.runtime != "torch":
                raise ValueError(
                    "context/prototype capabilities currently require runtime='torch'"
                )
            if self.model is None:
                raise ValueError(
                    "context/prototype capabilities require a torch model section"
                )
            if self.model.adapter_factory is None:
                raise ValueError(
                    "context/prototype capabilities require model.adapter_factory"
                )

        return self