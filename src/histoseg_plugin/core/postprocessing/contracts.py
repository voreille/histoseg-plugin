from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


LayoutKind = Literal["BCHW", "BHWC", "HW", "CHW"]
TaskType = Literal["semantic_segmentation", "classification", "embedding"]
RepresentationKind = Literal[
    "logits",
    "probs",
    "embeddings",
    "labels",
    "multi_binary_masks",
]
SemanticSourceKind = Literal["static", "context_dynamic"]


class LabelSpec(BaseModel):
    id: int
    name: str
    color: str | None = None


class ClientOutputSpec(BaseModel):
    display_name: str
    task_type: TaskType
    semantic_source: SemanticSourceKind = "static"
    representation: RepresentationKind
    output_layout: LayoutKind
    exposed_to_client: bool = True

    num_classes: int | None = None
    background_id: int | None = None
    labels: list[LabelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_client_output(self) -> "ClientOutputSpec":
        if (
            self.task_type == "semantic_segmentation"
            and self.semantic_source == "static"
        ):
            if self.num_classes is None:
                raise ValueError(
                    "static semantic_segmentation client outputs require num_classes"
                )
            if self.num_classes <= 0:
                raise ValueError("num_classes must be > 0")
            if self.labels and len(self.labels) != self.num_classes:
                raise ValueError(
                    "num_classes must match number of labels for static semantic_segmentation outputs"
                )

        if self.background_id is not None and self.num_classes is not None:
            if not (0 <= self.background_id < self.num_classes):
                raise ValueError("background_id must be in [0, num_classes)")

        return self


class ProcessorArtifactRef(BaseModel):
    """
    Reference to an artifact used by a postprocessing node.
    Example:
      artifact_ref:
        kind: conformal
        path: conformal.yaml
    """

    kind: str
    path: str

    @model_validator(mode="after")
    def validate_ref(self) -> "ProcessorArtifactRef":
        if not self.kind.strip():
            raise ValueError("artifact_ref.kind must be non-empty")
        if not self.path.strip():
            raise ValueError("artifact_ref.path must be non-empty")
        return self


class DerivedOutputSpec(BaseModel):
    """
    A postprocessing node that consumes stitched outputs and produces
    either:
      - one output (output=...)
      - multiple outputs (outputs={...})
    """

    type: str
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: ProcessorArtifactRef | None = None

    output: ClientOutputSpec | None = None
    outputs: dict[str, ClientOutputSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_spec(self) -> "DerivedOutputSpec":
        if not self.type.strip():
            raise ValueError("derived output type must be non-empty")

        has_single = self.output is not None
        has_multi = len(self.outputs) > 0

        if has_single == has_multi:
            raise ValueError(
                "derived output spec must define exactly one of 'output' or 'outputs'"
            )

        return self


class PostprocessingConfig(BaseModel):
    version: str | None = None
    derived_outputs: dict[str, DerivedOutputSpec] = Field(default_factory=dict)
