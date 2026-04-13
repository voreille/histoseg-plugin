from dataclasses import dataclass

from histoseg_plugin.core.model_runtime.base import BaseModelRunner
from histoseg_plugin.core.postprocessing.pipeline import WSIPostprocessor


@dataclass
class InferenceBundle:
    model_runner: BaseModelRunner
    postprocessor: WSIPostprocessor | None = None
