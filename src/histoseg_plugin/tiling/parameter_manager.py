import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import hashlib


@dataclass
class SegmentationParams:
    """Parameters for tissue segmentation"""
    seg_level: int = -1
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    keep_ids: str = "none"
    exclude_ids: str = "none"


@dataclass
class FilterParams:
    """Parameters for contour filtering"""
    a_t: int = 100  # area threshold (tissue)
    a_h: int = 16  # area threshold (holes)
    max_n_holes: int = 8


@dataclass
class VisualizationParams:
    """Parameters for visualization"""
    vis_level: int = -1
    line_thickness: int = 500


@dataclass
class PatchParams:
    """Parameters for patch extraction"""
    use_padding: bool = True
    contour_fn: str = "four_pt"


@dataclass
class TilingParams:
    """Complete set of tiling parameters"""
    patch_size: int = 256
    step_size: int = 256
    patch_level: int = 0
    seg_params: SegmentationParams = field(default_factory=SegmentationParams)
    filter_params: FilterParams = field(default_factory=FilterParams)
    vis_params: VisualizationParams = field(
        default_factory=VisualizationParams)
    patch_params: PatchParams = field(default_factory=PatchParams)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TilingParams':
        """Create from dictionary"""
        # Handle nested dataclasses
        seg_params = SegmentationParams(**data.get('seg_params', {}))
        filter_params = FilterParams(**data.get('filter_params', {}))
        vis_params = VisualizationParams(**data.get('vis_params', {}))
        patch_params = PatchParams(**data.get('patch_params', {}))

        return cls(patch_size=data.get('patch_size', 256),
                   step_size=data.get('step_size', 256),
                   patch_level=data.get('patch_level', 0),
                   seg_params=seg_params,
                   filter_params=filter_params,
                   vis_params=vis_params,
                   patch_params=patch_params)

    def get_hash(self) -> str:
        """Get hash of parameters for caching/comparison"""
        param_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
