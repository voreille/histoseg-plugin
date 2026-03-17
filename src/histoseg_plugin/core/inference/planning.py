from dataclasses import dataclass

import openslide


@dataclass
class ResolvedWSIInferencePlan:
    model_tile_size: int
    model_tile_mpp: float

    chosen_level: int
    chosen_level_mpp: float

    level_tile_size: int
    level_stride: int

    needs_resampling: bool


def get_slide_base_mpp(wsi: openslide.OpenSlide) -> float:
    x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
    y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if x is None or y is None:
        raise ValueError("Slide missing MPP metadata")
    return (float(x) + float(y)) / 2.0


def get_level_mpps(wsi: openslide.OpenSlide) -> list[float]:
    base_mpp = get_slide_base_mpp(wsi)
    return [base_mpp * float(ds) for ds in wsi.level_downsamples]


def within_relative_tolerance(actual: float, target: float, tol: float) -> bool:
    return abs(actual - target) / target <= tol


def resolve_wsi_inference_plan(
    *,
    wsi: openslide.OpenSlide,
    model_tile_size: int,
    model_tile_mpp: float,
    mpp_tolerance: float,
    overlap_ratio: float,
) -> ResolvedWSIInferencePlan:
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    level_mpps = get_level_mpps(wsi)

    closest_level = 0
    closest_mpp_diff = float("inf")
    for level, level_mpp in enumerate(level_mpps):
        mpp_diff = abs(level_mpp - model_tile_mpp)
        if mpp_diff < closest_mpp_diff:
            closest_level = level
            closest_mpp_diff = mpp_diff

    closest_mpp = level_mpps[closest_level]

    needs_resampling = not within_relative_tolerance(
        closest_mpp, model_tile_mpp, mpp_tolerance
    )
    if not needs_resampling:
        return ResolvedWSIInferencePlan(
            model_tile_size=model_tile_size,
            model_tile_mpp=model_tile_mpp,
            chosen_level=closest_level,
            chosen_level_mpp=closest_mpp,
            level_tile_size=model_tile_size,
            level_stride=max(1, round(model_tile_size * (1.0 - overlap_ratio))),
            needs_resampling=False,
        )

    if closest_mpp > model_tile_mpp:
        chosen_level = min(0, closest_level - 1)
    else:
        chosen_level = closest_level

    chosen_level_mpp = level_mpps[chosen_level]

    resampling_factor = model_tile_mpp / chosen_level_mpp
    level_tile_size = max(1, round(model_tile_size * resampling_factor))

    level_stride = max(1, round(level_tile_size * (1.0 - overlap_ratio)))

    return ResolvedWSIInferencePlan(
        model_tile_size=model_tile_size,
        model_tile_mpp=model_tile_mpp,
        chosen_level=chosen_level,
        chosen_level_mpp=chosen_level_mpp,
        level_tile_size=level_tile_size,
        level_stride=level_stride,
        needs_resampling=needs_resampling,
    )
