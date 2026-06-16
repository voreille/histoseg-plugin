from contextlib import contextmanager
from pathlib import Path

import openslide


@contextmanager
def open_wsi(path: Path):
    try:
        wsi = openslide.OpenSlide(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open WSI at {path}: {e}") from e
    try:
        yield wsi
    finally:
        wsi.close()


def _tiff_resolution_to_mpp(
    resolution: str | float,
    resolution_unit: str,
) -> float:
    """Convert TIFF resolution metadata to micrometres per pixel."""

    pixels_per_unit = float(resolution)
    if pixels_per_unit <= 0:
        raise ValueError(f"Invalid TIFF resolution: {pixels_per_unit}")

    unit = resolution_unit.strip().lower()

    if unit in {"centimeter", "centimetre", "cm", "3"}:
        micrometres_per_unit = 10_000.0
    elif unit in {"inch", "inches", "in", "2"}:
        micrometres_per_unit = 25_400.0
    else:
        raise ValueError(f"Unsupported TIFF resolution unit: {resolution_unit!r}")

    return micrometres_per_unit / pixels_per_unit


def get_slide_base_mpp(wsi: openslide.OpenSlide) -> float:
    """Return the mean level-0 MPP from OpenSlide or TIFF metadata."""

    properties = wsi.properties

    # Preferred OpenSlide metadata.
    mpp_x = properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is not None and mpp_y is not None:
        return (float(mpp_x) + float(mpp_y)) / 2.0

    # Fallback for generic TIFF files.
    resolution_unit = properties.get("tiff.ResolutionUnit")
    x_resolution = properties.get("tiff.XResolution")
    y_resolution = properties.get("tiff.YResolution")

    if (
        resolution_unit is not None
        and x_resolution is not None
        and y_resolution is not None
    ):
        tiff_mpp_x = _tiff_resolution_to_mpp(
            x_resolution,
            resolution_unit,
        )
        tiff_mpp_y = _tiff_resolution_to_mpp(
            y_resolution,
            resolution_unit,
        )
        return (tiff_mpp_x + tiff_mpp_y) / 2.0

    raise ValueError(
        "Slide is missing usable MPP metadata. Expected either "
        "'openslide.mpp-x'/'openslide.mpp-y' or TIFF "
        "'tiff.XResolution'/'tiff.YResolution'/'tiff.ResolutionUnit'."
    )


def get_level_mpps(wsi: openslide.OpenSlide) -> list[float]:
    base_mpp = get_slide_base_mpp(wsi)
    return [base_mpp * float(downsample) for downsample in wsi.level_downsamples]
