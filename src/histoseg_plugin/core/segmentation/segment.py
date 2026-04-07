# TODO: handle opening wsi better
# TODO: check when tiles need resampling how the output is treated, cause mpp will change
# TODO: check how the geometry is handled, maybe add a Stitcher whose job is to stitch tiles together and handle all the geometry, and just feed it tiles + logits + coords + meta, and it handles the rest (including final averaging). That way we can reuse it for other things like stitching support logits for prototypes.
# TODO: add Hann weighting as an option for stitching 


import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Callable

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from histoseg_plugin.core.model_runtime.base import BaseModelRunner


class TileDataset(Dataset):
    def __init__(
        self,
        wsi_path: str | Path,
        coords: np.ndarray,
        tile_level: int,
        tile_size: int,
        transforms: Optional[Callable] = None,
    ):
        self.wsi_path = str(wsi_path)
        self.coords = coords
        self.tile_size_lvl = tile_size
        self.tile_level = tile_level
        self.transforms = transforms or T.Compose([T.ToTensor()])
        self._wsi = None  # opened lazily in each worker

    def _get_wsi(self) -> openslide.OpenSlide:
        if self._wsi is None:
            self._wsi = openslide.OpenSlide(self.wsi_path)
        return self._wsi

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wsi = self._get_wsi()
        x0, y0 = self.coords[idx, :]
        tile = wsi.read_region(
            (int(x0), int(y0)),
            self.tile_level,
            (self.tile_size_lvl, self.tile_size_lvl),
        ).convert("RGB")
        tile = self.transforms(tile)
        return tile, torch.tensor([x0, y0], dtype=torch.int64)

    def __del__(self):
        if self._wsi is not None:
            self._wsi.close()


def get_mpp(wsi: openslide.OpenSlide) -> float:
    mpp_x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if mpp_x is None or mpp_y is None:
        raise ValueError("WSI is missing MPP properties (openslide.mpp-x / mpp-y).")
    if float(mpp_x) != float(mpp_y):
        raise ValueError(
            f"Non-square pixels not supported (mpp_x={mpp_x}, mpp_y={mpp_y})."
        )
    return float(mpp_x)


# -------------------------
# Stitching helpers
# -------------------------
def bbox_level0_to_mask_rect(
    x0: int,
    y0: int,
    extent0_w: float,
    extent0_h: float,
    fx: float,
    fy: float,
) -> Tuple[int, int, int, int]:
    """
    Map a level-0 bbox [x0, x1) x [y0, y1) to a mask pixel rectangle.
    floor for start, ceil for end -> stable tessellation.
    Returns (x0m, y0m, x1m, y1m) in mask pixels (end exclusive).
    """
    x1 = x0 + extent0_w
    y1 = y0 + extent0_h

    x0m = int(math.floor(x0 / fx))
    y0m = int(math.floor(y0 / fy))
    x1m = int(math.ceil(x1 / fx))
    y1m = int(math.ceil(y1 / fy))

    if x1m <= x0m:
        x1m = x0m + 1
    if y1m <= y0m:
        y1m = y0m + 1

    return x0m, y0m, x1m, y1m


def accumulate_tile_into_canvas(
    sum_map: torch.Tensor,  # (C,H,W) float32 CPU
    w_map: torch.Tensor,  # (H,W) float32 CPU
    tile_logits: torch.Tensor,  # (C,h,w) float32 CPU
    x0m: int,
    y0m: int,
    x1m: int,
    y1m: int,
) -> None:
    """
    Adds tile_logits into sum_map at the rectangle, with uniform weights.
    Clips safely at borders by cropping both target rect and tile.
    """
    C, H, W = sum_map.shape

    tx0 = max(0, x0m)
    ty0 = max(0, y0m)
    tx1 = min(W, x1m)
    ty1 = min(H, y1m)
    if tx1 <= tx0 or ty1 <= ty0:
        return

    ox0 = tx0 - x0m
    oy0 = ty0 - y0m
    ox1 = ox0 + (tx1 - tx0)
    oy1 = oy0 + (ty1 - ty0)

    tile_crop = tile_logits[:, oy0:oy1, ox0:ox1]
    sum_map[:, ty0:ty1, tx0:tx1] += tile_crop
    w_map[ty0:ty1, tx0:tx1] += 1.0


def finalize_canvas(
    sum_map: torch.Tensor, w_map: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    w = torch.clamp(w_map, min=eps)
    return sum_map / w.unsqueeze(0)


@dataclass
class StitchMeta:
    mask_h: int
    mask_w: int
    level0_w: int
    level0_h: int
    fx: float
    fy: float
    output_target_mpp: float
    tile_level: int
    tile_size_lvl: int
    tile_extent0: float


@dataclass
class StitchResult:
    avg_logits_by_head: Dict[str, torch.Tensor]  # each (C,H,W) float32 CPU
    weight_map: Optional[torch.Tensor]  # (H,W) float32 CPU
    meta: StitchMeta


def run_model_and_stitch_logits(
    *,
    slide_path: str,
    coords: np.ndarray,
    level_tile_size: int,
    tile_level: int,
    model_runner: BaseModelRunner,
    output_target_mpp: float = 4.0,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    return_weight_map: bool = True,
    resample: Optional[bool] = None,
    model_tile_size: Optional[int] = None,
) -> StitchResult:
    """
    Runs tile inference and stitches logits on a mask grid corresponding to output_target_mpp.

    - coords: list of (x0,y0) at level0 (tile top-left)
    - attrs: must contain patch_size, patch_level (OpenSlide read_region args)
    - model output: Tensor (B,C,H,W) or dict[str, Tensor] with (B,C,H,W)
    - returns stitched avg logits per head (CPU float32), and optional weight map.
    """
    if coords.shape[0] == 0:
        raise ValueError("coords is empty (no tiles).")

    device = (
        model_runner.device if hasattr(model_runner, "device") else torch.device("cpu")
    )

    if pin_memory is None:
        pin_memory = device.type == "cuda"

    # model_runner = model_runner.to(device).eval()

    wsi = openslide.OpenSlide(slide_path)
    mpp = get_mpp(wsi=wsi)
    level0_w, level0_h = wsi.level_dimensions[0]

    # mask grid corresponds to physical resolution output_target_mpp
    mask_h = int(math.ceil(level0_h * mpp / output_target_mpp))
    mask_w = int(math.ceil(level0_w * mpp / output_target_mpp))

    # level-0 pixels per mask pixel
    fx = level0_w / mask_w
    fy = level0_h / mask_h

    if resample:
        # TODO: add a check for model tile size
        transforms = T.Compose(
            [
                T.Resize(
                    size=(model_tile_size, model_tile_size),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
            ]
        )
    else:
        transforms = T.ToTensor()

    ds = TileDataset(
        wsi_path=slide_path,
        coords=coords,
        tile_size=level_tile_size,
        tile_level=tile_level,
        transforms=transforms,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # tile footprint in level-0 pixels 
    # TODO: if resampling, this will be different from the actual tile size fed to the model; check if that causes any issues with how we stitch (it shouldn't, but good to verify). We just need to make sure fx/fy are correct for the tile footprint in level-0 pixels, which is what determines where the logits go on the mask.
    level_downsample = float(wsi.level_downsamples[ds.tile_level])
    tile_extent0 = ds.tile_size_lvl * level_downsample  # assumes square tiles

    wsi.close() # needed since I pass the slide path to each worker and they open it lazily; we don't want the main process holding an open handle to the slide since it won't be used for reading and could cause issues on some filesystems if too many handles are open. Each worker will open its own handle when needed and close it when done (handled by TileDataset.__del__).

    # Allocate shared weight map once
    w_map = (
        torch.zeros((mask_h, mask_w), dtype=torch.float32)
        if return_weight_map
        else None
    )

    # We allocate sum maps lazily per head after first batch (need C)
    sum_maps: Dict[str, torch.Tensor] = {}

    for tiles, xy0 in loader:
        tiles = tiles.to(device, non_blocking=True)

        out_dict = model_runner.predict_tiles(tiles)

        # validate shapes + allocate sum_maps as needed
        for head, logits in out_dict.items():
            if logits.ndim != 4:
                raise ValueError(
                    f"Expected {head} logits (B,C,H,W), got {tuple(logits.shape)}"
                )
            if head not in sum_maps:
                C = int(logits.shape[1])
                sum_maps[head] = torch.zeros((C, mask_h, mask_w), dtype=torch.float32)

        # Move to CPU once per head (keeps stitching stable, predictable memory)
        out_cpu = {
            k: v.detach().float().cpu() for k, v in out_dict.items()
        }  # (B,C,h,w)
        xy0_cpu = xy0.cpu().tolist()  # [[x0,y0], ...]

        for b, (x0, y0) in enumerate(xy0_cpu):
            x0m, y0m, x1m, y1m = bbox_level0_to_mask_rect(
                x0=int(x0),
                y0=int(y0),
                extent0_w=tile_extent0,
                extent0_h=tile_extent0,
                fx=fx,
                fy=fy,
            )
            out_w = x1m - x0m
            out_h = y1m - y0m

            # weights are the same for all heads; increment once per tile
            if w_map is not None:
                # we increment weights inside accumulate_tile_into_canvas, but it expects logits crop.
                # easiest: increment via a fake single-channel add? No: just use the same call once.
                # We'll increment w_map when processing the first head only.
                pass

            first_head = True
            for head, logits_cpu in out_cpu.items():
                # resize this tile's logits to exactly match mask rectangle size
                tile_logits = F.interpolate(
                    logits_cpu[b : b + 1],  # (1,C,h,w)
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]  # (C,out_h,out_w)

                # For weights: increment only once using the first processed head
                if w_map is not None and first_head:
                    accumulate_tile_into_canvas(
                        sum_map=sum_maps[head],
                        w_map=w_map,
                        tile_logits=tile_logits,
                        x0m=x0m,
                        y0m=y0m,
                        x1m=x1m,
                        y1m=y1m,
                    )
                    first_head = False
                else:
                    # accumulate without touching w_map
                    # (we reuse the same function but with a dummy w_map view)
                    if w_map is None:
                        dummy_w = torch.zeros((1, 1), dtype=torch.float32)  # never used
                        accumulate_tile_into_canvas(
                            sum_map=sum_maps[head],
                            w_map=dummy_w,  # ignored effectively because regions won’t overlap (still safe)
                            tile_logits=tile_logits,
                            x0m=x0m,
                            y0m=y0m,
                            x1m=x1m,
                            y1m=y1m,
                        )
                    else:
                        # Create a lightweight "do-not-update" weight map by slicing zero area is messy.
                        # Better: just add logits manually (same cropping logic) without weights.
                        # We'll do it directly.
                        C, H, W = sum_maps[head].shape
                        tx0 = max(0, x0m)
                        ty0 = max(0, y0m)
                        tx1 = min(W, x1m)
                        ty1 = min(H, y1m)
                        if tx1 <= tx0 or ty1 <= ty0:
                            continue
                        ox0 = tx0 - x0m
                        oy0 = ty0 - y0m
                        ox1 = ox0 + (tx1 - tx0)
                        oy1 = oy0 + (ty1 - ty0)
                        sum_maps[head][:, ty0:ty1, tx0:tx1] += tile_logits[
                            :, oy0:oy1, ox0:ox1
                        ]

    if not sum_maps:
        raise RuntimeError(
            "No logits were stitched (model returned no usable outputs)."
        )

    if w_map is None:
        # If no weight map, treat as uniform 1 everywhere that got written at least once is unknown.
        # For correctness with overlaps you really want w_map; so if you disable it, you accept last-write semantics.
        # Here: just return sums (not averaged).
        avg_by_head = {k: v for k, v in sum_maps.items()}
    else:
        avg_by_head = {k: finalize_canvas(v, w_map) for k, v in sum_maps.items()}
    

    meta = StitchMeta(
        mask_h=mask_h,
        mask_w=mask_w,
        level0_w=level0_w,
        level0_h=level0_h,
        fx=float(fx),
        fy=float(fy),
        output_target_mpp=float(output_target_mpp),
        tile_level=int(ds.tile_level),
        tile_size_lvl=int(ds.tile_size_lvl),
        tile_extent0=float(tile_extent0),
    )

    return StitchResult(
        avg_logits_by_head=avg_by_head,
        weight_map=w_map,
        meta=meta,
    )
