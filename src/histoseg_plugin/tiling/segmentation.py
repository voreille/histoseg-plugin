from typing import Optional, Union
import openslide
import numpy as np
import cv2
from .wsi_utils import assert_level_downsamples
from .contour_utils import filter_contours, scaleContourDim, scaleHolesDim


def segment_tissue(wsi: Union[openslide.OpenSlide, openslide.ImageSlide],
                   seg_level: int = 0,
                   sthresh: int = 20,
                   sthresh_up: int = 255,
                   mthresh: int = 7,
                   close: int = 0,
                   use_otsu: bool = False,
                   filter_params: Optional[dict[str, Union[float,
                                                           int]]] = None,
                   ref_patch_size: int = 512,
                   exclude_ids: Optional[list[int]] = None,
                   keep_ids: Optional[list[int]] = None):
    """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
    """

    if exclude_ids is None:
        exclude_ids = []
    if keep_ids is None:
        keep_ids = []

    if filter_params is None:
        filter_params = {
            'a_t': 100,
            'a_h': 16,
            'max_n_holes': 8,
        }

    level_downsamples = assert_level_downsamples(wsi)

    img = np.array(
        wsi.read_region((0, 0), seg_level, wsi.level_dimensions[seg_level]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1],
                             mthresh)  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up,
                                    cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up,
                                    cv2.THRESH_BINARY)

    # Morphological closing
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    scale = level_downsamples[seg_level]
    scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
    filter_params_rescaled = filter_params.copy()
    filter_params_rescaled[
        'a_t'] = filter_params['a_t'] * scaled_ref_patch_area
    filter_params_rescaled[
        'a_h'] = filter_params['a_h'] * scaled_ref_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0, ))[:, 2:]

    foreground_contours, hole_contours = filter_contours(
        contours, hierarchy,
        filter_params_rescaled)  # Necessary for filtering out artifacts

    contours_tissue = scaleContourDim(foreground_contours, scale)
    holes_tissue = scaleHolesDim(hole_contours, scale)

    #exclude_ids = [0,7,9]
    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    contours_tissue = [contours_tissue[i] for i in contour_ids]
    holes_tissue = [holes_tissue[i] for i in contour_ids]
    return contours_tissue, holes_tissue

