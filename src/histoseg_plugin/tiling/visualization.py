import math

import cv2
import numpy as np
from PIL import Image

from .contour_utils import scaleContourDim
from .wsi_utils import assert_level_downsamples


def visWSI(wsi,
           contours_tissue,
           holes_tissue,
           contours_tumor=None,
           vis_level=0,
           color=(0, 255, 0),
           hole_color=(0, 0, 255),
           annot_color=(255, 0, 0),
           line_thickness=250,
           max_size=None,
           top_left=None,
           bot_right=None,
           custom_downsample=1,
           view_slide_only=False,
           number_contours=False,
           seg_display=True,
           annot_display=True):

    level_downsamples = assert_level_downsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]

    if top_left is not None and bot_right is not None:
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) -
                     (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
    else:
        top_left = (0, 0)
        region_size = wsi.level_dimensions[vis_level]

    img = np.array(
        wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

    if not view_slide_only:
        offset = tuple(-(np.array(top_left) * scale).astype(int))
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if contours_tissue is not None and seg_display:
            if not number_contours:
                cv2.drawContours(img,
                                 scaleContourDim(contours_tissue, scale),
                                 -1,
                                 color,
                                 line_thickness,
                                 lineType=cv2.LINE_8,
                                 offset=offset)

            else:  # add numbering to each contour
                for idx, cont in enumerate(contours_tissue):
                    contour = np.array(scaleContourDim(cont, scale))
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-9))
                    cY = int(M["m01"] / (M["m00"] + 1e-9))
                    # draw the contour and put text next to center
                    cv2.drawContours(img, [contour],
                                     -1,
                                     color,
                                     line_thickness,
                                     lineType=cv2.LINE_8,
                                     offset=offset)
                    cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

            for holes in holes_tissue:
                cv2.drawContours(img,
                                 scaleContourDim(holes, scale),
                                 -1,
                                 hole_color,
                                 line_thickness,
                                 lineType=cv2.LINE_8)

        if contours_tumor is not None and annot_display:
            cv2.drawContours(img,
                             scaleContourDim(contours_tumor, scale),
                             -1,
                             annot_color,
                             line_thickness,
                             lineType=cv2.LINE_8,
                             offset=offset)

    img = Image.fromarray(img)

    w, h = img.size
    if custom_downsample > 1:
        img = img.resize(
            (int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img


def StitchCoords(wsi,
                 hdf5_file_path,
                 downscale=16,
                 draw_grid=False,
                 bg_color=(0, 0, 0),
                 alpha=-1):
    w, h = wsi.level_dimensions[0]
    print('original size: {} x {}'.format(w, h))

    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('downscaled size for stiching: {} x {}'.format(w, h))

    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        print('start stitching {}'.format(dset.attrs['name']))
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']

    print(f'number of patches: {len(coords)}')
    print(
        f'patch size: {patch_size} x {patch_size} patch level: {patch_level}')
    patch_size = tuple((np.array((patch_size, patch_size)) *
                        wsi.level_downsamples[patch_level]).astype(np.int32))
    print(f'ref patch size: {patch_size} x {patch_size}')

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            "Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h),
                            mode="RGBA",
                            color=bg_color + (int(255 * alpha), ))

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap,
                                wsi,
                                coords,
                                patch_size,
                                vis_level,
                                indices=None,
                                draw_grid=draw_grid)
    return heatmap


def DrawMapFromCoords(canvas,
                      wsi,
                      coords,
                      patch_size,
                      vis_level,
                      indices=None,
                      draw_grid=True):
    downsamples = wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    patch_size = tuple(
        np.ceil(
            (np.array(patch_size) / np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    for idx in tqdm(range(total)):
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(
            wsi.read_region(tuple(coord), vis_level,
                            patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1],
                                   coord[0]:coord[0] +
                                   patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1],
               coord[0]:coord[0] + patch_size[0], :
               3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(img,
                  tuple(np.maximum([0, 0], coord - thickness // 2)),
                  tuple(coord - thickness // 2 + np.array(shape)),
                  (0, 0, 0, 255),
                  thickness=thickness)
    return img
