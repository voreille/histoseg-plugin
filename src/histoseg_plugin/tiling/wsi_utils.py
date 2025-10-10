from typing import List, Tuple, Union

import openslide


def assert_level_downsamples(
    wsi: Union[openslide.OpenSlide, openslide.ImageSlide]
) -> List[Tuple[float, float]]:
    """I am adapting the code from CLAM and don't know why they use this
    instead of the provided openslide API, will check to see the difference
    probably to have the downsample as tuple (x,y) instead of single float
    """
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]

    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]),
                                dim_0[1] / float(dim[1]))
        level_downsamples.append(
            estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append(
                    (downsample, downsample))

    return level_downsamples
