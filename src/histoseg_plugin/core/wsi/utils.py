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
