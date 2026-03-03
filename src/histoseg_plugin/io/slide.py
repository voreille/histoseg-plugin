from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote


def slide_uri_to_path(slide_uri: str) -> Path:
    # Accept raw paths too
    if "://" not in slide_uri and slide_uri.startswith("/"):
        return Path(slide_uri)

    u = urlparse(slide_uri)

    if u.scheme == "file":
        # Handles both file:/mnt/x and file:///mnt/x
        return Path(unquote(u.path))

    # If you later want ssh/s3/omero schemes, add here.
    raise ValueError(f"Unsupported slide_uri scheme: {u.scheme!r}")


def assert_allowed_root(path: Path, allowed_roots: list[Path]) -> None:
    rp = path.resolve()
    for root in allowed_roots:
        rr = root.resolve()
        try:
            rp.relative_to(rr)
            return
        except ValueError:
            continue
    raise PermissionError(f"Slide path not under allowed roots: {rp}")
