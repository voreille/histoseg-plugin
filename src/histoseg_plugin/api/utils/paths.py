# api/dependencies/paths.py

from pathlib import Path
from urllib.parse import unquote, urlparse


def slide_uri_to_path(slide_uri: str) -> Path:
    if "://" not in slide_uri and slide_uri.startswith("/"):
        return Path(slide_uri)

    parsed = urlparse(slide_uri)

    if parsed.scheme == "file":
        return Path(unquote(parsed.path))

    raise ValueError(f"Unsupported slide_uri scheme: {parsed.scheme!r}")


def resolve_allowed_path(slide_uri: str, allowed_roots: list[Path]) -> Path:
    path = slide_uri_to_path(slide_uri).resolve()

    for root in allowed_roots:
        try:
            path.relative_to(root.resolve())
            break
        except ValueError:
            continue
    else:
        raise PermissionError(f"Slide path not under allowed roots: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Slide file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Slide path is not a file: {path}")

    return path
