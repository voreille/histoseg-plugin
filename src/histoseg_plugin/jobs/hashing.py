import hashlib
import json


def canonical_json(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_dict(data) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def build_task_payload(slide_path: str, model_id: str, params: dict) -> dict:
    return {
        "slide_path": slide_path,
        "model_id": model_id,
        "params": params,
    }
