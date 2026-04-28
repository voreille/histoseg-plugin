import json

from histoseg_plugin.storage.results import build_result_dir, write_geojson, write_stats


def test_write_result_files(tmp_path):
    task_hash = "abc123"

    result_dir = build_result_dir(tmp_path, task_hash)

    geojson = {"type": "FeatureCollection", "features": []}
    stats = {"slide_path": "/tmp/slide.svs", "model_id": "default"}

    geojson_path = write_geojson(result_dir, geojson)
    stats_path = write_stats(result_dir, stats)

    assert geojson_path.exists()
    assert stats_path.exists()

    assert json.loads(geojson_path.read_text()) == geojson
    assert json.loads(stats_path.read_text()) == stats