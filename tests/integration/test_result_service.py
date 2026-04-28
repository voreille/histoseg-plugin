from histoseg_plugin.jobs.result_service import register_result, find_result_by_hash


def test_register_result(session_factory, tmp_path):
    result_dir = tmp_path / "results" / "abc123"
    result_dir.mkdir(parents=True)

    geojson_path = result_dir / "result.geojson"
    stats_path = result_dir / "stats.json"

    geojson_path.write_text("{}", encoding="utf-8")
    stats_path.write_text("{}", encoding="utf-8")

    with session_factory.begin() as session:
        result = register_result(
            session,
            task_hash="abc123",
            slide_path="/tmp/slide.svs",
            model_id="default",
            result_dir=str(result_dir),
            geojson_path=str(geojson_path),
            stats_path=str(stats_path),
        )

    with session_factory() as session:
        found = find_result_by_hash(session, "abc123")

    assert found is not None
    assert found.id == result.id
    assert found.task_hash == "abc123"
