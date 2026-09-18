import inspect


def test_flow_module_imports():
    """orchestration/flow.py imported ingest_rsqa_csv, which never existed."""
    import orchestration.flow as flow

    assert hasattr(flow, "daily_pipeline")


def test_flow_exposes_the_new_ingest_tasks():
    import orchestration.flow as flow

    assert hasattr(flow, "ingest_daily")
    assert hasattr(flow, "build_gold_table")


def test_flow_no_longer_references_the_missing_function():
    import orchestration.flow as flow

    source = inspect.getsource(flow)
    assert "ingest_rsqa_csv" not in source


def test_flow_has_no_hardcoded_download_urls():
    """URLs live in src/data/sources.py, resolved via CKAN (decision D8)."""
    import orchestration.flow as flow

    source = inspect.getsource(flow)
    assert "donnees.montreal.ca" not in source


def test_daily_pipeline_runs_ingest_and_build_before_training():
    """The pipeline must produce data before it trains on it."""
    import orchestration.flow as flow

    source = inspect.getsource(flow.daily_pipeline)
    for name in ("ingest_daily", "build_gold_table", "train_daily_model"):
        assert name in source, f"{name} missing from daily_pipeline"
    assert source.index("ingest_daily") < source.index("train_daily_model")
    assert source.index("build_gold_table") < source.index("train_daily_model")
