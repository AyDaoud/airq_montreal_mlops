from src.data.watermark import read_watermarks, should_skip, write_watermark


def test_roundtrip(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa_2022_2024", "2024-12-31T05:02:04")
    assert read_watermarks(path)["iqa_2022_2024"] == "2024-12-31T05:02:04"


def test_write_preserves_other_keys(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "a", "1")
    write_watermark(path, "b", "2")
    marks = read_watermarks(path)
    assert marks == {"a": "1", "b": "2"}


def test_should_skip_only_when_token_matches(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa", "2024-12-31T05:02:04")
    assert should_skip(path, "iqa", "2024-12-31T05:02:04") is True
    assert should_skip(path, "iqa", "2026-01-19T05:02:19") is False
    assert should_skip(path, "other", "anything") is False


def test_should_never_skip_on_missing_token(tmp_path):
    """A resource with no last_modified must always be re-fetched."""
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa", "2024-12-31T05:02:04")
    assert should_skip(path, "iqa", None) is False


def test_read_missing_file_returns_empty(tmp_path):
    assert read_watermarks(tmp_path / "nope.json") == {}
