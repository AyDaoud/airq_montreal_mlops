from src.data.http import BROWSER_UA, build_session, download


def test_session_sends_browser_user_agent():
    session = build_session()
    assert "Mozilla/5.0" in session.headers["User-Agent"]
    assert session.headers["User-Agent"] == BROWSER_UA


def test_download_writes_file_and_creates_parents(tmp_path, monkeypatch):
    dest = tmp_path / "nested" / "out.csv"

    class FakeResponse:
        status_code = 200
        headers = {"ETag": '"abc123"'}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"col\n"
            yield b"1\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeSession:
        headers = {"User-Agent": BROWSER_UA}

        def get(self, url, stream=False, timeout=None):
            return FakeResponse()

    result = download("https://example.invalid/x.csv", dest, session=FakeSession())

    assert result == dest
    assert dest.read_text() == "col\n1\n"


def test_download_skips_when_etag_unchanged(tmp_path):
    dest = tmp_path / "out.csv"
    dest.write_text("cached\n")
    etag_file = dest.with_suffix(dest.suffix + ".etag")
    etag_file.write_text('"abc123"')

    class NotModifiedResponse:
        status_code = 304
        headers = {}

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeSession:
        headers = {"User-Agent": BROWSER_UA}
        seen_headers = {}

        def get(self, url, stream=False, timeout=None, headers=None):
            FakeSession.seen_headers = headers or {}
            return NotModifiedResponse()

    result = download("https://example.invalid/x.csv", dest, session=FakeSession())

    assert result == dest
    assert dest.read_text() == "cached\n"
    assert FakeSession.seen_headers.get("If-None-Match") == '"abc123"'
