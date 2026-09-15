import pytest

from src.data import sources
from src.data.ckan import CkanError, resolve


def _fake_session(payload, status=200):
    class FakeResponse:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    class FakeSession:
        def get(self, url, timeout=None, params=None):
            FakeSession.last_params = params
            return FakeResponse()

    return FakeSession()


def test_resolve_returns_url_and_last_modified():
    payload = {
        "result": {
            "resources": [
                {"id": "other-id", "url": "https://x/other.csv", "last_modified": None},
                {
                    "id": "29db5545-89a4-4e4a-9e95-05aa6dc2fd80",
                    "url": "https://x/liste-des-stations_do_rev_2026-05-22.csv",
                    "last_modified": "2026-05-22T05:00:00",
                },
            ]
        }
    }
    resolved = resolve(sources.STATIONS, session=_fake_session(payload))
    assert resolved.url.endswith("liste-des-stations_do_rev_2026-05-22.csv")
    assert resolved.last_modified == "2026-05-22T05:00:00"


def test_resolve_falls_back_when_api_unavailable():
    class BrokenSession:
        def get(self, url, timeout=None, params=None):
            raise OSError("network down")

    resolved = resolve(sources.IQA_REALTIME, session=BrokenSession())
    assert resolved.url == sources.IQA_REALTIME.fallback_url
    assert resolved.last_modified is None


def test_resolve_raises_when_no_fallback_and_api_unavailable():
    class BrokenSession:
        def get(self, url, timeout=None, params=None):
            raise OSError("network down")

    with pytest.raises(CkanError, match="no fallback"):
        resolve(sources.STATIONS, session=BrokenSession())


def test_resolve_raises_when_resource_missing_from_payload():
    payload = {"result": {"resources": [{"id": "nope", "url": "https://x/a.csv"}]}}
    with pytest.raises(CkanError, match="not found"):
        resolve(sources.STATIONS, session=_fake_session(payload))
