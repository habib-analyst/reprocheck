"""Tests for the arXiv client — all offline via fixtures/mocks."""

import pytest

from reprocheck import arxiv_client
from reprocheck.arxiv_client import (
    ArxivClientError,
    fetch_paper,
    normalize_id,
    parse_response,
    validate_id,
)

ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <updated>2023-02-01T00:00:00Z</updated>
    <published>2023-01-20T00:00:00Z</published>
    <title>MedFormer: A Test Paper</title>
    <summary>We test reproducibility tooling on medical images.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <arxiv:comment>10 pages</arxiv:comment>
    <arxiv:journal_ref>Test Journal 2023</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2301.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.12345v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
    <category term="eess.IV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_normalize_id_strips_version_and_whitespace():
    assert normalize_id("  2301.12345v2 ") == "2301.12345"
    assert normalize_id("hep-th/9901001") == "hep-th/9901001"


def test_validate_id_accepts_modern_and_legacy():
    assert validate_id("2301.12345") == "2301.12345"
    assert validate_id("hep-th/9901001v3") == "hep-th/9901001"


def test_validate_id_rejects_garbage():
    for bad in ["not-an-id", "2301", "", "2301.123456789", "http://arxiv.org/abs/2301.12345"]:
        with pytest.raises(ArxivClientError):
            validate_id(bad)


def test_parse_response_extracts_metadata():
    meta = parse_response(ATOM_FIXTURE, "2301.12345")
    assert meta["id"] == "2301.12345"
    assert meta["title"] == "MedFormer: A Test Paper"
    assert meta["abstract"] == "We test reproducibility tooling on medical images."
    assert meta["authors"] == ["Jane Doe", "John Smith"]
    assert meta["categories"] == ["cs.CV", "eess.IV"]
    assert meta["primary_category"] == "cs.CV"
    assert meta["pdf_url"] == "http://arxiv.org/pdf/2301.12345v2"
    assert meta["comment"] == "10 pages"


def test_parse_response_empty_feed_raises():
    with pytest.raises(ArxivClientError):
        parse_response('<feed xmlns="http://www.w3.org/2005/Atom"/>', "2301.12345")


def test_fetch_paper_uses_mocked_http(monkeypatch):
    import requests

    def fake_get(url, params=None, timeout=None, headers=None):
        assert "export.arxiv.org" in url
        assert params["id_list"] == "2301.12345"
        return _FakeResponse(ATOM_FIXTURE)

    monkeypatch.setattr(requests, "get", fake_get)
    meta = fetch_paper("2301.12345v2")
    assert meta["title"] == "MedFormer: A Test Paper"
    assert meta["id"] == "2301.12345"


def test_fetch_paper_http_error_raises(monkeypatch):
    import requests

    def fake_get(*args, **kwargs):
        return _FakeResponse("oops", status_code=503)

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(ArxivClientError):
        fetch_paper("2301.12345")
