"""Tests for the optional LLM backend — never needs real credentials."""

import json

import pytest

from reprocheck import llm_backend
from reprocheck.llm_backend import LLMNotAvailable, audit, available, config


def test_unavailable_without_env():
    assert available() is False


def test_config_raises_without_env():
    with pytest.raises(LLMNotAvailable, match="not configured"):
        config()


def test_audit_raises_without_env():
    with pytest.raises(LLMNotAvailable):
        audit("some paper text")


def test_available_with_env(monkeypatch):
    monkeypatch.setenv("REPROCHECK_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("REPROCHECK_API_KEY", "sk-test")
    assert available() is True
    cfg = config()
    assert cfg["api_base"] == "https://api.example.com/v1"
    assert "api_key" not in json.dumps(cfg).lower()  # key never leaks via config()


def test_audit_success_with_mock(monkeypatch):
    import requests

    monkeypatch.setenv("REPROCHECK_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("REPROCHECK_API_KEY", "sk-test")

    body = {"summary": "solid", "missing": ["code"], "risk_flags": [], "verdicts": {}}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(body)}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "https://api.example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-test"
        assert json["response_format"] == {"type": "json_object"}
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    assert audit("paper text") == body


def test_audit_http_failure_raises(monkeypatch):
    import requests

    monkeypatch.setenv("REPROCHECK_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("REPROCHECK_API_KEY", "sk-test")

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(LLMNotAvailable, match="LLM request failed"):
        audit("paper text")
