"""Shared fixtures: guarantee tests never touch the network or need API keys."""

import os

import pytest


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    monkeypatch.delenv("REPROCHECK_API_BASE", raising=False)
    monkeypatch.delenv("REPROCHECK_API_KEY", raising=False)
    monkeypatch.delenv("REPROCHECK_MODEL", raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test tries real HTTP."""
    import requests

    def _blocked(*args, **kwargs):
        raise AssertionError(f"network access attempted in test: {args[0]!r}")

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)


@pytest.fixture()
def sample_paper_text():
    import importlib.resources

    ref = importlib.resources.files("reprocheck") / "fixtures" / "sample_paper.txt"
    return ref.read_text(encoding="utf-8")


@pytest.fixture()
def extraction(sample_paper_text):
    from reprocheck.cli import _load_demo_fixture

    title, abstract, full_text = _load_demo_fixture()
    from reprocheck import extract_signals

    return extract_signals(title, abstract, full_text)
