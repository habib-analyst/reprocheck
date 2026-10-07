"""Optional pluggable LLM backend (OpenAI-compatible).

The tool works FULLY OFFLINE with the heuristic extractor. This backend is
only used when explicitly requested (``reprocheck --llm ...``) AND the
environment provides ``REPROCHECK_API_BASE`` and ``REPROCHECK_API_KEY``.
Never required by tests; never imported on the default code path.
"""

from __future__ import annotations

import json
import os

import requests

DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 60

AUDIT_SYSTEM_PROMPT = (
    "You audit ML papers for reproducibility. Given the paper text, reply with "
    "ONLY a JSON object with this schema: "
    '{"summary": str, "missing": [str], "risk_flags": [str], '
    '"verdicts": {"Data": "PASS"|"PARTIAL"|"MISSING", "Code": ..., '
    '"Hyperparameters": ..., "Compute": ..., "Evaluation rigor": ..., "Claims": ...}}. '
    "Be strict: a category is PASS only with concrete, checkable detail."
)


class LLMNotAvailable(Exception):
    """Raised when the LLM backend is requested but not configured."""


def available() -> bool:
    """True when REPROCHECK_API_BASE and REPROCHECK_API_KEY are both set."""
    return bool(os.environ.get("REPROCHECK_API_BASE") and os.environ.get("REPROCHECK_API_KEY"))


def config() -> dict:
    """Return backend config from the environment (no secrets in return value)."""
    if not available():
        raise LLMNotAvailable(
            "LLM backend not configured. Set REPROCHECK_API_BASE and "
            "REPROCHECK_API_KEY to enable --llm."
        )
    return {
        "api_base": os.environ["REPROCHECK_API_BASE"].rstrip("/"),
        "model": os.environ.get("REPROCHECK_MODEL", DEFAULT_MODEL),
    }


def audit(text: str, max_chars: int = 12000) -> dict:
    """Ask the LLM to audit the paper text; returns the parsed JSON verdict."""
    cfg = config()
    api_key = os.environ["REPROCHECK_API_KEY"]  # read here, never logged or returned
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": text[:max_chars]},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(
            f"{cfg['api_base']}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMNotAvailable(f"LLM request failed: {exc}") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise LLMNotAvailable(f"Could not parse LLM response: {exc}") from exc
