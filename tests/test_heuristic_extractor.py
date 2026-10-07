"""Tests for the offline heuristic extractor."""

from reprocheck import extract_signals
from reprocheck.heuristic_extractor import ExtractionResult


def test_extractor_finds_expected_signals(extraction):
    signals = extraction.signals()
    for expected in [
        "dataset_named",      # Synapse
        "metric_reported",    # Dice
        "metric_value",       # Dice score of 84.6%
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",          # AdamW
        "seed",               # random seeds
        "gpu_type",           # A100
        "compute_time",       # 36 GPU-hours
        "multiple_runs",      # 3 independent runs
        "std_reported",       # ±0.4 / mean ± std
        "error_bars",
        "baseline",
        "comparison",         # outperforming
        "quant_claim",        # achieves ... 84.6%
    ]:
        assert expected in signals, f"missing signal: {expected}"


def test_extractor_reports_no_code_url_for_fixture(extraction):
    code = extraction.by_category("Code")
    assert code == [], "fixture intentionally has no code URL"


def test_findings_carry_evidence_spans(extraction):
    import re

    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    assert len(extraction.findings) > 10
    for finding in extraction.findings:
        assert finding.matched, "matched span must not be empty"
        assert finding.evidence, "evidence quote must not be empty"
        assert norm(finding.matched)[:30] in norm(finding.evidence)
        assert 0 <= finding.start < finding.end <= len(extraction.scanned_text())


def test_extractor_on_empty_text():
    result = extract_signals("", "")
    assert isinstance(result, ExtractionResult)
    assert result.findings == []
    assert result.signals() == set()


def test_extractor_is_deterministic(extraction):
    again = extract_signals(extraction.title, extraction.abstract, extraction.full_text)
    assert [f.signal for f in again.findings] == [f.signal for f in extraction.findings]
    assert [(f.start, f.end) for f in again.findings] == [
        (f.start, f.end) for f in extraction.findings
    ]


def test_extractor_detects_github_code_url():
    result = extract_signals(
        "T", "Code available at https://github.com/org/repo for reproduction."
    )
    code = result.by_category("Code")
    assert any("github.com/org/repo" in f.matched for f in code)


def test_extractor_handles_none_like_inputs():
    result = extract_signals(None, None, None)
    assert result.findings == []
