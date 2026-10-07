"""Tests for checklist building and scoring."""

from reprocheck import build_checklist, extract_signals
from reprocheck.checklist import CATEGORIES, Verdict


def test_fixture_checklist_verdicts(extraction):
    report = build_checklist(extraction, paper_id="demo")
    by_cat = {i.category: i for i in report.items}
    assert by_cat["Data"].verdict == Verdict.PASS
    assert by_cat["Code"].verdict == Verdict.MISSING  # fixture has no code URL
    assert by_cat["Hyperparameters"].verdict == Verdict.PASS
    assert by_cat["Compute"].verdict == Verdict.PASS
    assert by_cat["Evaluation rigor"].verdict == Verdict.PASS
    assert by_cat["Claims"].verdict == Verdict.PASS


def test_fixture_score_is_transparent(extraction):
    report = build_checklist(extraction, paper_id="demo")
    # 20 + 0 + 20 + 10 + 20 + 10 = 80 (only Code is missing)
    assert report.score == 80
    assert report.max_score == 100
    assert sum(i.points for i in report.items) == report.score


def test_weights_sum_to_100():
    assert sum(w for w, _ in CATEGORIES.values()) == 100


def test_empty_paper_scores_zero():
    result = extract_signals("", "")
    report = build_checklist(result)
    assert report.score == 0
    assert all(i.verdict == Verdict.MISSING for i in report.items)
    assert all(i.points == 0 for i in report.items)


def test_partial_verdicts_earn_half_weight():
    # Only a learning rate mentioned: Hyperparameters -> PARTIAL (10/20).
    result = extract_signals("T", "We used a learning rate of 1e-3.")
    report = build_checklist(result, paper_id="x")
    by_cat = {i.category: i for i in report.items}
    assert by_cat["Hyperparameters"].verdict == Verdict.PARTIAL
    assert by_cat["Hyperparameters"].points == 10
    assert report.score == 10


def test_to_dict_schema(extraction):
    report = build_checklist(extraction, paper_id="demo")
    d = report.to_dict()
    assert set(d) == {
        "paper_id", "title", "score", "max_score", "items",
        "finding_count", "llm_audit",
    }
    assert d["finding_count"] == len(extraction.findings)
    for item in d["items"]:
        assert item["verdict"] in {"PASS", "PARTIAL", "MISSING"}
        assert item["points"] <= item["weight"]


def test_llm_audit_attached_when_provided(extraction):
    fake_audit = {"summary": "ok", "missing": [], "risk_flags": [], "verdicts": {}}
    report = build_checklist(extraction, llm_audit=fake_audit)
    assert report.to_dict()["llm_audit"] == fake_audit
