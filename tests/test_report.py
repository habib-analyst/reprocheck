"""Tests for report rendering (Markdown + JSON)."""

import json

from reprocheck import build_checklist
from reprocheck.report import render_json, render_markdown


def test_markdown_report_structure(extraction):
    report = build_checklist(extraction, paper_id="demo:sample-paper")
    md = render_markdown(report)
    assert md.startswith("# Reproducibility Checklist")
    assert "## Score: 80 / 100" in md
    assert "| Data | ✅ PASS" in md
    assert "| Code | ❌ MISSING" in md
    assert "| Hyperparameters | ✅ PASS" in md
    assert "## Evidence" in md
    assert "demo:sample-paper" in md
    assert "reprocheck" in md  # footer attribution


def test_markdown_escapes_table_pipes(extraction):
    report = build_checklist(extraction, paper_id="demo")
    report.items[0].note = "a|b"
    md = render_markdown(report)
    assert "a\\|b" in md


def test_json_report_round_trips(extraction):
    report = build_checklist(extraction, paper_id="demo:sample-paper")
    parsed = json.loads(render_json(report))
    assert parsed["paper_id"] == "demo:sample-paper"
    assert parsed["score"] == 80
    assert len(parsed["items"]) == 6
    assert parsed["finding_count"] > 0


def test_empty_report_renders(extraction):
    from reprocheck import extract_signals

    report = build_checklist(extract_signals("", ""))
    md = render_markdown(report)
    assert "## Score: 0 / 100" in md
    assert "❌ MISSING" in md
    assert "_No evidence quotes extracted._" in md
