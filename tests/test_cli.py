"""Tests for the CLI — demo and file paths run fully offline."""

import json

import pytest

from reprocheck.cli import _load_demo_fixture, main


def test_demo_fixture_parses():
    title, abstract, full_text = _load_demo_fixture()
    assert title.startswith("MedFormer-V2")
    assert "hierarchical vision transformer" in abstract
    assert "METHODS" in full_text


def test_demo_end_to_end_md(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "# Reproducibility Checklist" in out
    assert "## Score: 80 / 100" in out
    assert "❌ MISSING" in out  # code URL missing in fixture


def test_demo_end_to_end_json(tmp_path, capsys):
    out_file = tmp_path / "report.json"
    assert main(["--demo", "--format", "json", "--out", str(out_file)]) == 0
    assert capsys.readouterr().out == ""  # nothing on stdout when --out used
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert parsed["score"] == 80
    assert parsed["paper_id"] == "demo:sample-paper"


def test_no_args_exits_2(capsys):
    assert main([]) == 2
    assert "provide an arXiv ID" in capsys.readouterr().err


def test_invalid_arxiv_id_fails_before_network(capsys):
    # validate_id raises before any HTTP request is made.
    assert main(["not-an-id"]) == 2
    assert "Invalid arXiv ID" in capsys.readouterr().err


def test_full_text_file_without_id(tmp_path, capsys):
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "We train on ImageNet with Adam, lr 1e-3, batch 256, for 90 epochs "
        "on 8x V100 GPUs. Code: https://github.com/org/repo",
        encoding="utf-8",
    )
    assert main(["--full-text", str(paper)]) == 0
    out = capsys.readouterr().out
    assert "## Score:" in out
    assert "✅ PASS" in out


def test_missing_full_text_file_exits_2():
    assert main(["--full-text", "/does/not/exist.txt"]) == 2


def test_llm_flag_without_config_exits_2(capsys):
    assert main(["--demo", "--llm"]) == 2
    assert "not configured" in capsys.readouterr().err
