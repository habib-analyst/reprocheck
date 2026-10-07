"""Command-line interface for reprocheck.

Usage:
    reprocheck 2301.12345 --full-text paper.txt --format md --out report.md
    reprocheck --demo
"""

from __future__ import annotations

import argparse
import importlib.resources
import sys

from . import __version__, build_checklist, extract_signals
from .report import render_json, render_markdown


class ReprocheckError(Exception):
    """A user-facing CLI failure (bad input, missing file, API error)."""


def _load_demo_fixture() -> tuple[str, str, str]:
    """Return (title, abstract, full_text) from the bundled sample paper."""
    ref = importlib.resources.files("reprocheck") / "fixtures" / "sample_paper.txt"
    text = ref.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    title = paragraphs[0].removeprefix("TITLE:").strip() if paragraphs else ""
    abstract = ""
    body_paragraphs: list[str] = []
    for para in paragraphs[1:]:
        if not abstract and para.startswith("ABSTRACT:"):
            abstract = para.removeprefix("ABSTRACT:").strip()
        else:
            body_paragraphs.append(para)
    return title, abstract, "\n\n".join(body_paragraphs)


def _read_full_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise ReprocheckError(f"cannot read --full-text file {path!r}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reprocheck",
        description="Audit an arXiv paper's reproducibility signals and emit a checklist.",
    )
    parser.add_argument("arxiv_id", nargs="?",
                        help="arXiv ID, e.g. 2301.12345 (requires network)")
    parser.add_argument("--full-text", metavar="FILE",
                        help="path to the paper's full text (plain text)")
    parser.add_argument("--format", choices=["md", "json"], default="md",
                        help="report format (default: md)")
    parser.add_argument("--out", metavar="FILE",
                        help="write report to FILE instead of stdout")
    parser.add_argument("--demo", action="store_true",
                        help="run on the bundled sample paper; no network needed")
    parser.add_argument("--llm", action="store_true",
                        help="also run the optional LLM audit "
                             "(needs REPROCHECK_API_BASE + REPROCHECK_API_KEY)")
    parser.add_argument("--version", action="version", version=f"reprocheck {__version__}")
    return parser


def _run(args: argparse.Namespace) -> str:
    """Run the audit and return the rendered report string."""
    if args.demo:
        paper_id = "demo:sample-paper"
        title, abstract, full_text = _load_demo_fixture()
    elif args.arxiv_id:
        # Network access happens ONLY here, inside arxiv_client.
        from . import arxiv_client

        try:
            meta = arxiv_client.fetch_paper(args.arxiv_id)
        except arxiv_client.ArxivClientError as exc:
            raise ReprocheckError(str(exc)) from exc
        paper_id, title, abstract = meta["id"], meta["title"], meta["abstract"]
        full_text = _read_full_text(args.full_text) if args.full_text else ""
    elif args.full_text:
        paper_id, title, abstract = "local-file", "", ""
        full_text = _read_full_text(args.full_text)
    else:
        raise ReprocheckError("provide an arXiv ID, --demo, or --full-text FILE")

    result = extract_signals(title, abstract, full_text)

    llm_audit = None
    if args.llm:
        from . import llm_backend

        try:
            llm_audit = llm_backend.audit(result.scanned_text())
        except llm_backend.LLMNotAvailable as exc:
            raise ReprocheckError(str(exc)) from exc

    report = build_checklist(result, paper_id=paper_id, llm_audit=llm_audit)
    return render_json(report) if args.format == "json" else render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = _run(args)
    except ReprocheckError as exc:
        print(f"reprocheck: {exc}", file=sys.stderr)
        return 2

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(output)
        except OSError as exc:
            print(f"reprocheck: cannot write {args.out!r}: {exc}", file=sys.stderr)
            return 2
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
