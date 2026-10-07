"""ArXiv API client.

This is the ONLY module in reprocheck that touches the network.
Everything else in the package works fully offline.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests

API_URL = "http://export.arxiv.org/api/query"
TIMEOUT_SECONDS = 30

# arXiv IDs look like "2301.12345" (optionally with a version suffix "v2")
# or legacy IDs like "hep-th/9901001".
_ARXIV_ID_RE = re.compile(
    r"^(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE
)


class ArxivClientError(Exception):
    """Raised when the arXiv API request fails or the ID is invalid."""


def normalize_id(arxiv_id: str) -> str:
    """Strip whitespace and optional version suffix from an arXiv ID."""
    cleaned = arxiv_id.strip()
    # Keep the ID valid for lookup; arXiv accepts versioned IDs, but we
    # canonicalize to the unversioned form for stable reporting.
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned


def validate_id(arxiv_id: str) -> str:
    """Return the normalized ID, raising ArxivClientError if malformed."""
    normalized = normalize_id(arxiv_id)
    if not _ARXIV_ID_RE.match(normalized):
        raise ArxivClientError(
            f"Invalid arXiv ID: {arxiv_id!r}. Expected e.g. '2301.12345'."
        )
    return normalized


def fetch_paper(arxiv_id: str) -> dict:
    """Fetch paper metadata from the arXiv API.

    Returns a dict with keys: id, title, abstract, authors, published,
    updated, categories, primary_category, pdf_url, comment, doi, journal_ref.
    """
    paper_id = validate_id(arxiv_id)
    try:
        response = requests.get(
            API_URL,
            params={"id_list": paper_id, "max_results": 1},
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": "reprocheck/0.1.0 (mailto:habib.gcuf.edu@gmail.com)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ArxivClientError(f"arXiv API request failed: {exc}") from exc

    return parse_response(response.text, paper_id)


def parse_response(xml_text: str, requested_id: str) -> dict:
    """Parse an arXiv API Atom response into a metadata dict (no network)."""
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivClientError(f"Could not parse arXiv API response: {exc}") from exc

    entries = root.findall("atom:entry", ns)
    if not entries:
        raise ArxivClientError(f"No paper found for arXiv ID {requested_id!r}.")

    entry = entries[0]

    def _text(tag: str, namespace: str = "atom") -> str:
        el = entry.find(f"{namespace}:{tag}", ns)
        return " ".join(el.text.split()) if el is not None and el.text else ""

    authors = [
        " ".join(a.find("atom:name", ns).text.split())
        for a in entry.findall("atom:author", ns)
        if a.find("atom:name", ns) is not None and a.find("atom:name", ns).text
    ]
    categories = [
        c.get("term", "")
        for c in entry.findall("atom:category", ns)
        if c.get("term")
    ]
    pdf_url = ""
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break

    return {
        "id": requested_id,
        "title": _text("title"),
        "abstract": _text("summary"),
        "authors": authors,
        "published": _text("published"),
        "updated": _text("updated"),
        "categories": categories,
        "primary_category": categories[0] if categories else "",
        "pdf_url": pdf_url,
        "comment": _text("comment", "arxiv"),
        "doi": _text("doi", "arxiv"),
        "journal_ref": _text("journal_ref", "arxiv"),
    }
