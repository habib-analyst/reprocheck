"""reprocheck: arXiv-paper → reproducibility-checklist tool.

Scans a paper's metadata, abstract and (optional) full text for
reproducibility signals — datasets, code, hyperparameters, compute,
evaluation rigor, claims — and produces a structured checklist with a
transparent 0–100 reproducibility score.
"""

from .checklist import (
    ChecklistItem,
    ReproducibilityReport,
    Verdict,
    build_checklist,
)
from .heuristic_extractor import ExtractionResult, Finding, extract_signals

__version__ = "0.1.0"

__all__ = [
    "ChecklistItem",
    "ExtractionResult",
    "Finding",
    "ReproducibilityReport",
    "Verdict",
    "build_checklist",
    "extract_signals",
    "__version__",
]
