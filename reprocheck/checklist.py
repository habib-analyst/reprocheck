"""Builds the structured reproducibility checklist from extracted signals.

Verdicts per category: PASS / PARTIAL / MISSING.
Score: 0–100, computed from transparent per-category weights —
PASS earns the full weight, PARTIAL earns half, MISSING earns zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .heuristic_extractor import ExtractionResult


class Verdict(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


# category -> (weight, description)
CATEGORIES: dict[str, tuple[int, str]] = {
    "Data": (20, "Datasets named and described (name, size, splits, access)."),
    "Code": (20, "Source code / implementation publicly available."),
    "Hyperparameters": (20, "Training details: lr, batch size, epochs, optimizer, seeds."),
    "Compute": (10, "Compute resources disclosed (hardware, training time)."),
    "Evaluation rigor": (20, "Metrics, baselines, multiple runs, statistical reporting."),
    "Claims": (10, "Quantitative claims backed by reported numbers."),
}


@dataclass
class ChecklistItem:
    category: str
    verdict: Verdict
    weight: int
    points: float
    evidence: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "verdict": self.verdict.value,
            "weight": self.weight,
            "points": self.points,
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass
class ReproducibilityReport:
    paper_id: str
    title: str
    score: int  # 0–100
    max_score: int = 100
    items: list[ChecklistItem] = field(default_factory=list)
    finding_count: int = 0
    llm_audit: dict | None = None

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "score": self.score,
            "max_score": self.max_score,
            "items": [i.to_dict() for i in self.items],
            "finding_count": self.finding_count,
            "llm_audit": self.llm_audit,
        }


def _evidence_for(result: ExtractionResult, signals: set[str], limit: int = 3) -> list[str]:
    out: list[str] = []
    for f in result.findings:
        if f.signal in signals and f.evidence not in out:
            out.append(f.evidence)
            if len(out) >= limit:
                break
    return out


def _check_data(result: ExtractionResult) -> ChecklistItem:
    named = result.by_category("Data")
    dataset_named = [f for f in named if f.signal == "dataset_named"]
    has_size = any(f.signal == "dataset_size" for f in named)
    weight, _ = CATEGORIES["Data"]
    if dataset_named:
        names = sorted({f.matched for f in dataset_named})
        note = f"Named dataset(s): {', '.join(names[:5])}."
        if has_size:
            note += " Dataset size reported."
        else:
            note += " Dataset size not quantified."
        return ChecklistItem("Data", Verdict.PASS, weight, float(weight),
                             _evidence_for(result, {"dataset_named", "dataset_size"}), note)
    if any(f.signal == "dataset_generic" for f in named):
        return ChecklistItem("Data", Verdict.PARTIAL, weight, weight / 2,
                             _evidence_for(result, {"dataset_generic"}),
                             "Data mentioned generically, but no named public dataset.")
    return ChecklistItem("Data", Verdict.MISSING, weight, 0.0, [],
                         "No dataset identified in title/abstract/text.")


def _check_code(result: ExtractionResult) -> ChecklistItem:
    code = result.by_category("Code")
    weight, _ = CATEGORIES["Code"]
    if code:
        return ChecklistItem("Code", Verdict.PASS, weight, float(weight),
                             _evidence_for(result, {"code_url", "code_available"}),
                             "Code availability statement or repository URL found.")
    return ChecklistItem("Code", Verdict.MISSING, weight, 0.0, [],
                         "No code URL or code-availability statement found.")


_HYPERPARAM_SIGNALS = {"learning_rate", "batch_size", "epochs", "optimizer", "weight_decay", "dropout", "scheduler"}


def _check_hyperparameters(result: ExtractionResult) -> ChecklistItem:
    found = {f.signal for f in result.by_category("Hyperparameters")} & _HYPERPARAM_SIGNALS
    has_seed = any(f.signal == "seed" for f in result.by_category("Hyperparameters"))
    weight, _ = CATEGORIES["Hyperparameters"]
    evidence = _evidence_for(result, _HYPERPARAM_SIGNALS | {"seed"})
    pretty = {"learning_rate": "learning rate", "batch_size": "batch size"}.get
    reported = sorted(pretty(s, s.replace("_", " ")) for s in found)
    if len(found) >= 3:
        note = f"Reported: {', '.join(reported)}."
        note += " Random seed(s) reported." if has_seed else " Random seed not mentioned."
        return ChecklistItem("Hyperparameters", Verdict.PASS, weight, float(weight), evidence, note)
    if found:
        return ChecklistItem("Hyperparameters", Verdict.PARTIAL, weight, weight / 2, evidence,
                             f"Only partial hyperparameters reported: {', '.join(reported)}.")
    return ChecklistItem("Hyperparameters", Verdict.MISSING, weight, 0.0, [],
                         "No hyperparameters (lr, batch size, epochs, optimizer) found.")


def _check_compute(result: ExtractionResult) -> ChecklistItem:
    found = {f.signal for f in result.by_category("Compute")}
    has_hw = bool(found & {"gpu_type", "gpu_generic", "compute_count"})
    has_time = "compute_time" in found
    weight, _ = CATEGORIES["Compute"]
    evidence = _evidence_for(result, {"gpu_type", "gpu_generic", "compute_time", "compute_count"})
    if has_hw and has_time:
        return ChecklistItem("Compute", Verdict.PASS, weight, float(weight), evidence,
                             "Hardware and training time both disclosed.")
    if has_hw or has_time:
        return ChecklistItem("Compute", Verdict.PARTIAL, weight, weight / 2, evidence,
                             "Partial compute disclosure (hardware or time only).")
    return ChecklistItem("Compute", Verdict.MISSING, weight, 0.0, [],
                         "No compute resources disclosed.")


def _check_evaluation(result: ExtractionResult) -> ChecklistItem:
    found = {f.signal for f in result.by_category("Evaluation rigor")}
    has_metric = bool(found & {"metric_reported", "metric_value"})
    has_baseline = bool(found & {"baseline", "comparison"})
    has_stats = bool(found & {"std_reported", "error_bars", "multiple_runs", "significance"})
    weight, _ = CATEGORIES["Evaluation rigor"]
    evidence = _evidence_for(
        result, {"metric_reported", "metric_value", "baseline", "comparison",
                 "std_reported", "error_bars", "multiple_runs", "significance"}, limit=4)
    dimensions = sum([has_metric, has_baseline, has_stats])
    if dimensions == 3:
        note = "Metrics, baseline comparisons, and statistical reporting all present."
        return ChecklistItem("Evaluation rigor", Verdict.PASS, weight, float(weight), evidence, note)
    if dimensions >= 1:
        parts = []
        if has_metric: parts.append("metrics")
        if has_baseline: parts.append("baselines")
        if has_stats: parts.append("statistical reporting")
        missing = [p for p in ("metrics", "baselines", "statistical reporting") if p not in parts]
        return ChecklistItem("Evaluation rigor", Verdict.PARTIAL, weight, weight / 2, evidence,
                             f"Found: {', '.join(parts)}. Missing: {', '.join(missing)}.")
    return ChecklistItem("Evaluation rigor", Verdict.MISSING, weight, 0.0, [],
                         "No metrics, baselines, or statistical reporting detected.")


def _check_claims(result: ExtractionResult) -> ChecklistItem:
    claims = result.by_category("Claims")
    has_quant = any(f.signal == "quant_claim" for f in claims)
    weight, _ = CATEGORIES["Claims"]
    if has_quant:
        return ChecklistItem("Claims", Verdict.PASS, weight, float(weight),
                             _evidence_for(result, {"quant_claim", "claim_verb"}),
                             "Quantitative claims backed by reported numbers.")
    if claims:
        return ChecklistItem("Claims", Verdict.PARTIAL, weight, weight / 2,
                             _evidence_for(result, {"claim_verb"}),
                             "Claims present but without quantified backing in the scanned text.")
    return ChecklistItem("Claims", Verdict.MISSING, weight, 0.0, [],
                         "No evaluative claims detected in the scanned text.")


def build_checklist(result: ExtractionResult, paper_id: str = "", llm_audit: dict | None = None) -> ReproducibilityReport:
    """Build the full checklist + score from an ExtractionResult."""
    items = [
        _check_data(result),
        _check_code(result),
        _check_hyperparameters(result),
        _check_compute(result),
        _check_evaluation(result),
        _check_claims(result),
    ]
    total = sum(i.points for i in items)
    score = int(round(total))
    return ReproducibilityReport(
        paper_id=paper_id,
        title=result.title,
        score=score,
        items=items,
        finding_count=len(result.findings),
        llm_audit=llm_audit,
    )
