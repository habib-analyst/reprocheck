"""Offline reproducibility-signal extractor.

No network, no API keys. Scans title + abstract + optional full text with
regex/NLP-lite heuristics and returns every finding with the exact matched
text span as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CONTEXT_WINDOW = 160  # characters of surrounding context kept as evidence

# ---------------------------------------------------------------------------
# Signal patterns: (signal_name, category, compiled_regex)
# ---------------------------------------------------------------------------

_CODE_PATTERNS = [
    ("code_url", r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|huggingface\.co|colab\.research\.google\.com)/[^\s)\]]+"),
    ("code_available", r"\bcode\s+(?:is|are)\s+(?:made\s+)?publicly\s+available\b|\bcode\s+available\s+at\b|\bour\s+code\b|\bimplementation\s+available\b"),
]

_DATASET_NAMES = (
    "ImageNet|CIFAR-?10|CIFAR-?100|MNIST|MedMNIST|Fashion-?MNIST|COCO|SQuAD|"
    "GLUE|SuperGLUE|ImageNet-?21k|LAION|BraTS|CheXpert|MIMIC(?:-IV|-CXR)?|"
    "HAM10000|ISIC|Camelyon16|Camelyon17|TCGA|PanNuke|MoNuSeg|Kather|"
    "PathMNIST|BloodMNIST|OrganMNIST|DermaMNIST|OCTMNIST|PneumoniaMNIST|"
    "RetinaMNIST|BreastMNIST|TissueMNIST|ChestMNIST|NoduleMNIST|AdrenalMNIST|"
    "FractureMNIST|VesselMNIST|SynapseMNIST|Kinetics|Something-?Something|"
    "UCF101|HMDB51|LibriSpeech|CommonVoice|WMT|IWSLT|OpenWebText|The Pile|"
    "RedPajama|MSMARCO|NQ|TriviaQA|HotpotQA|Cityscapes|ADE20K|Pascal VOC|"
    "Kitti|Waymo|nuScenes|ModelNet|ShapeNet|ABIDE|ADNI|OASIS|UK Biobank|"
    "TCIA|LIDC-IDRI|DeepLesion|NIH ChestX-ray|PadChest|VinDr|EyePACS|"
    "Messidor|DRIVE|STARE|CHASE_DB1|REFUGE|RIM-ONE|G1020|Kaggle|"
    "Synapse|BTCV"
)

_DATASET_PATTERNS = [
    ("dataset_named", rf"\b(?:{_DATASET_NAMES})\b"),
    ("dataset_generic", r"\b(?:we\s+(?:use|evaluate|train|test|pretrain)\s+(?:on|with)|evaluated?\s+on|benchmark(?:ed|s)?\s+on)\s+(?:the\s+)?([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){{0,2}})\s+(?:dataset|benchmark|corpus)\b"),
    ("dataset_size", r"\b\d[\d,]*(?:\.\d+)?\s*(?:k|M|million|thousand)?\s*(?:[A-Za-z\-]+\s+){0,2}?(?:images?|samples?|scans?|slices?|patients?|cases?|examples?|records?|instances?|volumes?)\b"),
]

_METRIC_PATTERNS = [
    ("metric_reported", r"\b(?:accuracy|top-?1|top-?5|AUC|AUROC|AUPRC|Dice(?:\s+score)?|F1(?:\s+score)?|mAP|mean\s+average\s+precision|precision|recall|IoU|mIoU|sensitivity|specificity|MAE|MSE|RMSE|R\^?2|BLEU|ROUGE(?:-L)?|METEOR|perplexity|FID|SSIM|PSNR|Hausdorff)\b"),
    ("metric_value", r"\b(?:accuracy|AUC|Dice|F1|mAP|IoU|MAE)\b[^.]{0,40}?\b(?:of|is|was|at|by|reaches?|achieves?|=|:)\s*~?\d+(?:\.\d+)?%?\b"),
]

_HYPERPARAM_PATTERNS = [
    ("learning_rate", r"\blearning\s+rate\b|\blr\b(?=\s*[=:]|\s+of)|\blr\s*=\s*[\d.eE\-+]+\b"),
    ("batch_size", r"\bbatch\s+size\b|\bbatch\s*=\s*\d+\b"),
    ("epochs", r"\bepochs?\b(?!\s+of\s+training\s+data)|\btrain(?:ed|ing)?\s+for\s+\d+\s+epochs?\b"),
    ("optimizer", r"\b(?:AdamW?|SGD|RMSprop|Adagrad|Adadelta|NAdam|LAMB|LARS|Lion)\b|\boptimizer\b"),
    ("weight_decay", r"\bweight\s+decay\b"),
    ("dropout", r"\bdropout\b"),
    ("scheduler", r"\b(?:cosine|step|linear|warmup|warm-?up)\s+(?:scheduler|schedule|annealing|decay)\b"),
]

_SEED_PATTERNS = [
    ("seed", r"\brandom\s+seeds?\b|\bseed(?:s|ed)?\s*(?:=|:|\d)|\bfixed\s+seed\b"),
]

_COMPUTE_PATTERNS = [
    ("gpu_type", r"\b(?:A100|H100|V100|P100|T4|K80|A40|A6000|RTX\s*\d{3,4}(?:\s*Ti)?|GTX\s*\d{3,4}|TPU\s*v?\d)\b"),
    ("gpu_generic", r"\bGPUs?\b|\bTPUs?\b"),
    ("compute_time", r"\b\d+(?:\.\d+)?\s*(?:GPU|TPU)?\s*-?hours?\b|\btrain(?:ed|ing)?\s+(?:for|in|takes?)\s+\d+\s+(?:hours?|days?)\b"),
    ("compute_count", r"\b\d+\s*x\s*(?:GPUs?|TPUs?)\b|\bon\s+\d+\s+(?:GPUs?|TPUs?)\b"),
]

_STAT_PATTERNS = [
    ("std_reported", r"\bstd(?:\.|dev)?\b|\bstandard\s+deviation\b|±\s*\d"),
    ("error_bars", r"\berror\s+bars?\b|\bconfidence\s+intervals?\b|\b95%\s*CI\b"),
    ("multiple_runs", r"\b\d+\s+runs?\b|\brepeated\s+\d+\s+times\b|\bover\s+(?:\d+|[a-z]+)\s+(?:independent\s+)?runs?\b|\baveraged?\s+over\b"),
    ("significance", r"\bp-?value\b|\bstatistical(?:ly)?\s+significant\b|\bpaired\s+t-?test\b|\bWilcoxon\b"),
]

_BASELINE_PATTERNS = [
    ("baseline", r"\bbaselines?\b"),
    ("comparison", r"\bcompar(?:ed|ison)\s+(?:with|to|against)\b|\boutperform(?:s|ed|ing)?\b|\bstate-?of-?the-?art\b|\bSOTA\b"),
]

_CLAIM_PATTERNS = [
    ("quant_claim", r"\b(?:achiev(?:e|es|ed|ing)|reach(?:es|ed|ing)|obtain(?:s|ed|ing)|improv(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|gain(?:s|ed|ing)?\s+of)\b[^.]{0,120}?\d+(?:\.\d+)?%"),
    ("claim_verb", r"\bwe\s+(?:show|demonstrate|prove|find|propose|introduce|present)\b"),
]

# category -> list of pattern groups
_PATTERN_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Code", _CODE_PATTERNS),
    ("Data", _DATASET_PATTERNS),
    ("Hyperparameters", _HYPERPARAM_PATTERNS),
    ("Hyperparameters", _SEED_PATTERNS),
    ("Compute", _COMPUTE_PATTERNS),
    ("Evaluation rigor", _STAT_PATTERNS),
    ("Evaluation rigor", _METRIC_PATTERNS),
    ("Evaluation rigor", _BASELINE_PATTERNS),
    ("Claims", _CLAIM_PATTERNS),
]


@dataclass
class Finding:
    """One reproducibility signal found in the paper text."""

    signal: str          # e.g. "learning_rate"
    category: str        # one of Data/Code/Hyperparameters/Compute/Evaluation rigor/Claims
    matched: str         # exact matched text span
    evidence: str        # matched span plus surrounding context
    start: int           # char offset in the scanned text
    end: int


@dataclass
class ExtractionResult:
    """All findings from a scan, plus the scanned sections."""

    title: str
    abstract: str
    full_text: str = ""
    findings: list[Finding] = field(default_factory=list)

    def by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def signals(self) -> set[str]:
        return {f.signal for f in self.findings}

    def scanned_text(self) -> str:
        parts = [self.title, self.abstract, self.full_text]
        return "\n\n".join(p for p in parts if p)


def _evidence(text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_WINDOW // 2)
    hi = min(len(text), end + CONTEXT_WINDOW // 2)
    snippet = text[lo:hi].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if lo > 0:
        snippet = "… " + snippet
    if hi < len(text):
        snippet = snippet + " …"
    return snippet


def extract_signals(title: str, abstract: str, full_text: str = "") -> ExtractionResult:
    """Scan title + abstract (+ optional full text) for reproducibility signals.

    Pure function: no network, no API keys, deterministic.
    """
    result = ExtractionResult(title=title or "", abstract=abstract or "", full_text=full_text or "")
    text = result.scanned_text()
    seen: set[tuple[str, int, int]] = set()

    for category, group in _PATTERN_GROUPS:
        for signal_name, pattern in group:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue  # a bad pattern must never crash extraction
            for match in regex.finditer(text):
                start, end = match.span()
                key = (signal_name, start, end)
                if key in seen:
                    continue
                seen.add(key)
                matched = match.group(0)
                # Trim overlong generic matches (e.g. claim verbs with 120 chars).
                if len(matched) > 200:
                    matched = matched[:200] + "…"
                result.findings.append(
                    Finding(
                        signal=signal_name,
                        category=category,
                        matched=matched.strip(),
                        evidence=_evidence(text, start, end),
                        start=start,
                        end=end,
                    )
                )

    result.findings.sort(key=lambda f: (f.start, f.end))
    return result
