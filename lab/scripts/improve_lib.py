"""Shared pieces for training the shipped heads: data loading and a linear classifier.

Everything here works on cached frozen embeddings, so a head costs seconds. Nothing selects a
model: the recipes (backbone, feature variants, weight decay) are fixed in
`lab/scripts/train_heads.py`.
"""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jevmon_lab.manifest import LABELS, split_of

LABEL_ALIASES = {"unusual-but-benign": "unusual"}
torch.set_num_threads(2)


# ---------------------------------------------------------------- data


@dataclass(slots=True)
class Data:
    rows: list[dict]
    split: np.ndarray  # "train" / "heldout" per row
    family: np.ndarray
    y: np.ndarray  # class index
    key: np.ndarray  # (text, label) identity used for the unique-context scores
    labels: tuple[str, ...] = LABELS

    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def load_rows(path: Path, *, text_key: str = "context") -> list[dict]:
    """Labelled rows of a lab `dataset.jsonl`; the family hash split is recomputed, never read.

    `_raw` is the full-format capture when `contexts.jsonl` sits beside the dataset.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ctx = path.parent / "contexts.jsonl"
    if ctx.exists():
        raw = [json.loads(line) for line in ctx.read_text().splitlines() if line.strip()]
        if len(raw) == len(rows):
            for row, c in zip(rows, raw, strict=True):
                row["_raw"] = c["context"]
                row["_fields"] = c.get("fields")
    out = []
    for row in rows:
        if not row.get("label"):
            continue
        row["label"] = LABEL_ALIASES.get(row["label"], row["label"])
        row["split"] = split_of(row["family"])
        row.setdefault("_raw", row["context"])
        out.append(row)
    return out


def make_data(rows: list[dict], text_of: Callable[[dict], str]) -> tuple[Data, list[str]]:
    texts = [text_of(r) for r in rows]
    data = Data(
        rows=rows,
        split=np.asarray([r["split"] for r in rows]),
        family=np.asarray([r["family"] for r in rows]),
        y=np.asarray([LABELS.index(r["label"]) for r in rows]),
        key=np.asarray([f"{t}\x00{r['label']}" for t, r in zip(texts, rows, strict=True)]),
    )
    return data, texts


# ---------------------------------------------------------------- classifier


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def class_balance(y: np.ndarray, w: np.ndarray, classes: int) -> np.ndarray:
    """Per-sample weights that give every present class equal total weight."""
    total = np.zeros(classes)
    for c in range(classes):
        total[c] = w[y == c].sum()
    present = (total > 0).sum()
    scale = np.where(total > 0, w.sum() / (present * np.maximum(total, 1e-12)), 0.0)
    return w * scale[y]


@dataclass(slots=True)
class Spec:
    """One linear-classifier configuration: weight decay and how repeated rows are weighted."""

    wd: float = 0.01
    balanced: bool = True
    smoothing: float = 0.0
    standardize: bool = False  # z-score with train statistics
    sample: str = "rows"  # rows | unique | sqrt: how repeated (text, label) rows are weighted
    tag: str = ""

    def name(self) -> str:
        parts = ["logreg", f"wd{self.wd}", "bal" if self.balanced else "nobal"]
        if self.smoothing:
            parts.append(f"ls{self.smoothing}")
        if self.standardize:
            parts.append("z")
        if self.sample != "rows":
            parts.append(self.sample)
        if self.tag:
            parts.append(self.tag)
        return "/".join(parts)


def _prep(spec: Spec, xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec.standardize:
        mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    return xtr.astype(np.float32), xte.astype(np.float32)


def sample_weights(spec: Spec, y: np.ndarray, key: np.ndarray, classes: int) -> np.ndarray:
    """Training weights for the rows. `unique` gives each (text, label) one unit of weight in
    total (a line repeated 490 times counts once); `sqrt` softens that; `rows` keeps counts."""
    counts = Counter(key.tolist())
    n = np.asarray([counts[k] for k in key], dtype=np.float64)
    if spec.sample == "unique":
        w = 1.0 / n
    elif spec.sample == "sqrt":
        w = 1.0 / np.sqrt(n)
    else:
        w = np.ones(len(y))
    if spec.balanced:
        w = class_balance(y, w, classes)
    return w


def fit_predict(
    spec: Spec,
    xtr: np.ndarray,
    ytr: np.ndarray,
    ktr: np.ndarray,
    xte: np.ndarray,
    classes: int = len(LABELS),
) -> np.ndarray:
    """Class probabilities (n_test, classes) from a model trained on (xtr, ytr)."""
    xtr, xte = _prep(spec, xtr, xte)
    w = sample_weights(spec, ytr, ktr, classes)
    w = w / max(w.mean(), 1e-12)
    return _logreg(spec, xtr, ytr, w, xte, classes)


def _logreg(spec, xtr, ytr, w, xte, classes) -> np.ndarray:
    xt, yt, wt = (
        torch.from_numpy(xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(w.astype(np.float32)),
    )
    wgt = torch.zeros(classes, xtr.shape[1], requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.LBFGS([wgt, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(xt @ wgt.T + b, yt, reduction="none", label_smoothing=spec.smoothing)
        loss = (ce * wt).sum() / wt.sum() + 0.5 * spec.wd * (wgt * wgt).sum()
        loss.backward()
        return loss

    opt.step(closure)
    logits = xte @ wgt.detach().numpy().T + b.detach().numpy()
    return softmax(logits)


# ---------------------------------------------------------------- context format

from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments

# The shipped context layout: header, exe, argv template and redacted argv.
SHIPPED = ("header", "exe", "template", "argv")


def render(row: dict, fmt: str = "shipped", model_max: int = 1000) -> str:
    if fmt != "shipped":
        raise ValueError(f"unknown format {fmt!r}")
    settings = ContextSettings(order=SHIPPED, model_max=model_max)
    return render_segments(parse_exec_state(row["_raw"]), settings)
