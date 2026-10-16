"""Shared pieces for the head-improvement experiments.

Everything here works on cached frozen-Laya embeddings, so a model variant costs seconds and the
GPU is only used once per distinct context text. Selection is always family-grouped
cross-validation inside the train families; held-out families are scored only through `final`.
"""

import hashlib
import json
import pickle
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS, split_of

LABEL_ALIASES = {"unusual-but-benign": "unusual"}
CACHE_DIR = Path("cache")
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
    extra: dict = field(default_factory=dict)

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


# ---------------------------------------------------------------- embeddings


class EmbeddingCache:
    """text -> vector, per (checkpoint, pooling, max_length); filled lazily from the GPU."""

    def __init__(self, agent_factory: Callable[[], object], tag: str = "english") -> None:
        self._factory = agent_factory
        self._agent = None
        self.tag = tag
        CACHE_DIR.mkdir(exist_ok=True)

    def _path(self, pooling: str, max_length: int) -> Path:
        return CACHE_DIR / f"emb-{self.tag}-{pooling}-{max_length}.pkl"

    def get(self, texts: Sequence[str], pooling: str = "mean", max_length: int = 512) -> np.ndarray:
        from jevmon.laya.head import embed_states

        path = self._path(pooling, max_length)
        store: dict[str, np.ndarray] = pickle.loads(path.read_bytes()) if path.exists() else {}
        missing = sorted({t for t in texts if t not in store})
        if missing:
            if self._agent is None:
                self._agent = self._factory()
            started = time.perf_counter()
            vecs = embed_states(
                self._agent, missing, pooling=pooling, max_length=max_length, batch_size=32
            )
            for t, v in zip(missing, vecs, strict=True):
                store[t] = v
            path.write_bytes(pickle.dumps(store))
            print(
                f"  embedded {len(missing)} new texts [{pooling}] "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        return np.stack([store[t] for t in texts]).astype(np.float32)


def laya_agent(checkpoint: str = "english"):
    from dataclasses import replace as _replace

    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    cfg = load_config()
    cfg = _replace(cfg, judge=_replace(cfg.judge, checkpoint=checkpoint))
    decider = build_decider(cfg)
    decider.load()
    if decider.error:
        raise SystemExit(f"model failed to load: {decider.error}")
    agent = decider._agent(decider._router)
    if agent is None:
        raise SystemExit("no loaded agent")
    return agent


# ---------------------------------------------------------------- classifiers


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


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
    """One classifier configuration (all knobs a sweep varies)."""

    kind: str = "logreg"  # logreg | mlp | knn | proto
    wd: float = 0.01
    balanced: bool = True
    smoothing: float = 0.0
    norm: bool = False  # L2-normalise embeddings first
    standardize: bool = False  # z-score with train statistics
    hidden: int = 256
    dropout: float = 0.2
    epochs: int = 200
    lr: float = 1e-3
    k: int = 5
    temp: float = 0.05
    seed: int = 0
    sample: str = "rows"  # rows | unique | sqrt: how repeated (text, label) rows are weighted
    tag: str = ""

    def name(self) -> str:
        parts = [self.kind]
        if self.kind == "logreg":
            parts += [f"wd{self.wd}", "bal" if self.balanced else "nobal"]
            if self.smoothing:
                parts.append(f"ls{self.smoothing}")
        elif self.kind == "mlp":
            parts += [
                f"h{self.hidden}",
                f"d{self.dropout}",
                f"wd{self.wd}",
                f"e{self.epochs}",
                "bal" if self.balanced else "nobal",
            ]
        elif self.kind == "knn":
            parts += [f"k{self.k}", "bal" if self.balanced else "nobal"]
        elif self.kind == "proto":
            parts += [f"t{self.temp}"]
        if self.norm:
            parts.append("l2")
        if self.standardize:
            parts.append("z")
        if self.sample != "rows":
            parts.append(self.sample)
        if self.tag:
            parts.append(self.tag)
        return "/".join(parts)


def _prep(spec: Spec, xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec.norm:
        xtr, xte = l2norm(xtr), l2norm(xte)
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
    if spec.kind == "logreg":
        return _logreg(spec, xtr, ytr, w, xte, classes)
    if spec.kind == "mlp":
        return _mlp(spec, xtr, ytr, w, xte, classes)
    if spec.kind == "knn":
        return _knn(spec, xtr, ytr, w, xte, classes)
    if spec.kind == "proto":
        return _proto(spec, xtr, ytr, w, xte, classes)
    raise ValueError(spec.kind)


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


def _mlp(spec, xtr, ytr, w, xte, classes) -> np.ndarray:
    g = torch.Generator().manual_seed(spec.seed)
    torch.manual_seed(spec.seed)
    d = xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Dropout(spec.dropout),
        torch.nn.Linear(d, spec.hidden),
        torch.nn.GELU(),
        torch.nn.Dropout(spec.dropout),
        torch.nn.Linear(spec.hidden, classes),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.wd)
    xt, yt, wt = (
        torch.from_numpy(xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(w.astype(np.float32)),
    )
    n = len(yt)
    bs = min(128, n)
    for _ in range(spec.epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            i = perm[s : s + bs]
            ce = F.cross_entropy(
                net(xt[i]), yt[i], reduction="none", label_smoothing=spec.smoothing
            )
            loss = (ce * wt[i]).sum() / wt[i].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return softmax(net(torch.from_numpy(xte)).numpy())


def _knn(spec, xtr, ytr, w, xte, classes) -> np.ndarray:
    a, b = l2norm(xtr), l2norm(xte)
    sims = b @ a.T
    k = min(spec.k, len(ytr))
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = np.zeros((len(xte), classes))
    for r in range(len(xte)):
        for j in top[r]:
            out[r, ytr[j]] += w[j] * max(sims[r, j], 0.0) ** 2
    out += 1e-6
    return out / out.sum(axis=1, keepdims=True)


def _proto(spec, xtr, ytr, w, xte, classes) -> np.ndarray:
    a, b = l2norm(xtr), l2norm(xte)
    protos = np.zeros((classes, a.shape[1]), dtype=np.float32)
    for c in range(classes):
        m = ytr == c
        if m.any():
            protos[c] = (a[m] * w[m, None]).sum(0) / w[m].sum()
    protos = l2norm(protos)
    return softmax((b @ protos.T) / spec.temp)


# ---------------------------------------------------------------- evaluation


def to_rows(data: Data, idx: np.ndarray, pred: np.ndarray) -> list[dict]:
    return [
        {"label": data.rows[i]["label"], "verdict": LABELS[int(p)]}
        for i, p in zip(idx, pred, strict=True)
    ]


def unique_rows(data: Data, idx: np.ndarray, pred: np.ndarray) -> list[dict]:
    """One row per (text, label); a text predicted differently across rows takes the majority."""
    votes: dict[str, Counter] = {}
    label_of: dict[str, str] = {}
    for i, p in zip(idx, pred, strict=True):
        k = str(data.key[i])
        votes.setdefault(k, Counter())[int(p)] += 1
        label_of[k] = data.rows[i]["label"]
    return [
        {"label": label_of[k], "verdict": LABELS[c.most_common(1)[0][0]]} for k, c in votes.items()
    ]


def score_pair(data: Data, idx: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rows": metrics.score(to_rows(data, idx, pred)),
        "unique": metrics.score(unique_rows(data, idx, pred)),
    }


def family_folds(data: Data, idx: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    """Family-grouped folds over `idx`, dealt per label (shuffled by `seed`) so every fold sees
    every class that has enough families; no family straddles two folds."""
    rng = np.random.default_rng(seed)
    label_of = {data.rows[i]["family"]: data.rows[i]["label"] for i in idx}
    fold_of: dict[str, int] = {}
    for label in sorted(set(label_of.values())):
        fams = sorted(f for f, lab in label_of.items() if lab == label)
        rng.shuffle(fams)
        offset = int(rng.integers(folds))
        for n, fam in enumerate(fams):
            fold_of[fam] = (n + offset) % folds
    out = [idx[[fold_of[data.family[i]] == f for i in idx]] for f in range(folds)]
    return [o for o in out if len(o)]


def cv_scores(
    spec: Spec,
    x: np.ndarray,
    data: Data,
    idx: np.ndarray,
    *,
    folds: int = 5,
    repeats: int = 3,
    base_seed: int = 0,
) -> dict:
    """Pooled out-of-fold macro-F1 (rows and unique) per repeat, with mean and sd over repeats."""
    rows_f1, uniq_f1, fold_means = [], [], []
    per_class = []
    for rep in range(repeats):
        parts = family_folds(data, idx, folds, base_seed + rep)
        preds = np.zeros(len(data.y), dtype=int)
        mf = []
        for val in parts:
            tr = np.setdiff1d(idx, val)
            proba = fit_predict(spec, x[tr], data.y[tr], data.key[tr], x[val])
            preds[val] = proba.argmax(1)
            mf.append(metrics.score(to_rows(data, val, preds[val]))["macro_f1"])
        pooled = np.concatenate(parts)
        sc = score_pair(data, pooled, preds[pooled])
        rows_f1.append(sc["rows"]["macro_f1"])
        uniq_f1.append(sc["unique"]["macro_f1"])
        fold_means.append(float(np.mean(mf)))
        per_class.append({k: v["recall"] for k, v in sc["unique"]["per_class"].items()})
    return {
        "rows_f1": float(np.mean(rows_f1)),
        "rows_f1_sd": float(np.std(rows_f1)),
        "unique_f1": float(np.mean(uniq_f1)),
        "unique_f1_sd": float(np.std(uniq_f1)),
        "fold_mean_f1": float(np.mean(fold_means)),
        "unique_recall": {k: float(np.mean([p.get(k, 0.0) for p in per_class])) for k in LABELS},
    }


def final(spec: Spec, x: np.ndarray, data: Data, *, seeds: Sequence[int] = (0,)) -> dict:
    """Fit on all train rows, score the held-out families once (mean over `seeds`)."""
    tr, te = data.idx("train"), data.idx("heldout")
    rows, uniq, preds = [], [], []
    for s in seeds:
        from dataclasses import replace

        proba = fit_predict(replace(spec, seed=s), x[tr], data.y[tr], data.key[tr], x[te])
        pred = proba.argmax(1)
        preds.append(pred)
        sc = score_pair(data, te, pred)
        rows.append(sc["rows"])
        uniq.append(sc["unique"])
    return {
        "rows": rows,
        "unique": uniq,
        "preds": preds,
        "rows_f1": [r["macro_f1"] for r in rows],
        "unique_f1": [u["macro_f1"] for u in uniq],
    }


def digest(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- context formats

from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments

_BASE = ("header", "exe")
FORMATS: dict[str, tuple[str, ...]] = {
    "shipped": ("header", "exe", "template", "argv"),
    "parent": ("header", "exe", "parent_cmd", "template", "argv"),
    "parent_first": ("header", "parent_cmd", "exe", "template", "argv"),
    "lineage": ("header", "exe", "parents", "parent_cmd", "template", "argv"),
    "struct": ("header", "exe", "parent_cmd", "template", "argv", "struct"),
    "seen": ("header", "exe", "parent_cmd", "template", "argv", "seen", "surprisal"),
    "all_stats": (
        "header",
        "exe",
        "parents",
        "parent_cmd",
        "template",
        "argv",
        "seen",
        "surprisal",
        "struct",
    ),
    "full": (),
}


def render(row: dict, fmt: str, model_max: int = 1000) -> str:
    settings = ContextSettings(order=FORMATS[fmt], model_max=model_max)
    return render_segments(parse_exec_state(row["_raw"]), settings)


class LayerCache:
    """text -> (layers+1, 2*dim) float16: mean-pooled and cls state of every encoder layer."""

    def __init__(self, agent_factory: Callable[[], object], tag: str = "english") -> None:
        self._factory = agent_factory
        self._agent = None
        self.path = CACHE_DIR / f"layers-{tag}.pkl"

    def get(self, texts: Sequence[str], max_length: int = 512) -> np.ndarray:
        from laya.common import encode_text

        store: dict[str, np.ndarray] = (
            pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        )
        missing = sorted({t for t in texts if t not in store})
        if missing:
            if self._agent is None:
                self._agent = self._factory()
            agent = self._agent
            started = time.perf_counter()
            for s in range(0, len(missing), 16):
                chunk = missing[s : s + 16]
                enc = encode_text(
                    agent.tok,
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(agent.device)
                mask = enc["attention_mask"].to(agent.device)
                with torch.inference_mode():
                    out = agent.model.encoder(
                        input_ids=ids, attention_mask=mask, output_hidden_states=True
                    )
                    m = mask.unsqueeze(-1).float()
                    layers = []
                    for h in out.hidden_states:
                        h = h.float()
                        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
                        layers.append(torch.cat([mean, h[:, 0]], -1))
                    arr = torch.stack(layers, 1).cpu().numpy().astype(np.float16)
                for t, a in zip(chunk, arr, strict=True):
                    store[t] = a
            self.path.write_bytes(pickle.dumps(store))
            print(
                f"  embedded {len(missing)} texts (all layers) in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        return np.stack([store[t] for t in texts])
