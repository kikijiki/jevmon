"""Frozen features of any Hugging Face backbone for the shipped context text, cached on disk.

    python scripts/backbone_embed.py NAME MODEL_ID_OR_PATH [--kind decoder|encoder] [--instruct]
        [--data DIR ...] [--max-length 512] [--device cuda]

One forward pass stores several pooled variants per text (the shipped recipes read `mean+mean75`):
decoders give the last token and the mean of the final layer and of a layer at 3/4
depth; encoders give mean and cls of the last layer. Left padding for decoders, the end token
appended after truncation (what the Qwen3-Embedding card prescribes). Nothing here reads labels.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
CACHE_DIR = Path("cache")
INSTRUCTION = (
    "Instruct: Classify this process launch on a Linux workstation as safe, warn or crit\nQuery: "
)


def cache_path(name: str, max_length: int) -> Path:
    return CACHE_DIR / f"bb-{name}-{max_length}.pkl"


def _load(model_id: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
    return tok, model.to(device).eval()


def _text_model(model):
    """The text stack of a multimodal wrapper (Qwen3.5 / openjev), else the model itself."""
    for attr in ("language_model",):
        inner = getattr(model, attr, None)
        if inner is not None:
            return inner
    inner = getattr(getattr(model, "model", None), "language_model", None)
    return inner if inner is not None else model


def embed(
    name: str,
    model_id: str,
    texts: list[str],
    *,
    kind: str = "decoder",
    instruct: bool = False,
    max_length: int = 512,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """{variant: (n, dim) float32} for `texts`; computed once per text and cached per name."""
    path = cache_path(name, max_length)
    CACHE_DIR.mkdir(exist_ok=True)
    store: dict[str, dict[str, np.ndarray]] = (
        pickle.loads(path.read_bytes()) if path.exists() else {}
    )
    missing = sorted({t for t in texts if t not in store})
    if missing:
        started = time.perf_counter()
        tok, model = _load(model_id, device)
        stack = _text_model(model)
        prefix = INSTRUCTION if instruct else ""
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else (eos or 0)
        order = sorted(missing, key=len)
        for s in range(0, len(order), batch_size):
            chunk = order[s : s + batch_size]
            if kind == "decoder":
                ids = [
                    tok(prefix + t, add_special_tokens=False)["input_ids"][: max_length - 1] + [eos]
                    for t in chunk
                ]
                width = max(len(i) for i in ids)
                inp = torch.tensor([[pad] * (width - len(i)) + i for i in ids])
                att = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids])
            else:
                enc = tok(
                    chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                )
                inp, att = enc["input_ids"], enc["attention_mask"]
            with torch.inference_mode():
                out = stack(
                    input_ids=inp.to(device),
                    attention_mask=att.to(device),
                    output_hidden_states=True,
                )
                hs = out.hidden_states
                m = att.to(device).unsqueeze(-1).float()
                n_layers = len(hs) - 1
                mid = hs[(3 * n_layers) // 4].float()
                last = out.last_hidden_state.float()

                def mean(h, m=m):
                    return ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()

                feats = {"mean": mean(last), "mean75": mean(mid)}
                if kind == "decoder":
                    feats["last"] = last[:, -1].cpu().numpy()
                    feats["last75"] = mid[:, -1].cpu().numpy()
                else:
                    feats["cls"] = last[:, 0].cpu().numpy()
            for j, t in enumerate(chunk):
                store[t] = {k: v[j].astype(np.float16) for k, v in feats.items()}
        path.write_bytes(pickle.dumps(store))
        print(
            f"  {name}: embedded {len(missing)} texts in {time.perf_counter() - started:.1f}s "
            f"(device {device})",
            flush=True,
        )
        del model, stack
        torch.cuda.empty_cache()
    variants = store[texts[0]].keys()
    return {v: np.stack([store[t][v] for t in texts]).astype(np.float32) for v in variants}


def main() -> None:
    from improve_lib import load_rows, make_data, render

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("model")
    ap.add_argument("--kind", choices=("decoder", "encoder"), default="decoder")
    ap.add_argument("--instruct", action="store_true")
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    rows = []
    for d in args.data:
        rows += load_rows(d / "dataset.jsonl")
    _, texts = make_data(rows, lambda r: render(r, "shipped"))
    uniq = sorted(set(texts))
    print(f"{args.name}: {len(uniq)} unique texts", flush=True)
    embed(
        args.name,
        args.model,
        uniq,
        kind=args.kind,
        instruct=args.instruct,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
