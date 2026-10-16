"""`jevmon-lab`: run the process-launch lab and produce a labelled dataset plus report.

  jevmon-lab run   [--smoke] [--only FAMILY,ID] [--manifest PATH] [--out DIR]
                   [--judge none|fake|laya] [--config PATH]
  jevmon-lab report --in DIR [--judge none|fake|laya] [--config PATH]

`run` brings up the rootless podman lab, drives the scenario timeline while the real exec
pipeline (host /proc poller + context builder) records every observed launch, then judges the
recorded contexts according to `--judge` and writes, under the output directory (default
`lab-data/`, gitignored):

- `contexts.jsonl`  raw contexts with their scenario tag (the ground truth join)
- `scenarios.jsonl` ground truth by construction + observed/missed launch counts
- `dataset.jsonl`   the labelled dataset rows (context, signature, label, family, split, verdict)
- `report.md`       per family / per category, confusion matrix, FPR, latency, honest limits
- `meta.json`       run + poller + judge statistics

`report` re-judges a stored run with a different judge mode without re-running podman. The lab
never touches the real journal, the daemon's state or anything outside its own containers.
"""

import argparse
import json
from pathlib import Path

from jevmon.config import load_config
from jevmon_lab import dataset, manifest, runner

DEFAULT_MANIFEST = Path("lab/scenarios.toml")
DEFAULT_DATA = Path("lab-data")


def _add_judge(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--judge",
        choices=dataset.JUDGE_MODES,
        default="none",
        help="none: contexts only; fake: deterministic stand-in; laya: the real model",
    )
    p.add_argument(
        "--config", type=Path, default=None, help="config file ([judge], [context], ...)"
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jevmon-lab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the lab and capture + judge a dataset")
    r.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    r.add_argument("--out", type=Path, default=DEFAULT_DATA)
    r.add_argument("--only", default="", help="comma-separated scenario families or ids")
    r.add_argument("--smoke", action="store_true", help="the 3-scenario smoke set")
    _add_judge(r)
    g = sub.add_parser("report", help="re-judge a stored run and rebuild dataset + report")
    g.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_DATA)
    g.add_argument("--out", type=Path, default=None, help="defaults to --in")
    _add_judge(g)
    return p


def _judge_and_write(
    rows: list[dict],
    records: tuple[dict, ...],
    poller: dict,
    meta: dict,
    out_dir: Path,
    mode: str,
    config,
) -> dict:
    stats = dataset.judge_rows(rows, mode, config=config)
    dataset.write_dataset(rows, out_dir / "dataset.jsonl")
    report = dataset.build_report(records, rows, poller, stats, meta)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    latencies = stats.get("latencies_ms", [])
    judge_meta = {k: v for k, v in stats.items() if k != "latencies_ms"}
    judge_meta["mean_ms"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    judge_meta["latency_p50_ms"] = dataset.percentile_ms(latencies, 0.50)
    judge_meta["latency_p95_ms"] = dataset.percentile_ms(latencies, 0.95)
    meta = {**meta, "judge": judge_meta}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return stats


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    mf = manifest.load(args.manifest)
    only = tuple(x for x in args.only.split(",") if x)
    result = runner.run(mf, args.out, only=only, smoke=args.smoke, config=cfg)
    rows = dataset.build_rows(result.observations, result.records, cfg.context)
    stats = _judge_and_write(
        rows, result.records, result.poller, result.meta, Path(args.out), args.judge, cfg
    )
    print(
        json.dumps(
            {
                "run_id": result.meta["run_id"],
                "out": str(args.out),
                "scenarios": len(result.records),
                "contexts": len(rows),
                "labelled": sum(1 for r in rows if r["label"]),
                "judge": stats.get("mode"),
                "judged": stats.get("judged", 0),
                "miss_ratio": result.poller.get("miss_ratio", 0),
            },
            indent=2,
        )
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out) if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    records = [json.loads(ln) for ln in (in_dir / "scenarios.jsonl").read_text().splitlines() if ln]
    meta = json.loads((in_dir / "meta.json").read_text())
    poller = meta.get("poller", {})
    rows: list[dict] = []
    contexts_path = in_dir / "contexts.jsonl"
    if contexts_path.exists():
        raw = [json.loads(ln) for ln in contexts_path.read_text().splitlines() if ln]
        observations = _observations_from_raw(raw)
        rows = dataset.build_rows(observations, records, cfg.context)
    stats = _judge_and_write(rows, tuple(records), poller, meta, out_dir, args.judge, cfg)
    print(
        json.dumps(
            {
                "in": str(in_dir),
                "out": str(out_dir),
                "contexts": len(rows),
                "judge": stats.get("mode"),
                "judged": stats.get("judged", 0),
            },
            indent=2,
        )
    )
    return 0


def _observations_from_raw(raw: list[dict]):
    """Rebuild Observation objects from stored context rows (state text is all the judge needs)."""
    from jevmon.detect.context import Context
    from jevmon_lab.capture import Observation

    out = []
    for row in raw:
        context = Context(
            time=float(row["time"]),
            kind=str(row.get("kind", "exec")),
            subject=str(row.get("subject", "")),
            state=str(row["context"]),
            signature=str(row.get("signature", "")),
            fields={},
        )
        out.append(
            Observation(
                context,
                row.get("scenario_id"),
                str(row.get("container", "")),
                row.get("designated"),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "report":
        return _report(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
