"""Convert QuasarNix (synthesized Linux reverse-shell strings, plus NL2Bash) to the launch schema.

    python lab/scripts/ingest_quasarnix.py --src data/external/hf-quasarnix \
        --out lab-data-ext-quasarnix

The malicious file is 235k strings from a few dozen generator templates with random address, port,
shell, FIFO path and variable names. Strings are collapsed to template families with the masks in
`lab/ingest/quasarnix.toml`, deduplicated on the launch key, and capped per family (seeded). All of
them are crit by the mapping table. A sample of `nl2bash.json` is kept unlabelled (role
`background`) to see what the judge says on ordinary one-liners. Nothing is executed.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from jevmon_lab import ingest

DRIVER = {"exe": "/usr/bin/dash", "argv0": "sh"}


def read_strings(path: Path) -> tuple[list[str], int]:
    """One command per line of a JSON list; lines with invalid escapes are read verbatim."""
    out: list[str] = []
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().rstrip(",")
        if line in ("[", "]", ""):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            bad += 1
            value = line[1:-1] if line.startswith('"') and line.endswith('"') else line
        if isinstance(value, str) and value.strip():
            out.append(value)
    return out, bad


_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_NUM = re.compile(r"\d+")


def variant_key(command: str) -> str:
    """Dedupe key inside this source: address and numbers masked, shell and names kept, so the
    shell/protocol/variable variants of one family stay distinct (the cap then picks among them)."""
    return _NUM.sub("<N>", _IP.sub("<IP>", command))


def family_of(command: str, masks: list[tuple[re.Pattern, str]]) -> str:
    for pattern, to in masks:
        command = pattern.sub(to, command)
    return command


def drop_lab(rows: list[ingest.Row], lab: set[str]) -> list[ingest.Row]:
    """Remove rows whose launch template (the shipped masking) is already a lab template, then
    re-key the rest on the finer in-source key for deduplication."""
    kept = [r for r in rows if r.key not in lab]
    for r in kept:
        r.extra["template_key"] = r.key
        r.key = variant_key(" ".join(r.launch.argv))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lab", type=Path, nargs="*", default=None, help="lab sources for dedupe")
    args = ap.parse_args()
    table = ingest.load_toml("quasarnix")
    shell = ingest.load_toml("shell")
    lab = ingest.load_lab_keys(args.lab if args.lab is not None else ingest.DEFAULT_LAB_SOURCES)
    masks = [(re.compile(m["pattern"]), m["to"]) for m in table["family_mask"]]
    rng = random.Random(table["seed"])

    mal, bad = read_strings(args.src / "X_test_malicious_cmd_orig.json")
    target, rule, why = ingest.apply_rules(table["rule"], {"file": "malicious"})
    fam_names: dict[str, str] = {}
    by_family: dict[str, dict[str, str]] = defaultdict(dict)  # family -> key -> command
    for command in mal:
        fam = family_of(command, masks)
        name = fam_names.setdefault(fam, f"qn{len(fam_names):02d}")
        by_family[name].setdefault(variant_key(command), command)
    rows: list[ingest.Row] = []
    cap = table["per_family_cap"]
    for name in sorted(by_family):
        pool = sorted(by_family[name].items())
        picked = rng.sample(pool, min(cap, len(pool)))
        for _key, command in sorted(picked):
            rows += ingest.command_rows(
                command,
                source=table["source"],
                role="test",
                label=target,
                rule=rule,
                uncertain=why,
                family=name,
                source_label={"file": "X_test_malicious_cmd_orig.json"},
                driver=DRIVER,
                shell=shell,
            )
    rows, dd = ingest.dedupe(drop_lab(rows, lab), forbidden=set())

    benign, _ = read_strings(args.src / "nl2bash.json")
    btarget, brule, bwhy = ingest.apply_rules(table["rule"], {"file": "nl2bash"})
    benign = sorted(set(benign))
    sample = rng.sample(benign, min(table["benign_sample"], len(benign)))
    brows: list[ingest.Row] = []
    for command in sample:
        brows += ingest.command_rows(
            command,
            source=table["source"],
            role="background",
            label=None if btarget == "null" else btarget,
            rule=brule,
            uncertain=bwhy,
            family="nl2bash",
            source_label={"file": "nl2bash.json"},
            driver=DRIVER,
            shell=shell,
        )
    brows, bdd = ingest.dedupe(drop_lab(brows, lab), forbidden=set())

    out = ingest.render_all(rows + brows)
    meta = {
        "source": table["source"],
        "mapping": "lab/ingest/quasarnix.toml",
        "lab_keys": len(lab),
        "malicious_strings": len(mal),
        "unparsed_lines": bad,
        "families": len(by_family),
        "unique_keys_per_family": {k: len(v) for k, v in sorted(by_family.items())},
        "dedupe_malicious": dd,
        "dedupe_benign": bdd,
        "summary": ingest.summarize(out),
        "note": "single class (crit) test: recall only; nl2bash rows are unlabelled",
    }
    ingest.write_dataset(args.out, out, meta)
    print(json.dumps(meta["summary"], indent=1, sort_keys=True))
    print(meta["families"], meta["unique_keys_per_family"])


if __name__ == "__main__":
    main()
