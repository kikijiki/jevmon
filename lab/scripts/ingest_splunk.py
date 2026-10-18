"""Convert Splunk attack_data Linux logs (Sysmon for Linux EventID 1, auditd EXECVE) to launches.

    python lab/scripts/ingest_splunk.py --src data/external/splunk-attack-data-linux \
        --out lab-data-ext-splunk [--provisional lab/ingest/splunk_provisional_labels.json]

One row per distinct launch (exe + argv template). Real exe, argv, cwd, user and parent are kept
when the log has them (sysmon); auditd EXECVE records give argv only. The only label source is the
technique folder the log sits in (`lab/ingest/splunk.toml`: technique -> tactic -> class). The same
logs hold background launches (polling agents, package hooks), so the folder label is weak:
launches that repeat inside a log or appear under several technique folders are role `background`
with no label, and a seeded sample of about 300 rows goes to a review sheet. `--provisional` merges
labels written for that sheet; they are provisional and unreviewed. Nothing is executed.
"""

import argparse
import csv
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from jevmon_lab import ingest

DATA = re.compile(r'<Data Name="([^"]+)">(.*?)</Data>', re.DOTALL)
EVENT = re.compile(r"<Event.*?</Event>", re.DOTALL)
USER_ID = re.compile(r'<Security UserId="(\d+)"')
AUDIT = re.compile(r"type=(\w+) msg=audit\((.*?):(\d+)\)\s*:?\s*(.*)")
ARG = re.compile(r"(?:^|\s)a(\d+)=")
TECHNIQUE = re.compile(r"(T\d{4})(\.\d{3})?")
DRIVER = {"exe": "/usr/bin/dash", "argv0": "sh"}


def sysmon_launches(text: str):
    for ev in EVENT.findall(text):
        if "<EventID>1</EventID>" not in ev:
            continue
        d = {k: html.unescape(v) for k, v in DATA.findall(ev)}
        cmd = d.get("CommandLine", "")
        if not cmd or cmd == "-" or not d.get("Image"):
            continue
        uid = USER_ID.search(ev)
        parent = d.get("ParentImage", "")
        yield ingest.Launch(
            exe=d["Image"],
            argv=cmd.split(),
            cwd=d.get("CurrentDirectory", "") if d.get("CurrentDirectory", "-") != "-" else "",
            uid=int(uid.group(1)) if uid else None,
            user=d.get("User", "") if d.get("User", "-") != "-" else "",
            parent_exe=parent if parent != "-" else "",
            parent_argv=(
                d.get("ParentCommandLine", "") if d.get("ParentCommandLine", "-") != "-" else ""
            ),
        )


def auditd_launches(text: str):
    for line in text.splitlines():
        m = AUDIT.match(line)
        if not m or m.group(1) != "EXECVE":
            continue
        rest = m.group(4)
        pos = [(mm.start(), int(mm.group(1)), mm.end()) for mm in ARG.finditer(rest)]
        args: list[str] = []
        for i, (_s, _n, end) in enumerate(pos):
            stop = pos[i + 1][0] if i + 1 < len(pos) else len(rest)
            val = rest[end:stop].rstrip()
            if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            args.append(val)
        if args:
            yield ingest.Launch(exe=args[0], argv=args)


def tactic_of(folder: str, table: dict) -> str:
    for prefix, tactic in table["folder_tactic"].items():
        if folder == prefix or folder.startswith(prefix + "/"):
            return tactic
    m = TECHNIQUE.search(folder)
    return table["technique_tactic"].get(m.group(1), "") if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lab", type=Path, nargs="*", default=None, help="lab sources for dedupe")
    ap.add_argument("--provisional", type=Path, default=None)
    args = ap.parse_args()
    table = ingest.load_toml("splunk")
    lab = ingest.load_lab_keys(args.lab if args.lab is not None else ingest.DEFAULT_LAB_SOURCES)
    root = args.src / "datasets"

    # 1. parse every log; remember where each launch key was seen
    seen: dict[str, dict] = {}
    stats = Counter()
    for path in sorted(root.rglob("*.log")):
        folder = str(path.parent.relative_to(root))
        text = path.read_text(errors="replace")
        is_sysmon = "<Event" in text[:300]
        launches = sysmon_launches(text) if is_sysmon else auditd_launches(text)
        per_file: Counter = Counter()
        firsts: dict[str, ingest.Launch] = {}
        for la in launches:
            stats["sysmon_eid1" if is_sysmon else "auditd_execve"] += 1
            key = ingest.template_key(la.argv)
            per_file[key] += 1
            firsts.setdefault(key, la)
        for key, la in firsts.items():
            rec = seen.setdefault(
                key,
                {"launch": la, "folders": set(), "files": set(), "max_repeat": 0, "events": 0},
            )
            rec["folders"].add(folder)
            rec["files"].add(str(path.relative_to(root)))
            rec["max_repeat"] = max(rec["max_repeat"], per_file[key])
            rec["events"] += per_file[key]
            if la.parent_exe and not rec["launch"].parent_exe:
                rec["launch"] = la
    stats["distinct_launches"] = len(seen)

    # 2. label from the folder, mark background by repetition across folders / inside a log
    rows: list[ingest.Row] = []
    for key, rec in sorted(seen.items()):
        folders = sorted(rec["folders"])
        background = (
            len(folders) >= table["background_min_folders"]
            or rec["max_repeat"] >= table["background_min_repeats"]
        )
        tactic = tactic_of(folders[0], table)
        target, rule, why = ingest.apply_rules(table["rule"], {"tactic": tactic})
        label = None if background or target == "null" else target
        la = rec["launch"]
        row = ingest.Row(
            "splunk-attack-data-linux",
            "background" if background else "test",
            "launch",
            la,
            label,
            "background-repeat" if background else rule,
            "" if background else why,
            family=folders[0],
            count=rec["events"],
            source_label={"folders": folders, "tactic": tactic, "weak": target},
            extra={"files": sorted(rec["files"])[:3], "max_repeat": rec["max_repeat"]},
        )
        row.key = key
        rows.append(row)
    rows, dd = ingest.dedupe(rows, forbidden=lab)

    # 3. review sample: round-robin over folders (so every technique appears), plus background
    rng = random.Random(table["seed"])
    size = table["review_size"]
    replay = [r for r in rows if r.role == "test"]
    bg = [r for r in rows if r.role == "background"]
    by_folder: dict[str, list[ingest.Row]] = defaultdict(list)
    for r in replay:
        by_folder[r.family].append(r)
    for v in by_folder.values():
        rng.shuffle(v)
    n_bg = min(len(bg), size // 8)
    picks: list[ingest.Row] = []
    while len(picks) < size - n_bg and any(by_folder.values()):
        for folder in sorted(by_folder):
            if by_folder[folder] and len(picks) < size - n_bg:
                picks.append(by_folder[folder].pop())
    picks += rng.sample(bg, n_bg)
    review_ids = {r.ident() for r in picks}

    provisional: dict = {}
    if args.provisional and args.provisional.exists():
        provisional = json.loads(args.provisional.read_text())["labels"]
    for r in rows:
        r.extra["in_review"] = r.ident() in review_ids
        if r.ident() in provisional:
            lab_, note = provisional[r.ident()]
            r.extra["provisional_label"] = lab_
            r.extra["provisional_note"] = note
            r.extra["review_status"] = "provisional, unreviewed (written by the assistant)"

    rows = ingest.render_all(rows)
    meta = {
        "source": table["source"],
        "mapping": "lab/ingest/splunk.toml",
        "lab_keys": len(lab),
        "parsed": dict(stats),
        "dedupe": dd,
        "review_rows": len(review_ids),
        "provisional_labels": len(provisional),
        "summary": ingest.summarize(rows),
        "note": "folder labels are weak; headline numbers use the reviewed sample",
    }
    ingest.write_dataset(args.out, rows, meta)

    sheet = args.out / "review_sheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "id", "technique_folder", "tactic", "folder_label", "role", "user", "parent",
                "exe", "command_line", "events", "assistant_label_unreviewed", "assistant_note",
                "reviewer_label", "reviewer_note",
            ]
        )  # fmt: skip
        for r in sorted(
            (r for r in rows if r.extra["in_review"]), key=lambda r: (r.family, r.ident())
        ):
            la = r.launch
            w.writerow(
                [
                    r.ident(), r.family, r.source_label["tactic"], r.source_label["weak"], r.role,
                    la.user, la.parent_exe, la.exe, " ".join(la.argv), r.count,
                    r.extra.get("provisional_label", ""), r.extra.get("provisional_note", ""),
                    "", "",
                ]
            )  # fmt: skip
    print(json.dumps(meta["summary"], indent=1, sort_keys=True))
    print(stats, dd, "review", len(review_ids))


if __name__ == "__main__":
    main()
