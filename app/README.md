# jevmon app

The `app/` project holds everything that runs on the machine being monitored: the `/proc`
poller, the context builder, the judge, the event store, the daemon (`jevmond`), the command line
(`jevmon`) and the terminal UI (`jevmon-tui`). The lab, scenarios and experiments are in
[`../lab/`](../lab/README.md). For a short overview and screenshots see the
[root README](../README.md).

Contents: [usage](#usage) - [architecture](#architecture) - [install and run](#install-and-run) -
[configuration](#configuration) - [judge modes](#judge-modes) -
[measured performance](#measured-performance) - [limitations](#limitations)

## What it does

- Daemon (`jevmond`): polls `/proc` for new process launches, builds a short text context for
  each one, asks a small model whether the launch is `safe`, `warn` or `crit`, and stores the
  verdict with the context it was made from. It serves a Unix socket.
- TUI (`app/tui/`, Rust, Ratatui): a client of that socket. List and detail views, filtering,
  labelling, acknowledging, pausing the judge.
- `jevmon`: `exec` runs the same pipeline in the foreground, `label` and `ack` write to a
  store, `ctl` talks to a running daemon (`status`, `events`, `label`, `ack`, `pause`,
  `resume`, `shutdown`).

It reads `/proc` as the current user, so it needs no privileges, auditd or eBPF. The model runs
offline from a cached checkpoint; nothing leaves the machine.

## Design principle

The model is the judge. The code computes context (facts about a launch, rendered as text) and
never a verdict. There are no rules, scores, thresholds or process name and path lists anywhere in
the pipeline. Names only appear as text inside the context the model reads. The class set
(`safe`, `warn`, `crit`) and what each means are prompt text in the configuration, not code.

Two consequences:

- With no model available there is no fallback heuristic. Contexts are stored with
  `judged=false` and the status says why.
- Every verdict is stored together with the exact context text, so events can be labelled by hand
  and used as training data.

## Usage

Run `direnv allow` (or `nix develop`) once, then:

```sh
just exec                 # poll /proc and judge; print crit verdicts as JSON lines
just exec --once          # one scan, print a summary
just exec --db /tmp/e.db  # store verdicts at a given path
just daemon               # jevmond in the foreground, serving the socket
just tui                  # the terminal UI
just run jevmon ctl status
```

Verdicts are stored in `$XDG_STATE_HOME/jevmon/events.db`. Without a model, contexts are stored
with `judged=false`.

### TUI

`jevmon-tui` (`app/tui/`, Rust, Ratatui) is a client of the daemon socket (`--socket PATH`
overrides the default `$XDG_RUNTIME_DIR/jevmon/jevmond.sock`). It decides nothing: labels and
acks are sent to the daemon.

The header shows the connection state, the model status and the judge counters. The list has the
columns `verdict` (class and confidence), `time`, `command` and `label` (the human label or
`ack`). It follows new events until you move the selection; going back to the bottom resumes
following. The detail pane shows the selected event: id, time, class, confidence, command, exe,
the verdict's probabilities as bars, the context the model read as key/value facts, and
whether the verdict was judged, cached or only available. Colours follow the configured
severity, never a class name.

| key | action |
| --- | --- |
| `up` `down` / `j` `k` | move the selection (scroll the detail pane when it has focus) |
| `pgup` `pgdn`, `home` / `g`, `end` / `G` | page, first, last |
| `tab` | switch focus between the list and the detail pane |
| mouse wheel, click | scroll the focused pane; select a row (the newest row resumes follow) |
| `t` / `f` | label the selected event true-positive / false-positive |
| `y` | acknowledge the selected event |
| `p` | pause or resume judging (the poller keeps running) |
| `/` | text filter on the command; `enter` applies, `esc` cancels |
| `u` | show unlabelled events only |
| `s` | cycle the minimum class: safe, warn, crit |
| `d` | detail pane layout: below, beside, hidden |
| `r` | reconnect now |
| `?` | help overlay |
| `q` | quit |

### Command line

| command | purpose |
| --- | --- |
| `jevmon exec [--once] [--db PATH] [--config PATH] [--verbose]` | run the pipeline in the foreground; `--once` prints one scan summary |
| `jevmon label <event_id> <label> [--db PATH]` | write a label to a store (for example `true-positive`) |
| `jevmon ack <event_id> [--db PATH]` | acknowledge an event in a store |
| `jevmon ctl status` | model, device, poller and judge counters of a running daemon, as JSON |
| `jevmon ctl events [--since-id N] [--min-severity N] [--limit N] [--unlabeled] [--text S]` | query stored events as JSON |
| `jevmon ctl label <id> <label>`, `ack <id>` | label or acknowledge through the daemon |
| `jevmon ctl pause`, `resume`, `shutdown` | control the daemon |

`jevmon ctl` takes `--socket PATH` before the subcommand.

## Architecture

```
/proc  ->  ExecSource  ->  ContextBuilder  ->  Judge (queue, memo, batches)  ->  SQLite store
 tier 0     detect new      facts -> text       Laya model, optional head          |
            pids, count     (redacted,                                             v
            misses          sanitized)                          jevmond socket (protocol v4)
                                                                      |           |
                                                                  jevmon ctl    jevmon-tui
```

### Source: tier 0 and tier 0+ (`app/src/jevmon/sources/`)

Tier 0 polls `/proc` and diffs pids. A pid is a new launch when it is unseen, when its
`(pid, starttime)` changed (pid reuse), or when its argv or exe changed while the start time held
(an exec in place). Per pid it reads `stat`, `cmdline`, `status` (uid), `cgroup` (mapped to a
systemd unit or scope) and, where readable, the `exe` and `cwd` links. Kernel threads and zombies
are tracked and not emitted. A process that vanishes mid-read is skipped.

Tier 0+ also reads the cumulative `processes` counter in `/proc/stat`. Forks minus new pids
observed (floored at 0) estimates how many launches were never seen. It is an upper bound: the
counter counts forks, not execs.

The poller has a hard CPU budget (`cpu_budget`, a fraction of one core measured on its own
thread). When a scan costs more than the budget allows, the interval widens; it shrinks back when
it does not. Work per scan is bounded by `max_pids` and `max_tracked`.

Everything is redacted before it is stored or shown (secret flags, `KEY=value`, URL userinfo,
bearer tokens, long hex and base64 blobs) and sanitized (ANSI, control and format characters
removed).

### Context (`detect/context.py`)

A context is a compact text built from facts: exe path and basename, masked argv template and
truncated redacted argv, uid, cgroup, cwd, ancestors, structural facts (is the exe's directory
writable by a non-root user, who owns the file, how recently it was written, the mount type), and
learned counts and surprisal from the box's own history. `[context] order` selects which
segments the model reads. The shipped order is `header, exe, template, argv`: the command only.
The other facts are still computed, stored with the event and used in the memoisation signature.
Adding the parent command or sibling rate to the model's text did not help in the lab (see the
[lab README](../lab/README.md)), so `[exec] parent_argv` is off.

### Launch-time facts (`sources/facts.py`)

`[exec] facts = [...]` (default empty) makes the poller read short per-process facts in one pass
of `/proc/<pid>` reads per new exec: `exe_state` (normal, deleted, memfd), `argv0_vs_exe`,
`stdio` (types of fd 0, 1, 2), `session` (controlling tty, session leader, loginuid), `workdir`,
`cgroup_path` (tail of the cgroup path, long ids shortened), `privileges` (uid, euid, CapEff,
NoNewPrivs, Seccomp), `namespaces` (each of mnt, pid, net, user, uts, ipc, cgroup, time same or
differs against init, else the daemon; uid_map and NSpid hints) and `exe_file` (owner, mode,
setuid, world-writable). They are facts for the model to read, not rules: nothing looks a value
up in a list. A vanished or unreadable process gives `unknown`. A fact reaches the model only if
its name is also in `[context] order` (a name in `order` that is not in `facts` is a config
error). Cost measured on this box: about 37 microseconds of poller CPU per new process. The lab
found no gain from them yet (see the [lab README](../lab/README.md)), so the default is none.
`[context] env_priority` is a display order for a future environment segment, not a rule.

### Judge (`detect/judge.py`, `laya/`)

- The judge interface is the typed-question API shape Laya and the hosted Jev models share (a
  state string plus named `choice` questions with per-category descriptions in, probabilities out).
- The model is [Laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large, 421M
  parameters), answering a typed choice question over the three class descriptions from
  `[judge.categories]`. Its answer is the verdict and its severity (`safe` 0, `warn` 1, `crit`
  2). The confidence is stored and shown, and nothing is thresholded on it.
- `[judge] tuned` replaces the zero-shot choice with a small classification head over the frozen
  encoder (a `.safetensors` written by `lab/scripts/finetune_head.py` or `train_head_v3.py`). Its classes
  must be the three above.
- Volume is handled by memoisation, not filtering: each distinct context signature is judged once
  and reused for `memo_ttl_seconds`; the signature changes as statistics move, so the context is
  judged again.
- One worker thread behind a bounded queue. The worker takes the first waiting context and drains
  up to the batch limit without waiting, so an idle judge answers one at a time and a backlog is
  judged in batches. A full queue is counted and the context is stored `judged=false`.
- Device selection: `auto` uses the iGPU when a working HIP device is present and CPU otherwise.
  A GPU failure at load or inference time falls back to CPU, and `ctl status` says so.

### Store (`core.py`)

SQLite at `$XDG_STATE_HOME/jevmon/events.db` (default `~/.local/state/jevmon/events.db`): events,
learned counts, surprisal samples and labels, with day-based retention (`retention_days`, 14 by
default). The schema version is in `PRAGMA user_version`; a store from another version is deleted
and recreated empty rather than migrated.

### Daemon protocol (`daemon/protocol.py`)

Newline-delimited JSON over a Unix socket (`$XDG_RUNTIME_DIR/jevmon/jevmond.sock`), protocol
version 4. One request per line; unknown request types and unknown fields are rejected, and
requests and replies are bounded in size.

| request | purpose |
| --- | --- |
| `status` | model, device, poller and judge counters |
| `events` | query stored events (`since_id`, `min_severity`, `limit`, `unlabeled`, `text`) |
| `event` | one event by id |
| `label`, `ack` | attach a human label, acknowledge |
| `pause`, `resume` | stop and restart judging; the poller keeps running |
| `reload` | re-read the configuration |
| `shutdown` | stop the daemon |
| `subscribe` | turn the connection into a stream of new events and periodic stats |

From version 3 an event's `severity` is the class (0 safe, 1 warn, 2 crit) and `class` is its name.
There is no compatibility with version 1 peers.

## Install and run

Everything runs from the repository root through [`just`](https://github.com/casey/just). The dev
shell pins Python, uv, just and ruff.

```sh
nix develop            # or `direnv allow`
just sync              # app/.venv, CPU wheels
just exec --once       # one scan, print a summary
just exec              # poll and judge; print crit verdicts as JSON lines
just daemon            # jevmond in the foreground
just tui               # build and open the TUI
just run jevmon ctl status
```

### On the GPU

The default environment is CPU-only. The GPU environment uses the PyTorch ROCm wheel (about 6 GB,
shared with the lab's GPU environment through uv's cache).

```sh
just gpu-sync          # app/.venv-gpu
just gpu-run jevmond   # run the daemon on the GPU
just run jevmon ctl status   # model_status: "ready (gpu, bf16)"
```

On NixOS the flake's `LD_LIBRARY_PATH` supplies the wheel's runtime libraries. The wheel used
here (`torch 2.14.1+rocm7.2`) lists `gfx1151` in its architecture list, so no
`HSA_OVERRIDE_GFX_VERSION` is needed. Other wheels and architectures were not tried.

### As a systemd user service

```sh
just daemon-install    # sync the GPU env, install the unit, enable it, restart
just daemon-status
just daemon-logs
just daemon-stop       # also: daemon-start, daemon-restart, daemon-uninstall
```

The unit is [`contrib/jevmond.service`](contrib/jevmond.service). It runs the daemon from the
repository checkout with `VIRTUAL_ENV` set to `app/.venv-gpu`, `Nice=10`, `CPUQuota=200%` and
`MemoryMax=8G`, and restarts on failure. `WorkingDirectory` is the repository root so relative
config paths such as `models/lab-head-v4.safetensors` resolve. The unit file assumes the checkout is at
`~/dev/jevmon`; edit it if yours is elsewhere. For a CPU-only service the file has the
alternative `ExecStart` in a comment.

## Configuration

Defaults are in [`src/jevmon/default.toml`](src/jevmon/default.toml), which documents every key.
Copy it to `$XDG_CONFIG_HOME/jevmon/config.toml` (usually `~/.config/jevmon/config.toml`) and keep
only the keys you change. Unknown keys are an error. The daemon re-reads the file on a
`reload` request or restart.

| section | what it controls |
| --- | --- |
| `[exec]` | source (`proc` or `replay`), poll `interval`, `min_interval` / `max_interval`, `cpu_budget` (default 0.03 of one core), `max_pids`, `max_tracked`, `summary_seconds`, `parent_argv`, `replay_file` |
| `[context]` | what the model reads: `order` of segments, `sep`, `model_max` characters, `argv_max`, `template_tokens`, rarity settings, `sibling_seconds` |
| `[judge]` | `mode` (`laya` or `fake`), `question`, `tuned` head file, `checkpoint` / `checkpoint_dir`, `device`, `dtype`, `batch_size`, `threads`, `threads_gpu`, `memo_ttl_seconds`, `queue_capacity` |
| `[judge.categories]` | the three classes and what each means, as prompt text |
| `[store]` | `retention_days`, queue capacity, whether plain exec events are persisted |

The `fake` judge mode and the `replay` source are deterministic stand-ins used by the tests and
by the screenshot recording; they involve no model and no real processes.

The class descriptions are the zero-shot prompt. Rewrite them for your own machine, or train a
head on your own labels and point `tuned` at the `.safetensors` file (see the lab README for how the head was trained
and how far to trust it).

## Judge modes

There is one axis and three classes, and the model's confidence is shown next to the verdict with
no threshold on top. jevmon judges one launch at a time, so a class describes that launch, not a
rate or a trend.

| class | severity | meaning |
| --- | --- | --- |
| `safe` | 0 | expected everyday activity here; no action needed |
| `warn` | 1 | out of the ordinary for this workstation but plausibly harmless; worth a glance |
| `crit` | 2 | looks hostile from the launch alone |

`safe` means the model saw nothing to act on. It does not mean the launch was checked.

Zero-shot. With `tuned = ""` the model answers a typed choice question over the class
descriptions in `[judge.categories]`. The descriptions are the prompt; rewrite them for your own
machine. On the lab's held-out rows this is close to always answering safe.

```toml
[judge]
tuned = ""
question = "Is this process launch expected on this machine?"

[judge.categories]
safe = "normal operation for this machine"
warn = "unexpected but plausibly harmless"
crit = "possible attack, intrusion or abuse"
```

Shipped head. A small classifier over the frozen encoder, trained on labelled lab contexts, is
packaged with the app and loaded by default (`tuned = "lab-head-v4"`, 12.5 KB). To use your own,
give a path to a `.safetensors` file; its classes must be the three above.

```toml
[judge]
tuned = "models/my-head.safetensors"
```

Device and source settings:

```toml
[judge]
device = "gpu"       # "auto" | "cpu" | "gpu"
dtype = "bf16"       # "auto" | "fp32" | "fp16" | "bf16"
batch_size = 0       # 0 = auto: 32 on GPU, 1 on CPU
threads = 2          # CPU threads while the model runs on the CPU
```

```toml
[exec]
proc_root = "/proc"   # another /proc tree, for containers or tests
interval = 0.1        # seconds between scans; widened to stay under cpu_budget
```

`device = "auto"` uses the iGPU when a working HIP device is available and falls back to CPU
otherwise; `jevmon ctl status` shows which one is in use.

## Measured performance

All numbers are from one workstation (32 cores, AMD Radeon 8060S iGPU, 740 to 960 pids running)
with short interleaved runs; treat the digits as plus or minus 10 percent. Measured on
2026-10-12 (poller, judge) and 2026-10-14 (accuracy); code at commit `4482c98`.

### Summary

| metric | value | source |
| --- | --- | --- |
| poller CPU budget | 3% of one core (`cpu_budget = 0.03`) | `default.toml` |
| poller CPU, observed | about 2.0% of one core | 62 s run, 46 scans, exec-tier0 |
| scan time p50 / p95 | 24 ms / 42 ms at 740 to 960 pids | same |
| poll interval | about 1.29 s effective (configured 0.1 s, widened by the budget) | same |
| missed launches, real box | 0.974 (estimated from `/proc/stat`, upper bound) | same, with about 230 short processes/s |
| missed launches, lab capture | 0.956 to 0.961 | E14, E15 lab captures |
| per-launch fact cost | about 37 microseconds of poller CPU per new process | `[exec] facts`, E17 |
| judge latency, iGPU bf16 | 65 ms (batch 1), 55 ms (8), 44 ms (32) per context | `bench_judge.py` |
| judge latency, CPU, 2 threads | 2506 ms per context, batch 1 (5021 ms CPU time) | `bench_judge.py` |
| judge latency, CPU, 8 threads | 787 ms per context, batch 1 | `bench_judge.py` |
| throughput, iGPU | 15 to 22 contexts/s; about 0.75/s need the model after memoisation | live daemon, 20 s |
| model memory | about 3.9 GiB GTT with one Laya resident; 0.42 GiB peak VRAM | sysfs counters |
| size on disk | Laya checkpoint 2.3 GiB (HF cache); shipped head 12.5 KB | filesystem |
| daemon idle and active CPU | not measured | |
| store size per day | not measured | |
| TUI memory and CPU | not measured | |

### Accuracy

Macro-F1 on the held-out lab rows (E15 pool: 420 rows, 61 families; one synthetic capture).
Family-cluster bootstrap 95% CI; paired differences between rows have CIs of about plus or minus
0.12, so only the gap to zero-shot is clear.

| judge | macro-F1 [95% CI] | source |
| --- | --- | --- |
| always answer safe | not measured on the pool; 0.185 on the earlier 130 rows | E12 |
| Laya, zero-shot | 0.391 | E15 stage 1 |
| Laya with the shipped head | 0.667 [0.53, 0.77] | E15, E18 |
| Foundation-Sec-8B with a head (not shipped) | 0.709 [0.58, 0.81] | E18 |
| hosted Jev 1.13, zero-shot (flat text) | 0.616 [0.50, 0.71] | E18 |

Foundation-Sec-8B costs 179 ms p50 per context at batch 1 and 16.1 GiB of GPU memory, against
44 ms and about 4 GiB for Laya, and its gain is not distinguishable from zero. Hosted Jev was
queried over the network (median 0.42 s per request) and about 4% of its verdicts change when the
same request is repeated.

### Poller

62 seconds at the shipped defaults, with a loop launching about 230 short-lived processes per
second in the background:

| metric | value |
| --- | --- |
| scans | 46 |
| scan time p50 / p95 | 24 ms / 42 ms |
| effective interval | about 1.29 s (the 0.05 s target is unreachable inside the budget) |
| poller CPU | about 2.0% of one core (budget 3%) |
| launches seen of the probe commands | 0 of about 4,700 each |
| missed fraction estimated from `/proc/stat` | 0.974 |

At a 1.3 second interval essentially every sub-millisecond process is missed. Only processes that
live longer than the scan interval (builds, agents, servers) are seen. The
tier 0+ miss estimate does not say what ran; it bounds how much was not seen.

### Judge, GPU against CPU

`lab/scripts/bench_judge.py`, the shipped `LayaDecider.decide_batch`, contexts of 537 to 1000
characters built by the real context builder. "CPU ms" is process CPU time per context.

| cell | wall ms/context | CPU ms/context |
| --- | --- | --- |
| CPU, 8 threads, batch 1 | 787 | 6252 (8 cores pegged) |
| CPU, 2 threads, batch 1 | 2506 | 5021 |
| GPU bf16, batch 1 / 8 / 32 | 65 / 55 / 44 | 92 / 96 / 80 |
| GPU fp16, batch 1 / 8 / 32 | 67 / 52 / 44 | 98 / 90 / 78 |
| GPU fp32, batch 1 / 8 / 32 | 246 / 362 / 259 | 250 / 351 / 262 |

- bf16 on the GPU is about 12 times faster than 8 CPU threads at batch 1 and about 18 times at a
  drain of 32. Against 2 CPU threads, the daemon's CPU default, it is about 38 times faster.
- Batching helps on the GPU and not on CPU, where a padded batch costs at least the single-context
  floor. That is why `batch_size = 0` means 32 on the GPU and 1 on CPU.
- Verdicts across dtypes on 64 contexts: no label differs between CPU fp32 and GPU fp32, fp16 or
  bf16. Largest probability difference 0.0038 (fp16) and 0.032 (bf16).
- Load takes 8 to 9 seconds the first time and 0.2 to 5 seconds with a warm cache. The model
  holds about 4 GiB of GTT (system RAM on this APU) while loaded.
- A live daemon on the GPU judged 158 contexts in the tail of a 20 second run with an empty
  queue. About 0.75 contexts per second need the model after memoisation, against 15 to 22 it can
  judge.

### Zero-shot and tuned verdicts

In the first lab run the zero-shot model gave the same answer
(`routine`, in the four-class setup of the time) for every one of 563 recorded contexts. A head
trained on lab labels does clearly better, but the evidence is small and synthetic. The
[lab README](../lab/README.md) has the tables and their caveats.

## Limitations

- Coverage. Tier 0 only sees processes alive at a scan instant. Short-lived commands are
  missed, as measured above. Exact launch events need a stronger source (a process connector or
  audit with a capability, or eBPF), which this project does not use.
- Single launch. The judge sees one launch at a time. Rates and resource growth (fork storms,
  retry loops, memory growth) cannot be read from one launch and are out of scope for this
  judge.
- Zero-shot is weak. Without a tuned head (`tuned = ""`) the verdicts are close to the
  always-`safe` baseline. The shipped head, on by default, was trained on a synthetic lab; how it behaves on a real workstation
  has not been measured.
- Attacker-influenced text. Command lines are attacker-controlled and the model reads them.
  Redaction, masking, truncation and sanitizing reduce the risk and do not remove prompt
  injection. Verdicts are advisory.
- Learning can be poisoned. Counts and the rarity distribution are learned on the fly. Normal
  looking activity during a learning period shifts the baseline. There is no frozen learning
  window.
- `safe` is not verified. It means the model saw nothing to act on.
- Best-effort facts. The `exe` link is unreadable for other users (falls back to argv0), and
  file stats can race with deletion.
- Not measured. Multi-hour stability, thermals, power draw and the effect of judging at
  80 to 100 percent iGPU load on desktop rendering.
