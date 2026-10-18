# jevmon app

The `app/` project holds everything that runs on the machine being monitored: the exec sources
(an eBPF helper stream, or a `/proc` poller), the context builder, the judge, the event store, the daemon (`jevmond`), the command line
(`jevmon`) and the terminal UI (`jevmon-tui`). The lab, scenarios and experiments are in
[`../lab/`](../lab/README.md). For a short overview and screenshots see the
[root README](../README.md).

Contents: [usage](#usage) - [architecture](#architecture) - [install and run](#install-and-run) -
[configuration](#configuration) - [judge modes](#judge-modes) -
[measured performance](#measured-performance) - [limitations](#limitations)

## What it does

- Daemon (`jevmond`): reads new process launches (every exec from the eBPF helper, or a share of
  them by polling `/proc`), builds a short text context for
  each one, asks a small model whether the launch is `safe`, `warn` or `crit`, and stores the
  verdict with the context it was made from. It serves a Unix socket.
- TUI (`app/tui/`, Rust, Ratatui): a client of that socket. List and detail views, filtering,
  labelling, acknowledging, pausing the judge.
- `jevmon`: `start`, `stop` and `status` bring the daemon up and down, `exec` runs the same
  pipeline in the foreground, `label` and `ack` write to a store, `ctl` talks to a running daemon
  (`status`, `events`, `label`, `ack`, `pause`, `resume`, `shutdown`).

Two exec sources, chosen with `[exec] source`:

| source | sees | needs |
| --- | --- | --- |
| `ebpf` (default) | every successful exec, with up to 4 KB of argv | a small root helper (`jevmon-exec`) that jevmon starts for you; the daemon, CLI and TUI stay unprivileged |
| `proc` | processes alive at a scan instant, so a share of short launches | nothing: it reads `/proc` as the current user |

There is no automatic switch between them. With `ebpf` and no helper running, the daemon says so
in `jevmon status`, `ctl status` and the TUI header and keeps retrying. The model runs offline
from a cached checkpoint; nothing leaves the machine.

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
jevmon start              # the helper (ebpf source only, one password prompt) and the daemon
jevmon status             # configured source, helper, daemon (no privileges)
jevmon stop
just exec                 # judge in the foreground; print crit verdicts as JSON lines
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
| `a` | allow launches like the selected one (detail pane open): choose a scope, `enter` adds the rule, `esc` cancels |
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
| `jevmon ctl allow list`, `add <rule>`, `remove <rule>` | show or edit the `[filter] allow` rules; changes are written to the config file and applied live |
| `jevmon ctl pause`, `resume`, `shutdown` | control the daemon |

`jevmon ctl` takes `--socket PATH` before the subcommand.

## Architecture

```
jevmon-exec (root, eBPF) --socket--> EbpfSource  --+
/proc ------------------------------> ExecSource  --+-> allow rules -> ContextBuilder -> Judge -> SQLite store
                                        (poller)         (optional)     facts -> text    queue, memo,      |
                                                                        (redacted,       batches            v
                                                                        sanitized)            jevmond socket (protocol v4)
                                                                                                    |           |
                                                                                                jevmon ctl    jevmon-tui
```

### Source: ebpf (`sources/ebpf.py`, `app/helper/`)

`jevmon-exec` is a small C program: a CO-RE BPF program on `sched_process_exec` plus a loader.
It runs as root with only `CAP_BPF`, `CAP_PERFMON` and `CAP_SYS_RESOURCE`, no network and a
read-only filesystem, and does nothing but stream events to a unix socket in `/run/jevmon/`.
It accepts no commands. A client is served only if `SO_PEERCRED` shows the uid the helper was
started for (`--uid`); any other connection is closed without a byte.

`sched_process_exec` fires after an exec succeeded, so failed `execve` calls (a shell probing
`PATH`, `EACCES`) never appear. Per exec the helper reports: pid, ppid, uid/euid/gid, process
and parent start time, cgroup id, `comm` and the parent's `comm`, the resolved path of the file
that runs as `exe` (walked from the kernel's `bprm->file`: symlinks and the nix profile are
resolved, a script reports its interpreter, like `/proc/<pid>/exe`), the raw `execve` argument as
`filename`, up to 4096 bytes of argv split into at most 64 arguments, a truncation flag, a
monotonic and a wall timestamp.

The wire format is newline-delimited JSON: a `hello` line (protocol version, limits), one `exec`
line per exec, and a `hb` line every second with exact totals: execs seen by the kernel program,
records lost because the ring buffer was full, events not delivered to the client. Nothing is
dropped uncounted; `jevmon status` and `ctl status` show `exec_lost`, `exec_client_dropped`,
the per-second rate and the busiest second.

The source parses each line into the same event the poller produces. The parent's command comes
from a bounded cache of recent execs keyed by pid and checked against the parent's start time,
so a reused pid is never mistaken for the parent; a parent that exec'd before the helper started
is known by its `comm` only. Nothing is read from `/proc` per exec (so no races); `cwd` is not
collected, and `cgroup` is the path resolved from the cgroup id by a rate-limited walk of
`/sys/fs/cgroup` (empty when the cgroup is already gone). The reader has a CPU ceiling
(`helper_cpu_budget`): above it, it stops reading and the helper counts the events it could not
deliver.

If the helper is missing the source reports `exec source unavailable (helper not running)`,
reconnects with exponential backoff (`helper_retry_min` to `helper_retry_max`) and never turns
into polling.

Volume is handled after the source: the allow rules, the memo (a situation is judged once) and the
bounded judge queue. Verdicts for a situation already stored in the last `coalesce_seconds`
(default 60) are counted (`coalesced`) instead of stored again, and the learned counts are kept
in memory and written once a second. See [measured performance](#measured-performance).

### Source: proc, tier 0 and tier 0+ (`app/src/jevmon/sources/exec.py`)

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
- Five judge models are available, selected in the configuration. Each is a classifier over a
  local or downloaded encoder, trained on a mixed pool of labelled launches and shipped with the
  app: [Laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large, 421M parameters,
  the default), `laya-lora` (the same Laya encoder with a small LoRA adapter merged into the
  weights at load, no extra dependency), `jhu-clsp/ettin-encoder-1b`, `ettin-lora` (Ettin with a
  LoRA adapter merged in) and `Qwen/Qwen3-Embedding-8B`. Laya and `laya-lora` are the smallest and
  the ones meant for ordinary machines; the others are larger backbones. Without a head, Laya answers a zero-shot typed
  choice question over the three class descriptions from `[judge.categories]`.
- The model's answer is the verdict and its severity (`safe` 0, `warn` 1, `crit` 2). The
  confidence is stored and shown, and nothing is thresholded on it.
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
just daemon-install    # the unprivileged user service (see below)
just exec-build        # the exec helper, once (nix build into app/helper/result)
jevmon start           # starts the helper if [exec] source = "ebpf", then the daemon
jevmon status
just tui               # build and open the TUI
jevmon stop
```

With `source = "ebpf"` (the default) `jevmon start` asks for a password once, only to start the
helper as a transient system unit (`systemd-run --unit=jevmon-exec`) through `run0`, else
`sudo`, else `pkexec`. The daemon and everything else stay unprivileged. The helper path is
resolved first and must be in the nix store or owned by root in root-owned directories; a path
you can write is refused. A transient helper lasts until `jevmon stop` or a reboot. To have the
helper start at boot instead, `just exec-install` installs a system service
([`contrib/jevmon-exec.service`](contrib/jevmon-exec.service)) for your uid; `jevmon stop` then
leaves it running. `just exec-uninstall` removes it. With `source = "proc"` nothing is ever
elevated.

Checks without touching the host: `just exec-trace 10` runs only the helper for 10 seconds
(one `sudo`) and prints a summary. `just exec-vmtest` loads the BPF program in a throwaway
NixOS VM under qemu (needs `/dev/kvm`, no root) and lists what it recorded.

Advanced: `jevmon helper start|stop` runs just the elevated step.

The foreground commands still work without the service:

```sh
just exec --once       # one /proc scan, print a summary
just exec              # judge in the foreground; print crit verdicts as JSON lines
just daemon            # jevmond in the foreground
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
config paths such as `models/my-head.safetensors` resolve. The unit file assumes the checkout is at
`~/dev/jevmon`; edit it if yours is elsewhere. For a CPU-only service the file has the
alternative `ExecStart` in a comment.

## Configuration

Defaults are in [`src/jevmon/default.toml`](src/jevmon/default.toml), which documents every key.
Copy it to `$XDG_CONFIG_HOME/jevmon/config.toml` (usually `~/.config/jevmon/config.toml`) and keep
only the keys you change. Unknown keys are an error. The daemon re-reads the file on a
`reload` request or restart.

| section | what it controls |
| --- | --- |
| `[exec]` | `source` (`ebpf`, `proc` or `replay`); ebpf: `helper_socket` (default `/run/jevmon/exec.sock`), `helper_path`, `helper_retry_min` / `helper_retry_max` (backoff seconds), `recent_exec_max` (parent cache size), `helper_cpu_budget` (reader ceiling, default 0.5 of one core); proc: poll `interval`, `min_interval` / `max_interval`, `cpu_budget` (default 0.03 of one core), `max_pids`, `max_tracked`; both: `summary_seconds`, `parent_argv`, `facts`, `replay_file`. The ebpf keys are accepted and ignored when another source is selected |
| `[context]` | what the model reads: `order` of segments, `sep`, `model_max` characters, `argv_max`, `template_tokens`, rarity settings, `sibling_seconds` |
| `[judge]` | `mode` (`laya` or `fake`), `model` (the shipped judge: `laya` / `laya-lora` / `ettin` / `ettin-lora` / `qwen3e`), `question`, `tuned` head override, `checkpoint` / `checkpoint_dir`, `device`, `dtype`, `batch_size`, `threads`, `threads_gpu`, `memo_ttl_seconds`, `queue_capacity` |
| `[judge.categories]` | the three classes and what each means, as prompt text |
| `[store]` | `retention_days`, `queue_capacity`, `store_exec_events`, `coalesce_seconds` (ebpf: repeated memoised verdicts are counted, not stored) |
| `[filter]` | `allow` rules that skip the judge for launches you have decided are routine, `store_skipped` |
| `[store]` | `retention_days`, queue capacity, whether plain exec events are persisted |

### `[filter]`: allow rules

Launches matching an allow rule are not judged; everything else is. The code contains no rules and
the shipped `allow` list is empty. On a busy agent workstation a few processes (read-only `git`
queries, compilers and linkers, build caches, `rocm-smi`-style pollers) run hundreds of times a
minute and make up most of the execs, so the owner can allow the ones they know are routine. A
full exec feed produces more distinct launches per second than the model can judge, which is what
this is for. Setting `allow` in your config replaces the default list as a whole.

```toml
[filter]
allow = [
  "git rev-parse *",
  "git config --get *",
  "/nix/store/*-clang-*/bin/clang -c *",
  "rocm-smi *",
]
store_skipped = false
```

A rule is `"<exe> <args>"`, in the style of Claude Code permission rules:

- The exe part is a glob. If it starts with `/` or `*` it is matched against the full exe path.
  Otherwise it is matched against the exe basename. A bare name such as `git` matches a binary
  of that name in any directory, so a renamed or copied binary (`/tmp/git`) matches too; an
  absolute path (`/nix/store/*-git-*/bin/git`) does not. Use absolute paths where the path is
  stable.
- The args part is a glob over `argv[1:]` joined with single spaces. `*` is the only wildcard and
  matches any run of characters, spaces included. Everything else is literal.
- A rule with only an exe part matches that exe with no arguments.
- A trailing ` *` needs a space boundary: `git diff *` matches `git diff` and `git diff HEAD`,
  not `git diffx`. `git diff:*` means the same as `git diff *`.
- To match a literal `*` or `\` in an argument, write `\*` or `\\`. A backslash before any other
  character is literal.
- `<exe> *` matches that exe with any arguments, including none.
- Rules are checked in order; the first match wins. An empty rule is a config error that lists
  the rule.

Rules can be added from the TUI. With the detail pane open, `a` offers three scopes for the
selected launch and shows the rule that would be written:

- exact: the exe path and these exact arguments;
- prefix: the exe path, the first `k` arguments and ` *`; `k` starts at 1 (the subcommand) and
  changes with `left` / `right` or `-` / `+`;
- exe only: the exe path and ` *`.

The rule always uses the full exe path. For `git -C /w/repo diff --name-only HEAD` the prefix
rule is `/nix/store/...-git-2.55.0/bin/git -C *`; for `clang -c src/a.c -o build/a.o` it is
`/nix/store/...-clang-21/bin/clang -c *`. `enter` sends it to the daemon, which appends it to
`[filter] allow` in your config file (`$XDG_CONFIG_HOME/jevmon/config.toml`, created if missing;
comments and ordering are kept), applies it to the running matcher and de-duplicates. If the file
has no `allow` key yet, the shipped list is copied in first, since a list in your file replaces
it. The selected event stays in the list; later matching launches are skipped. `jevmon ctl allow`
does the same from the command line. Errors (invalid rule, unwritable file) show in the TUI status
line.

Children of an allowed process (hooks, aliases, ssh commands) are separate execs and are still
judged. Matching uses the exe path and argv as the source recorded them, after secret redaction
and length bounds. Both sources report the resolved exe path (a Nix store path, not the
profile symlink); the ebpf source also records the raw `execve` argument as `filename`.

Skipped launches are counted: `allow_skipped` and `allow_hits` (hits per rule) in `jevmon ctl
status`. With `store_skipped = true` they are also stored as unjudged events with reason `allow`.

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

Shipped heads. Every selectable judge is a classifier over a local or downloaded encoder, fitted on
the mixed lab pool and packaged with the app. `[judge] model` picks it; `tuned` overrides the
packaged head for that model with your own `.safetensors` file.

| `model` | backbone | measured cost on this box's iGPU |
| --- | --- | --- |
| `laya` (default) | the local Laya encoder, frozen linear head | 42-44 ms per context |
| `laya-lora` | the same Laya encoder, LoRA adapter merged in at load | same as `laya` (60 ms p50 / 116 ms p95 per context at batch 1 through the judge path, against 60 / 116 for `laya`) |
| `ettin` | `jhu-clsp/ettin-encoder-1b` (encoder) | not measured |
| `ettin-lora` | the same Ettin encoder, LoRA adapter merged in at load | not measured; the merge adds no forward-pass cost over `ettin` |
| `qwen3e` | `Qwen/Qwen3-Embedding-8B` (decoder) | not measured; an 8B encoder, so it needs a GPU with spare memory |

`laya` is recommended for ordinary machines: it is small, runs on the iGPU or CPU, and needs no
download. `laya-lora` is the same Laya encoder with a small LoRA adapter (27.5 MiB) folded into the
weights at load: no `peft` at runtime, no extra download beyond the Laya checkpoint, and about the
same latency as `laya`. `ettin-lora` does the same for Ettin-1B (a 22.1 MiB adapter). The Hugging Face backbones load lazily, only when selected, and need the
optional extra (`uv sync --extra backbone`); the model is downloaded from Hugging Face on first use.
A selected backbone that cannot load is reported in `ctl status` and the judge reports itself
unavailable; it never falls back to Laya. A broken Laya head or LoRA adapter still degrades to the
zero-shot typed question over the class descriptions (a small, always-near-`safe` baseline); a broken
backbone head makes the judge unavailable.

```toml
[judge]
model = "laya"                        # "laya" | "laya-lora" | "ettin" | "ettin-lora" | "qwen3e"
tuned = "models/my-head.safetensors"  # empty = the packaged head for `model`
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

### ebpf source: load shape and daemon cost

The stream the bench replays is synthetic (`app/scripts/bench_ebpf.py`), shaped like a
30-second eBPF capture of a busy workstation: 170 execs/s on average with peaks above 400, about
480 distinct executables with a skewed frequency, 7 arguments and about 100 bytes of argv at the
median, and about 14 new masked templates per second (`masked_template([exe] + argv[1:])`, 416
in 30 s). A fake helper serves it over a unix socket at a fixed rate in the helper's wire format;
the daemon side (socket, parse, redaction, allow check, context build, memo, judge queue, store)
runs unchanged, with a stand-in model that sleeps 45 ms per context (a GPU wait, no CPU) and a
SQLite store. The CPU column is the daemon process; the model's own CPU (tokenizing, at most
about 22 judged contexts/s) is not included.

| run | rate | daemon CPU | observed | judged | memo hit rate | shed | max queue age |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cold store, 60 s | 500/s | 11% of one core | 29,999 | 1,334 | 84% | 0 | 49 s (cold-start backlog) |
| warm store, 60 s | 500/s | 12% | 29,999 | 1,332 | 95.5% | 0 | 20 s (at the start) |
| warm store, 30 s | 2,000/s | 31% | 59,991 | 673 | 98.2% | 0 | 27 s |

Nothing was shed and the helper's lost counter stayed 0. The judge is the bottleneck: it handles
about 22 new situations per second, so a cold store (every situation new, and the count buckets
in the memo signature still moving) queues a backlog of a few thousand contexts that drains in
about a minute; verdicts for those are late by up to the queue age. When the queue is full
(`queue_capacity`) contexts are stored `judged=false` and counted as `dropped`. The reader's
own ceiling is `helper_cpu_budget`.

Two changes made this possible, both used only by the ebpf source: the learned counts and the
rarity window are kept in memory and written once a second (before, a store query per exec
pegged a core at about 300 execs/s), and a memoised verdict for a situation already stored in
the last 60 s is counted, not stored again (`coalesced`).

### Polling numbers

The rest of this section is about the `proc` source and the judge. All numbers are from one workstation (32 cores, AMD Radeon 8060S iGPU, 740 to 960 pids running)
with short interleaved runs; treat the digits as plus or minus 10 percent.

### Summary

| metric | value | source |
| --- | --- | --- |
| poller CPU budget | 3% of one core (`cpu_budget = 0.03`) | `default.toml` |
| poller CPU, observed | about 2.0% of one core | 62 s run, 46 scans, exec-tier0 |
| scan time p50 / p95 | 24 ms / 42 ms at 740 to 960 pids | same |
| poll interval | about 1.29 s effective (configured 0.1 s, widened by the budget) | same |
| missed launches, real box | 0.974 (estimated from `/proc/stat`, upper bound) | same, with about 230 short processes/s |
| missed launches, lab capture | 0.956 to 0.961 | lab captures |
| per-launch fact cost | about 37 microseconds of poller CPU per new process | `[exec] facts` |
| judge latency, iGPU bf16 | 65 ms (batch 1), 55 ms (8), 44 ms (32) per context | `bench_judge.py` |
| judge latency, `laya-lora` vs `laya`, iGPU bf16 | 59.7 / 59.9 ms p50 and 98.5 / 115.9 p95 per context at batch 1; 104.6 / 103.0 p50 at batch 32 | `bench_lora.py` |
| judge latency, CPU, 2 threads | 2506 ms per context, batch 1 (5021 ms CPU time) | `bench_judge.py` |
| judge latency, CPU, 8 threads | 787 ms per context, batch 1 | `bench_judge.py` |
| throughput, iGPU | 15 to 22 contexts/s; about 0.75/s need the model after memoisation | live daemon, 20 s |
| model memory | about 3.9 GiB GTT with one Laya resident; 0.42 GiB peak VRAM | sysfs counters |
| size on disk | Laya checkpoint 2.3 GiB (HF cache); Laya head 12.5 KB | filesystem |
| daemon idle and active CPU | not measured | |
| store size per day | not measured | |
| TUI memory and CPU | not measured | |

### Accuracy

The five shipped judges on the lab held-out (417 rows, 61 script families), macro-F1 with a 95%
family-cluster bootstrap CI. The LoRA rows are the per-seed mean ± sd over three seeds (each
shipped file is a single seed). The set is small and synthetic, so the CIs are wide and overlap.
Every set, the per-set splits and the method are in the [root README](../README.md#accuracy) and
[lab/README.md](../lab/README.md).

| judge | lab held-out macro-F1 [95% CI] |
| --- | --- |
| Laya, default | 0.519 [0.40, 0.63] |
| Laya-LoRA | 0.582 ± 0.046 |
| Ettin-1B | 0.638 [0.50, 0.74] |
| Ettin-LoRA | 0.669 ± 0.020 |
| Qwen3-Embedding-8B | 0.688 [0.56, 0.79] |
| zero-shot Laya | 0.411\* |
| always answer safe | 0.185 |

\* On the original 130-row held-out subset, not the full 417-row pool.

Laya runs a context in 42-44 ms on the iGPU and holds about 4 GiB; `laya-lora` runs at the same
speed (the merge leaves no runtime `peft` and no extra forward pass) with a 27.5 MiB adapter file.
Qwen3-Embedding-8B is an 8B encoder of the class measured at 95-179 ms per context and 16.1 GiB of
GPU memory; Ettin-1B and `ettin-lora` were not measured on the iGPU.

### Poller (proc source)

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

With no head, zero-shot Laya is close to always answering `safe` on the lab rows. The shipped
heads, trained on a mixed pool of labelled launches, do clearly better, but the lab data is small
and synthetic and the external labels are machine-derived, so treat the gap as indicative rather
than established. The [lab README](../lab/README.md) has the tables and their caveats.

## Limitations

- Coverage. The `proc` source only sees processes alive at a scan instant; short-lived commands
  are missed, as measured above. The `ebpf` source sees every successful exec, but needs the root
  helper. It cannot see an `execve` that fails (by design), and a launch is lost only when the
  helper's ring buffer or a slow client overflows, which is counted.
- ebpf source details. argv is cut at 4096 bytes and 64 arguments (flagged `argv_truncated`);
  `cwd` is not recorded; a parent that exec'd before the helper started has no exe or command in
  the context; the helper's view is the host's (a process in a container reports paths as the
  kernel sees them for that mount namespace's root, which may differ from the host path); a
  process whose argv memory is not resident at exec time is recorded with empty argv and a flag.
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
