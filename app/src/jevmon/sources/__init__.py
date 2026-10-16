"""Process exec sources. Tier 0 polls `/proc`; tier 0+ adds the fork-counter miss estimate.

`replay` re-emits a recorded stream instead (see `replay.py`), and `demo` is the fictional
recording `just screenshots` replays.
"""

from jevmon.sources.demo import demo_events
from jevmon.sources.exec import ExecSettings, ExecSource, ScanResult
from jevmon.sources.proc import ProcInfo, ProcReader, ProcStat, RealProcReader
from jevmon.sources.replay import ReplayError, ReplaySource, load_events, write_events

__all__ = [
    "ExecSettings",
    "ExecSource",
    "ProcInfo",
    "ProcReader",
    "ProcStat",
    "RealProcReader",
    "ReplayError",
    "ReplaySource",
    "ScanResult",
    "demo_events",
    "load_events",
    "write_events",
]
