"""A deterministic stand-in judge: no model, no GPU, no network, no judgement.

`FakeDecider` implements the `detect.judge.Decider` protocol the same way `LayaDecider` does, so
the whole pipeline - context building, memoisation, batching, storage and the TUI -
runs unchanged. The label is a stable SHA-256 of the context text over the configured categories
and the confidence comes from the same digest: reproducible, but it means nothing about the world.
The verdict summary says `mock:` so a reader of the detail pane can never mistake it for a model.

Selected with `[judge] mode = "fake"` (the default stays `"laya"`). Used by demos, tests and
`just screenshots`, where a real checkpoint and a GPU would make the output slow and unstable.
"""

import hashlib

from jevmon.detect.judge import SAFE, JudgeSettings
from jevmon.laya.decider import Decision

STATUS = "fake deterministic stand-in"


def fake_verdict(
    state: str, categories: tuple[str, ...], forced: str | None = None
) -> tuple[str, float, dict[str, float]]:
    """A (label, confidence, probabilities) triple derived only from `state`.

    `forced` names the category to answer (a demo fixture); confidence and the spread over the
    others still come from the digest.
    """
    if not categories:
        raise ValueError("the judge needs at least one category")
    if len(categories) == 1:
        return categories[0], 1.0, {categories[0]: 1.0}
    digest = hashlib.sha256(state.encode("utf-8", "replace")).digest()
    label = forced if forced in categories else categories[digest[0] % len(categories)]
    confidence = round(0.60 + (digest[1] % 40) / 100, 2)
    others = [name for name in categories if name != label]
    weights = [1.0 + digest[2 + i % (len(digest) - 2)] / 256.0 for i in range(len(others))]
    scale = (1.0 - confidence) / sum(weights)
    probabilities = {label: confidence}
    for name, weight in zip(others, weights, strict=True):
        probabilities[name] = round(weight * scale, 4)
    return label, confidence, probabilities


class FakeDecider:
    """The `Decider` protocol over a stable hash of the state."""

    def __init__(self, settings: JudgeSettings | None = None) -> None:
        self.settings = settings or JudgeSettings()
        self._categories = tuple(self.settings.categories)

    def load(self) -> None:
        """There is nothing to load; the monitor still calls this on its load thread."""

    @property
    def ready(self) -> bool:
        return True

    @property
    def status(self) -> str:
        return STATUS

    @property
    def active_device(self) -> str:
        return "fake"

    @property
    def active_dtype(self) -> str:
        return "fake"

    def decide(self, state: str) -> Decision:
        forced = next(
            (c for fragment, c in self.settings.fake_verdicts.items() if fragment in state), None
        )
        label, confidence, probabilities = fake_verdict(state, self._categories, forced)
        return Decision(
            label=label,
            confidence=confidence,
            attack=0.0,
            flagged=label != SAFE,
            summary=f"mock: {label} {confidence:.2f}",
            available=True,
            probabilities=probabilities,
        )

    def decide_batch(self, states: list[str]) -> list[Decision]:
        return [self.decide(state) for state in states]


__all__ = ["STATUS", "FakeDecider", "fake_verdict"]
