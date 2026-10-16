from jevmon.detect.fake import FakeDecider
from jevmon.detect.judge import JudgeSettings


def test_fake_verdicts_force_matching_fragment_and_fall_back_to_hash():
    settings = JudgeSettings(mode="fake", fake_verdicts={"poll-queue": "crit"})
    decider = FakeDecider(settings)
    assert decider.decide("exe: /bin/poll-queue --once").label == "crit"
    other = decider.decide("exe: /bin/make -j8")
    assert other.label in settings.categories
