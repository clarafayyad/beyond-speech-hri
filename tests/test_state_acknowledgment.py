"""Tests for state acknowledgment during the thinking phase.

The robot should react to spymaster cues (hesitation words, long thinking
time) that were detected in the audio features, rather than always saying
a generic thinking filler.
"""

import pytest

# ---------------------------------------------------------------------------
# Pure-function mirror of Guesser.get_state_acknowledgment_during_thinking()
# so tests don't need to instantiate the full Guesser with its heavy deps.
# ---------------------------------------------------------------------------

HESITATION_REACTIONS = [
    "You seemed a bit unsure there. Let me think carefully.",
    "I noticed some hesitation. I'll take my time.",
    "Sounds like this was tricky for you too. Let me think.",
]

LONG_THINKING_REACTIONS = [
    "That took some thought from you. Let me work through this.",
    "This seems complex. I'll think carefully.",
    "I'll take my time with this one.",
]

GENERIC_THINKING = [
    "Hmm.",
    "Hmm…",
    "Hmm, let's see.",
    "Let's see…",
    "Okay…",
    "Alright…",
    "One moment…",
    "Hmm, let me think.",
    "Just thinking…",
    "Thinking…",
]


def _get_acknowledgment(features):
    """Mirrors the selection logic in Guesser.get_state_acknowledgment_during_thinking()."""
    if not features:
        return "generic"

    hesitation_count = features.get('verbal_hesitation_count') or 0
    duration = features.get('duration') or 0

    if hesitation_count >= 2:
        return "hesitation"
    if duration > 12:
        return "long_thinking"
    return "generic"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStateAcknowledgmentDuringThinking:
    def test_no_features_falls_back_to_generic(self):
        assert _get_acknowledgment(None) == "generic"

    def test_empty_features_falls_back_to_generic(self):
        assert _get_acknowledgment({}) == "generic"

    def test_high_hesitation_count_triggers_hesitation_reaction(self):
        features = {"verbal_hesitation_count": 2, "duration": 5}
        assert _get_acknowledgment(features) == "hesitation"

    def test_hesitation_count_above_two_also_triggers_hesitation(self):
        features = {"verbal_hesitation_count": 5, "duration": 5}
        assert _get_acknowledgment(features) == "hesitation"

    def test_hesitation_count_below_two_does_not_trigger_hesitation(self):
        features = {"verbal_hesitation_count": 1, "duration": 5}
        assert _get_acknowledgment(features) != "hesitation"

    def test_long_duration_triggers_long_thinking_reaction(self):
        features = {"verbal_hesitation_count": 0, "duration": 15}
        assert _get_acknowledgment(features) == "long_thinking"

    def test_duration_at_threshold_does_not_trigger_long_thinking(self):
        features = {"verbal_hesitation_count": 0, "duration": 12}
        assert _get_acknowledgment(features) != "long_thinking"

    def test_hesitation_takes_priority_over_long_duration(self):
        # Both signals present — hesitation should be checked first.
        features = {"verbal_hesitation_count": 3, "duration": 20}
        assert _get_acknowledgment(features) == "hesitation"

    def test_short_duration_low_hesitation_falls_back_to_generic(self):
        features = {"verbal_hesitation_count": 0, "duration": 5}
        assert _get_acknowledgment(features) == "generic"

    def test_missing_keys_treated_as_zero(self):
        # Missing verbal_hesitation_count and duration should default to generic.
        features = {"transcript": "ocean"}
        assert _get_acknowledgment(features) == "generic"
