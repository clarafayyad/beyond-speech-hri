"""Tests for the pre-guess utterance limiting feature.

The robot should say exactly one utterance before displaying a guess,
chosen via a simple fallback chain that depends on the turn number:

* Turn 0 (no history): confidence reaction → thinking filler
* Later turns: continuity remark → confidence reaction → thinking filler
"""

import pytest


# ---------------------------------------------------------------------------
# Helper: mirrors the fallback logic used in TurnManager.play_turn()
# ---------------------------------------------------------------------------

def _pick_utterance(turn, continuity_text, confidence_text, thinking_text):
    """Pure-function version of the fallback chain in play_turn().

    This keeps tests focused on the selection logic without needing to
    instantiate a full TurnManager with its heavy dependencies.
    """
    if turn == 0:
        return confidence_text or thinking_text
    return continuity_text or confidence_text or thinking_text


# ---------------------------------------------------------------------------
# Turn 0 – no game history available
# ---------------------------------------------------------------------------

class TestTurnZeroFallback:
    def test_prefers_confidence_reaction(self):
        result = _pick_utterance(0, None, "confidence", "thinking")
        assert result == "confidence"

    def test_falls_back_to_thinking_when_no_confidence(self):
        result = _pick_utterance(0, None, None, "thinking")
        assert result == "thinking"

    def test_falls_back_to_thinking_when_confidence_empty(self):
        result = _pick_utterance(0, None, "", "thinking")
        assert result == "thinking"

    def test_ignores_continuity_even_if_present(self):
        """Continuity is not considered on turn 0."""
        result = _pick_utterance(0, "continuity", "confidence", "thinking")
        assert result == "confidence"


# ---------------------------------------------------------------------------
# Later turns – game history exists
# ---------------------------------------------------------------------------

class TestLaterTurnFallback:
    def test_prefers_continuity_remark(self):
        result = _pick_utterance(1, "continuity", "confidence", "thinking")
        assert result == "continuity"

    def test_falls_back_to_confidence_when_no_continuity(self):
        result = _pick_utterance(2, None, "confidence", "thinking")
        assert result == "confidence"

    def test_falls_back_to_confidence_when_continuity_empty(self):
        result = _pick_utterance(3, "", "confidence", "thinking")
        assert result == "confidence"

    def test_falls_back_to_thinking_when_only_thinking(self):
        result = _pick_utterance(1, None, None, "thinking")
        assert result == "thinking"

    def test_falls_back_to_thinking_when_others_empty(self):
        result = _pick_utterance(1, "", "", "thinking")
        assert result == "thinking"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_always_returns_something_because_thinking_never_none(self):
        """get_random_thinking() always returns a non-empty string, so the
        fallback chain always produces a result in practice."""
        result = _pick_utterance(0, None, None, "filler")
        assert result
        result = _pick_utterance(5, None, None, "filler")
        assert result
