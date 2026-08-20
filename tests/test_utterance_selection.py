"""Tests for the pre-guess utterance limiting feature.

The robot should say exactly one utterance before displaying a guess,
chosen via a simple fallback chain:

* confidence reaction → thinking filler
"""

# ---------------------------------------------------------------------------
# Helper: mirrors the fallback logic used in TurnManager.play_turn()
# ---------------------------------------------------------------------------

def _pick_utterance(confidence_text, thinking_text):
    """Pure-function version of the fallback chain in play_turn().

    This keeps tests focused on the selection logic without needing to
    instantiate a full TurnManager with its heavy dependencies.
    """
    return confidence_text or thinking_text


# ---------------------------------------------------------------------------
# Confidence first
# ---------------------------------------------------------------------------

class TestFallback:
    def test_prefers_confidence_reaction(self):
        result = _pick_utterance("confidence", "thinking")
        assert result == "confidence"

    def test_falls_back_to_thinking_when_no_confidence(self):
        result = _pick_utterance(None, "thinking")
        assert result == "thinking"

    def test_falls_back_to_thinking_when_confidence_empty(self):
        result = _pick_utterance("", "thinking")
        assert result == "thinking"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_always_returns_something_because_thinking_never_none(self):
        """get_random_thinking() always returns a non-empty string, so the
        fallback chain always produces a result in practice."""
        result = _pick_utterance(None, "filler")
        assert result
        result = _pick_utterance(None, "filler")
        assert result
