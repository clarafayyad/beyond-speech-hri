"""Tests for confidence-aware say_random_thinking in Guesser."""
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out hardware / framework dependencies so the module is importable
# without physical devices or the SIC framework installed.
# ---------------------------------------------------------------------------
_STUBS = [
    "PIL", "PIL.Image",
    "sic_framework", "sic_framework.devices",
    "sic_framework.devices.desktop", "sic_framework.devices.naoqi_shared",
    "sic_framework.services", "sic_framework.services.dialogflow",
    "agents.dialog_manager", "agents.llm_agent",
    "agents.pepper_tablet", "agents.pepper_tablet.display_service",
    "agents.stt_manager", "interaction.audio_pipeline",
]
for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from agents.guesser import Guesser  # noqa: E402  (must come after stubs)
from multimodal_perception.model.confidence_classifier import (  # noqa: E402
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_guesser():
    """Return a bare Guesser with hardware dependencies replaced by mocks."""
    g = Guesser.__new__(Guesser)
    g.dialog_manager = MagicMock()
    return g


def _collect_reactions(guesser, confidence_level, n=200):
    """Call say_random_thinking *n* times and return the set of distinct phrases."""
    seen = set()
    for _ in range(n):
        with patch.object(guesser, "say", side_effect=lambda text, **kw: seen.add(text)):
            guesser.say_random_thinking(confidence_level)
    return seen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSayRandomThinkingConfidence:
    """say_random_thinking should pick phrases that match the confidence level."""

    # ---- high confidence ----

    def test_high_confidence_uses_decisive_phrases(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_HIGH)
        assert len(reactions) > 0
        for phrase in reactions:
            assert any(
                kw in phrase.lower()
                for kw in ["clear", "easy", "got it", "on it", "no hesitation"]
            ), f"Unexpected high-confidence phrase: {phrase!r}"

    def test_high_confidence_no_uncertainty_phrases(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_HIGH)
        combined = " ".join(reactions).lower()
        assert "tricky" not in combined
        assert "careful" not in combined
        assert "not sure" not in combined

    # ---- medium confidence ----

    def test_medium_confidence_mentions_options_or_candidates(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_MEDIUM)
        combined = " ".join(reactions).lower()
        assert any(kw in combined for kw in ["candidate", "options", "weighing", "compare", "possibilities"])

    def test_medium_confidence_no_blunt_decisive_phrases(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_MEDIUM)
        assert "On it." not in reactions
        assert "Easy. I know this one." not in reactions

    # ---- low confidence ----

    def test_low_confidence_expresses_uncertainty_or_caution(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_LOW)
        combined = " ".join(reactions).lower()
        assert any(kw in combined for kw in ["careful", "cautiously", "not sure", "tricky", "risk"])

    def test_low_confidence_no_decisive_phrases(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, CONFIDENCE_LOW)
        assert "On it." not in reactions
        assert "Easy. I know this one." not in reactions

    # ---- unknown / None ----

    def test_no_confidence_uses_generic_phrases(self):
        g = _make_guesser()
        reactions = _collect_reactions(g, None)
        combined = " ".join(reactions).lower()
        assert any(kw in combined for kw in ["thinking", "give me a second", "big brain", "analyzing"])

    def test_unknown_pool_differs_from_high_pool(self):
        g = _make_guesser()
        unknown_reactions = _collect_reactions(g, None)
        high_reactions = _collect_reactions(g, CONFIDENCE_HIGH)
        assert unknown_reactions != high_reactions

    # ---- general contract ----

    def test_exactly_one_phrase_said_per_call(self):
        g = _make_guesser()
        for level in [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, None]:
            said = []
            with patch.object(g, "say", side_effect=lambda text, **kw: said.append(text)):
                g.say_random_thinking(level)
            assert len(said) == 1, f"Expected exactly one phrase for level={level!r}, got {said}"

    def test_each_pool_contains_multiple_phrases(self):
        g = _make_guesser()
        for level in [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, None]:
            reactions = _collect_reactions(g, level, n=500)
            assert len(reactions) > 1, f"Pool for level={level!r} has only one phrase: {reactions}"
