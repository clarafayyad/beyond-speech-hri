"""Tests for the utterance-limiting feature.

The robot should say at most MAX_PRE_GUESS_UTTERANCES before displaying a
guess, choosing the most relevant utterance(s) based on game context.
"""

import pytest

from interaction.utterance_selection import select_utterances, MAX_PRE_GUESS_UTTERANCES


# ---------------------------------------------------------------------------
# select_utterances – unit tests
# ---------------------------------------------------------------------------

class TestSelectUtterances:
    def test_returns_highest_priority_when_all_present(self):
        candidates = [
            (0, "confidence reaction"),
            (1, "continuity remark"),
            (2, "thinking filler"),
        ]
        result = select_utterances(candidates, max_count=1)
        assert result == ["confidence reaction"]

    def test_skips_none_values(self):
        candidates = [
            (0, None),
            (1, "continuity remark"),
            (2, "thinking filler"),
        ]
        result = select_utterances(candidates, max_count=1)
        assert result == ["continuity remark"]

    def test_skips_empty_strings(self):
        candidates = [
            (0, ""),
            (1, ""),
            (2, "thinking filler"),
        ]
        result = select_utterances(candidates, max_count=1)
        assert result == ["thinking filler"]

    def test_returns_empty_when_all_none(self):
        candidates = [
            (0, None),
            (1, None),
            (2, None),
        ]
        result = select_utterances(candidates, max_count=1)
        assert result == []

    def test_returns_empty_for_empty_candidates(self):
        assert select_utterances([], max_count=1) == []

    def test_respects_max_count_two(self):
        candidates = [
            (0, "confidence"),
            (1, "continuity"),
            (2, "thinking"),
        ]
        result = select_utterances(candidates, max_count=2)
        assert result == ["confidence", "continuity"]

    def test_respects_priority_ordering(self):
        # Candidates given in reverse priority order
        candidates = [
            (2, "thinking"),
            (0, "confidence"),
            (1, "continuity"),
        ]
        result = select_utterances(candidates, max_count=2)
        assert result == ["confidence", "continuity"]

    def test_default_max_count_is_constant(self):
        candidates = [
            (0, "confidence"),
            (1, "continuity"),
            (2, "thinking"),
        ]
        result = select_utterances(candidates)
        assert len(result) == MAX_PRE_GUESS_UTTERANCES

    def test_fewer_valid_than_max_count(self):
        candidates = [
            (0, None),
            (1, "continuity"),
            (2, None),
        ]
        result = select_utterances(candidates, max_count=3)
        assert result == ["continuity"]

    def test_max_pre_guess_utterances_is_one(self):
        """Verify the default limit is 1."""
        assert MAX_PRE_GUESS_UTTERANCES == 1


# ---------------------------------------------------------------------------
# Context-dependent priority – integration-style examples
# ---------------------------------------------------------------------------

class TestContextDependentPriority:
    """Demonstrate that the priority assignment in play_turn varies by turn.

    These tests build candidates exactly the way TurnManager.play_turn does
    and verify that the correct utterance wins.
    """

    def test_turn_zero_prefers_confidence(self):
        """On the first turn, confidence reaction beats thinking filler."""
        # Turn 0 layout: (0, confidence), (1, thinking)
        candidates = [(0, "confidence"), (1, "thinking")]
        assert select_utterances(candidates) == ["confidence"]

    def test_turn_zero_falls_back_to_thinking(self):
        """On the first turn with no confidence, thinking filler is used."""
        candidates = [(0, None), (1, "thinking")]
        assert select_utterances(candidates) == ["thinking"]

    def test_later_turn_prefers_continuity(self):
        """On later turns, continuity remark beats confidence reaction."""
        # Later-turn layout: (0, continuity), (1, confidence), (2, thinking)
        candidates = [
            (0, "continuity"),
            (1, "confidence"),
            (2, "thinking"),
        ]
        assert select_utterances(candidates) == ["continuity"]

    def test_later_turn_falls_back_to_confidence(self):
        """On later turns, if continuity is None, confidence reaction wins."""
        candidates = [
            (0, None),
            (1, "confidence"),
            (2, "thinking"),
        ]
        assert select_utterances(candidates) == ["confidence"]

    def test_later_turn_falls_back_to_thinking(self):
        """On later turns, if both are None, thinking filler wins."""
        candidates = [
            (0, None),
            (1, None),
            (2, "thinking"),
        ]
        assert select_utterances(candidates) == ["thinking"]
