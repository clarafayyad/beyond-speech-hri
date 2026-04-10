"""Tests for the minimal interaction memory feature in TurnManager."""

import pytest
from unittest.mock import MagicMock, patch

from interaction.turn_manager import TurnManager
from interaction.game_state import BLUE, RED, NEUTRAL
from multimodal_perception.model.confidence_classifier import (
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_HIGH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeGameState:
    """Minimal game-state stub."""
    def __init__(self):
        self.board = ["river", "mountain", "apple", "bridge", "cloud"]
        self.revealed = {}
        self.card_descriptions = {
            "river": "A flowing body of water",
            "mountain": "A tall peak",
            "apple": "A red fruit",
            "bridge": "A structure crossing a gap",
            "cloud": "A mass of water vapor",
        }
        self.history = []
        self.turn = 0
        self.game_over = False
        self.win = None


def _make_turn_manager():
    guesser = MagicMock()
    game_state = _FakeGameState()
    return TurnManager(guesser, game_state), guesser, game_state


# ---------------------------------------------------------------------------
# _build_turn_memory
# ---------------------------------------------------------------------------

class TestBuildTurnMemory:
    def test_returns_correct_confidence(self):
        tm, _, _ = _make_turn_manager()
        memory = tm._build_turn_memory(CONFIDENCE_LOW, [RED])
        assert memory["confidence"] == CONFIDENCE_LOW

    def test_had_correct_guess_true_when_blue_in_results(self):
        tm, _, _ = _make_turn_manager()
        memory = tm._build_turn_memory(CONFIDENCE_MEDIUM, [RED, BLUE])
        assert memory["had_correct_guess"] is True

    def test_had_correct_guess_false_when_no_blue(self):
        tm, _, _ = _make_turn_manager()
        memory = tm._build_turn_memory(CONFIDENCE_LOW, [RED, NEUTRAL])
        assert memory["had_correct_guess"] is False

    def test_had_correct_guess_false_for_empty_results(self):
        tm, _, _ = _make_turn_manager()
        memory = tm._build_turn_memory(CONFIDENCE_HIGH, [])
        assert memory["had_correct_guess"] is False


# ---------------------------------------------------------------------------
# _should_say_memory_reference
# ---------------------------------------------------------------------------

class TestShouldSayMemoryReference:
    def test_no_memory_returns_false(self):
        tm, _, _ = _make_turn_manager()
        assert tm.last_turn_memory is None
        assert tm._should_say_memory_reference(CONFIDENCE_LOW) is False

    def test_both_low_confidence_returns_true(self):
        tm, _, _ = _make_turn_manager()
        tm.last_turn_memory = {"confidence": CONFIDENCE_LOW, "had_correct_guess": False}
        assert tm._should_say_memory_reference(CONFIDENCE_LOW) is True

    def test_previous_low_current_medium_returns_false(self):
        tm, _, _ = _make_turn_manager()
        tm.last_turn_memory = {"confidence": CONFIDENCE_LOW, "had_correct_guess": False}
        assert tm._should_say_memory_reference(CONFIDENCE_MEDIUM) is False

    def test_previous_medium_current_low_returns_false(self):
        tm, _, _ = _make_turn_manager()
        tm.last_turn_memory = {"confidence": CONFIDENCE_MEDIUM, "had_correct_guess": True}
        assert tm._should_say_memory_reference(CONFIDENCE_LOW) is False

    def test_both_high_returns_false(self):
        tm, _, _ = _make_turn_manager()
        tm.last_turn_memory = {"confidence": CONFIDENCE_HIGH, "had_correct_guess": True}
        assert tm._should_say_memory_reference(CONFIDENCE_HIGH) is False

    def test_previous_none_confidence_current_low_returns_false(self):
        tm, _, _ = _make_turn_manager()
        tm.last_turn_memory = {"confidence": None, "had_correct_guess": False}
        assert tm._should_say_memory_reference(CONFIDENCE_LOW) is False


# ---------------------------------------------------------------------------
# Memory reference is called (or not) in play_turn
# ---------------------------------------------------------------------------

class TestPlayTurnMemoryIntegration:
    def _simulate_turn(self, tm, guesser, game_state, confidence_level, result=BLUE):
        """Simulate one turn by setting up the mocks and calling play_turn."""
        guess_idx = 0
        guesser.prompt_llm.return_value = {"guess_index": guess_idx, "reason": "test"}
        # Provide feedback immediately
        game_state.revealed[guess_idx] = result
        tm.play_turn("river", 1, confidence_level)

    def test_memory_reference_not_called_on_first_turn(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, RED)
        guesser.say_memory_reference.assert_not_called()

    def test_memory_reference_called_on_repeated_low_confidence(self):
        tm, guesser, game_state = _make_turn_manager()
        # First turn: low confidence, misses (RED)
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, RED)
        guesser.say_memory_reference.reset_mock()

        # Second turn: low confidence again → memory reference should fire
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, BLUE)
        guesser.say_memory_reference.assert_called_once_with(
            {"confidence": CONFIDENCE_LOW, "had_correct_guess": False}
        )

    def test_memory_reference_not_called_when_confidence_changes(self):
        tm, guesser, game_state = _make_turn_manager()
        # First turn: low confidence
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, RED)
        guesser.say_memory_reference.reset_mock()

        # Second turn: high confidence → no memory reference
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_HIGH, BLUE)
        guesser.say_memory_reference.assert_not_called()

    def test_last_turn_memory_updated_after_turn(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_MEDIUM, BLUE)
        assert tm.last_turn_memory is not None
        assert tm.last_turn_memory["confidence"] == CONFIDENCE_MEDIUM
        assert tm.last_turn_memory["had_correct_guess"] is True

    def test_last_turn_memory_reflects_failed_guess(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, RED)
        assert tm.last_turn_memory["had_correct_guess"] is False


# ---------------------------------------------------------------------------
# say_memory_reference on Guesser (phrase sanity check)
# ---------------------------------------------------------------------------

class TestGuesserSayMemoryReference:
    def _make_guesser_with_say(self):
        """Return a real Guesser-like object that captures what was said."""
        from agents.guesser import Guesser
        guesser = MagicMock(spec=Guesser)
        # Restore the real method but bind it to the mock
        guesser.say_memory_reference = Guesser.say_memory_reference.__get__(guesser, type(guesser))
        return guesser

    def test_says_something_when_had_correct_guess(self):
        guesser = self._make_guesser_with_say()
        memory = {"confidence": CONFIDENCE_LOW, "had_correct_guess": True}
        guesser.say_memory_reference(memory)
        guesser.say.assert_called_once()
        phrase = guesser.say.call_args[0][0]
        assert isinstance(phrase, str) and len(phrase) > 0

    def test_says_something_when_no_correct_guess(self):
        guesser = self._make_guesser_with_say()
        memory = {"confidence": CONFIDENCE_LOW, "had_correct_guess": False}
        guesser.say_memory_reference(memory)
        guesser.say.assert_called_once()

    def test_says_nothing_when_memory_is_none(self):
        guesser = self._make_guesser_with_say()
        guesser.say_memory_reference(None)
        guesser.say.assert_not_called()
