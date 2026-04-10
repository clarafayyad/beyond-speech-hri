"""Tests for the lightweight 2-agent inner discussion feature in TurnManager."""

import pytest
from unittest.mock import MagicMock, call

from interaction.turn_manager import TurnManager
from interaction.game_state import BLUE, RED, NEUTRAL
from interaction.prompts import INNER_DISCUSSION_SYSTEM_PROMPT, build_inner_discussion_prompt
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
        self.board = ["river", "mountain", "apple", "bridge", "seal"]
        self.revealed = {}
        self.card_descriptions = {
            "river": "A flowing body of water",
            "mountain": "A tall peak",
            "apple": "A red fruit",
            "bridge": "A structure crossing a gap",
            "seal": "A marine mammal",
        }
        self.history = []
        self.turn = 0
        self.game_over = False
        self.win = None


def _make_turn_manager():
    guesser = MagicMock()
    game_state = _FakeGameState()
    return TurnManager(guesser, game_state), guesser, game_state


_VALID_DISCUSSION_RESPONSE = {
    "player1_line": "Could it be 'Seal'?",
    "player2_line": "Maybe, but that risks the assassin. 'Bridge' is safer.",
    "final_decision": "I'll go with 'Bridge'.",
    "guess_index": 3,  # "bridge" is at index 3
}


# ---------------------------------------------------------------------------
# _should_trigger_inner_discussion
# ---------------------------------------------------------------------------

class TestShouldTriggerInnerDiscussion:
    def test_low_confidence_first_call_returns_true(self):
        tm, _, _ = _make_turn_manager()
        assert tm._should_trigger_inner_discussion(CONFIDENCE_LOW) is True

    def test_medium_confidence_returns_false(self):
        tm, _, _ = _make_turn_manager()
        assert tm._should_trigger_inner_discussion(CONFIDENCE_MEDIUM) is False

    def test_high_confidence_returns_false(self):
        tm, _, _ = _make_turn_manager()
        assert tm._should_trigger_inner_discussion(CONFIDENCE_HIGH) is False

    def test_none_confidence_returns_false(self):
        tm, _, _ = _make_turn_manager()
        assert tm._should_trigger_inner_discussion(None) is False

    def test_low_confidence_after_discussion_used_returns_false(self):
        tm, _, _ = _make_turn_manager()
        tm.inner_discussion_used = True
        assert tm._should_trigger_inner_discussion(CONFIDENCE_LOW) is False

    def test_initial_inner_discussion_used_is_false(self):
        tm, _, _ = _make_turn_manager()
        assert tm.inner_discussion_used is False


# ---------------------------------------------------------------------------
# _run_inner_discussion
# ---------------------------------------------------------------------------

class TestRunInnerDiscussion:
    def test_marks_discussion_as_used(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        assert tm.inner_discussion_used is True

    def test_says_player1_line(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        guesser.say.assert_any_call("Could it be 'Seal'?")

    def test_says_clara_line(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        guesser.say_as_clara.assert_called_once_with(
            "Maybe, but that risks the assassin. 'Bridge' is safer."
        )

    def test_says_final_decision(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        guesser.say.assert_any_call("I'll go with 'Bridge'.")

    def test_displays_guess_card(self):
        tm, guesser, game_state = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        guesser.display_guess.assert_called_once_with(game_state.board[3])

    def test_returns_correct_guess_index(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        result = tm._run_inner_discussion("water")
        assert result == 3

    def test_returns_none_for_invalid_index(self):
        tm, guesser, _ = _make_turn_manager()
        invalid_response = dict(_VALID_DISCUSSION_RESPONSE, guess_index=99)
        guesser.prompt_llm.return_value = invalid_response
        result = tm._run_inner_discussion("water")
        assert result is None

    def test_returns_none_for_already_revealed_index(self):
        tm, guesser, game_state = _make_turn_manager()
        game_state.revealed[3] = BLUE  # bridge already revealed
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        result = tm._run_inner_discussion("water")
        assert result is None

    def test_uses_inner_discussion_system_prompt(self):
        tm, guesser, _ = _make_turn_manager()
        guesser.prompt_llm.return_value = _VALID_DISCUSSION_RESPONSE
        tm._run_inner_discussion("water")
        call_kwargs = guesser.prompt_llm.call_args
        assert call_kwargs[1]["system_prompt"] == INNER_DISCUSSION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# play_turn integration
# ---------------------------------------------------------------------------

class TestPlayTurnInnerDiscussionIntegration:
    def _simulate_turn(self, tm, guesser, game_state, confidence_level,
                       result=BLUE, llm_response=None):
        """Simulate one turn, auto-revealing the card that was guessed."""
        if llm_response is None:
            llm_response = {"guess_index": 0, "reason": "test"}

        def fake_prompt_llm(**kwargs):
            # If asked for inner discussion, return valid discussion response
            if kwargs.get("system_prompt") == INNER_DISCUSSION_SYSTEM_PROMPT:
                return _VALID_DISCUSSION_RESPONSE
            return llm_response

        guesser.prompt_llm.side_effect = fake_prompt_llm

        # Auto-reveal whatever card would be guessed
        def fake_display(card):
            # find index and auto-reveal
            idx = game_state.board.index(card)
            game_state.revealed[idx] = result

        guesser.display_guess.side_effect = fake_display

        tm.play_turn("water", 1, confidence_level)

    def test_discussion_triggered_on_low_confidence(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, BLUE)
        guesser.say_inner_discussion_intro.assert_called_once()

    def test_discussion_not_triggered_on_medium_confidence(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_MEDIUM, BLUE)
        guesser.say_inner_discussion_intro.assert_not_called()

    def test_discussion_not_triggered_on_high_confidence(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_HIGH, BLUE)
        guesser.say_inner_discussion_intro.assert_not_called()

    def test_discussion_only_triggers_once_per_game(self):
        tm, guesser, game_state = _make_turn_manager()
        # First turn: low confidence → discussion fires
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, RED)
        guesser.say_inner_discussion_intro.reset_mock()

        # Second turn: low confidence again → should NOT fire again
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, BLUE)
        guesser.say_inner_discussion_intro.assert_not_called()

    def test_discussion_used_flag_set_after_trigger(self):
        tm, guesser, game_state = _make_turn_manager()
        self._simulate_turn(tm, guesser, game_state, CONFIDENCE_LOW, BLUE)
        assert tm.inner_discussion_used is True

    def test_fallback_to_make_guess_when_discussion_returns_none(self):
        tm, guesser, game_state = _make_turn_manager()

        def fake_prompt_llm(**kwargs):
            if kwargs.get("system_prompt") == INNER_DISCUSSION_SYSTEM_PROMPT:
                # Return invalid index to force fallback
                return dict(_VALID_DISCUSSION_RESPONSE, guess_index=99)
            return {"guess_index": 0, "reason": "fallback"}

        guesser.prompt_llm.side_effect = fake_prompt_llm
        # Auto-reveal card 0 on display
        game_state.revealed[0] = BLUE
        guesser.display_guess.side_effect = None

        tm.play_turn("water", 1, CONFIDENCE_LOW)
        # prompt_llm should have been called twice: once for discussion, once for fallback
        assert guesser.prompt_llm.call_count == 2


# ---------------------------------------------------------------------------
# Inner discussion prompts
# ---------------------------------------------------------------------------

class TestInnerDiscussionPrompt:
    def test_system_prompt_mentions_player1(self):
        assert "Player 1" in INNER_DISCUSSION_SYSTEM_PROMPT

    def test_system_prompt_mentions_clara(self):
        assert "Clara" in INNER_DISCUSSION_SYSTEM_PROMPT

    def test_system_prompt_json_schema_fields(self):
        assert "player1_line" in INNER_DISCUSSION_SYSTEM_PROMPT
        assert "player2_line" in INNER_DISCUSSION_SYSTEM_PROMPT
        assert "final_decision" in INNER_DISCUSSION_SYSTEM_PROMPT
        assert "guess_index" in INNER_DISCUSSION_SYSTEM_PROMPT

    def test_build_inner_discussion_prompt_contains_clue(self):
        game_state = _FakeGameState()
        prompt = build_inner_discussion_prompt("water", game_state)
        assert "water" in prompt

    def test_build_inner_discussion_prompt_contains_unrevealed_cards(self):
        game_state = _FakeGameState()
        prompt = build_inner_discussion_prompt("water", game_state)
        assert "bridge" in prompt
        assert "seal" in prompt

    def test_build_inner_discussion_prompt_excludes_revealed_cards(self):
        game_state = _FakeGameState()
        game_state.revealed[3] = BLUE  # bridge revealed
        prompt = build_inner_discussion_prompt("water", game_state)
        # bridge should not appear in the unrevealed list
        import json
        # The prompt JSON should not contain bridge's index
        assert '"index": 3' not in prompt


# ---------------------------------------------------------------------------
# Guesser say_as_clara
# ---------------------------------------------------------------------------

class TestGuesserSayAsClara:
    def _make_guesser_with_method(self, method_name):
        """Return a mock Guesser with the named real method bound to it."""
        from agents.guesser import Guesser
        guesser = MagicMock(spec=Guesser)
        real_method = getattr(Guesser, method_name)
        setattr(guesser, method_name, real_method.__get__(guesser, type(guesser)))
        return guesser

    def test_say_as_clara_prefixes_with_clara(self):
        guesser = self._make_guesser_with_method("say_as_clara")
        guesser.say_as_clara("Bridge is safer.")
        guesser.say.assert_called_once_with("Clara: Bridge is safer.")

    def test_say_inner_discussion_intro_says_something(self):
        guesser = self._make_guesser_with_method("say_inner_discussion_intro")
        guesser.say_inner_discussion_intro()
        guesser.say.assert_called_once()
        phrase = guesser.say.call_args[0][0]
        assert isinstance(phrase, str) and len(phrase) > 0
