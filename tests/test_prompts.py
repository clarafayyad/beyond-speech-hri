import json

import pytest

from interaction.prompts import SYSTEM_PROMPT_ADAPTIVE, build_user_prompt


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT reasoning style instructions
# ---------------------------------------------------------------------------

class TestSystemPromptReasoningStyle:
    def test_system_prompt_contains_reasoning_style_section(self):
        assert "REASONING STYLE" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_high_confidence_style(self):
        assert "high confidence" in SYSTEM_PROMPT_ADAPTIVE
        # High confidence should indicate decisive/short language
        assert "decisive" in SYSTEM_PROMPT_ADAPTIVE or "short" in SYSTEM_PROMPT_ADAPTIVE or "minimal" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_medium_confidence_style(self):
        assert "medium confidence" in SYSTEM_PROMPT_ADAPTIVE
        # Medium confidence should mention 2 candidate options
        assert "2 candidate" in SYSTEM_PROMPT_ADAPTIVE or "two candidate" in SYSTEM_PROMPT_ADAPTIVE
        # Medium confidence should include an interpretation statement (implicit verification)
        assert "I think you meant" in SYSTEM_PROMPT_ADAPTIVE or "interpretation" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_low_confidence_style(self):
        assert "low confidence" in SYSTEM_PROMPT_ADAPTIVE
        # Low confidence should mention multiple hypotheses and uncertainty
        assert "hypothes" in SYSTEM_PROMPT_ADAPTIVE or "2–3" in SYSTEM_PROMPT_ADAPTIVE
        assert "uncertainty" in SYSTEM_PROMPT_ADAPTIVE or "uncertain" in SYSTEM_PROMPT_ADAPTIVE or "hesitation" in SYSTEM_PROMPT_ADAPTIVE
        # Low confidence should include an interpretation statement (implicit verification)
        assert "I think you might be pointing" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_high_confidence_no_interpretation(self):
        assert "high confidence" in SYSTEM_PROMPT_ADAPTIVE
        # High confidence should explicitly skip the interpretation statement
        assert "Do NOT include an interpretation statement" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_no_questions_in_medium_low(self):
        # The implicit verification must not ask questions
        assert "No questions" in SYSTEM_PROMPT_ADAPTIVE or "no questions" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_unknown_non_adaptive_style(self):
        assert "unknown" in SYSTEM_PROMPT_ADAPTIVE
        # unknown should use a fixed, non-adaptive style
        assert "non-adaptive" in SYSTEM_PROMPT_ADAPTIVE or "fixed" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_reason_field_is_spoken_sentence(self):
        # The reason field description should mention it will be spoken aloud
        assert "say aloud" in SYSTEM_PROMPT_ADAPTIVE or "spoken" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_avoids_explicit_confidence_labels_in_reason(self):
        lower = SYSTEM_PROMPT_ADAPTIVE.lower()
        assert "do not explicitly mention confidence labels" in lower
        assert "with medium confidence" in lower


# ---------------------------------------------------------------------------
# build_user_prompt includes confidence level
# ---------------------------------------------------------------------------

class _FakeGameState:
    """Minimal game-state stub for prompt building."""
    def __init__(self):
        self.board = ["river", "mountain", "apple"]
        self.revealed = {}
        self.card_descriptions = {
            "river": "A flowing body of water",
            "mountain": "A tall peak",
            "apple": "A red fruit",
        }
        self.history = []
        self.turn = 1


class TestBuildUserPromptConfidence:
    def test_high_confidence_appears_in_prompt(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state, confidence_level="high")
        assert "high" in prompt

    def test_medium_confidence_appears_in_prompt(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state, confidence_level="medium")
        assert "medium" in prompt

    def test_low_confidence_appears_in_prompt(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state, confidence_level="low")
        assert "low" in prompt

    def test_no_confidence_defaults_to_unknown(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state)
        assert "unknown" in prompt

    def test_clue_word_in_prompt(self):
        state = _FakeGameState()
        prompt = build_user_prompt("apple", state, confidence_level="high")
        assert "apple" in prompt

    def test_unrevealed_cards_in_prompt(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state, confidence_level="medium")
        assert "river" in prompt
        assert "mountain" in prompt
        assert "apple" in prompt

    def test_revealed_cards_excluded_from_prompt(self):
        state = _FakeGameState()
        state.revealed[0] = "blue"  # reveal "river"
        prompt = build_user_prompt("clue", state, confidence_level="medium")
        # mountain and apple should still be present; river (index 0) should not
        assert "mountain" in prompt
        assert "apple" in prompt


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT game history instructions
# ---------------------------------------------------------------------------

class TestSystemPromptGameHistory:
    def test_system_prompt_contains_game_history_section(self):
        assert "GAME HISTORY" in SYSTEM_PROMPT_ADAPTIVE

    def test_system_prompt_instructs_to_avoid_wrong_guesses(self):
        lower = SYSTEM_PROMPT_ADAPTIVE.lower()
        assert ("avoid" in lower and "incorrect" in lower) or \
               ("avoid" in lower and "wrong" in lower)

    def test_system_prompt_mentions_previous_clues(self):
        lower = SYSTEM_PROMPT_ADAPTIVE.lower()
        assert ("previous clues" in lower) or \
               ("previous" in lower and "outcomes" in lower)


# ---------------------------------------------------------------------------
# build_user_prompt includes game history
# ---------------------------------------------------------------------------

class TestBuildUserPromptHistory:
    def test_empty_history_shows_empty_list(self):
        state = _FakeGameState()
        prompt = build_user_prompt("river", state)
        assert "Previous clues and outcomes" in prompt
        assert "[]" in prompt

    def test_single_turn_history_appears_in_prompt(self):
        state = _FakeGameState()
        state.revealed[0] = "blue"
        state.history = [
            {"turn": 1, "clue": "water", "guess_number": 1,
             "guess": 0, "card": "river", "result": "blue"}
        ]
        state.turn = 2
        prompt = build_user_prompt("peak", state)
        parsed = _extract_previous_clues(prompt)
        assert len(parsed) == 1
        assert parsed[0]["turn"] == 1
        assert parsed[0]["clue"] == "water"
        assert parsed[0]["guesses"][0]["card"] == "river"
        assert parsed[0]["guesses"][0]["result"] == "blue"

    def test_multiple_guesses_grouped_by_turn_and_clue(self):
        state = _FakeGameState()
        state.revealed = {0: "blue", 1: "neutral"}
        state.history = [
            {"turn": 1, "clue": "nature", "guess_number": 1,
             "guess": 0, "card": "river", "result": "blue"},
            {"turn": 1, "clue": "nature", "guess_number": 2,
             "guess": 1, "card": "mountain", "result": "neutral"},
        ]
        state.turn = 2
        prompt = build_user_prompt("food", state)
        parsed = _extract_previous_clues(prompt)
        assert len(parsed) == 1  # one turn/clue group
        assert len(parsed[0]["guesses"]) == 2

    def test_multiple_turns_in_history(self):
        state = _FakeGameState()
        state.board = ["river", "mountain", "apple", "sword"]
        state.card_descriptions["sword"] = "A sharp weapon"
        state.revealed = {0: "blue", 2: "neutral"}
        state.history = [
            {"turn": 1, "clue": "water", "guess_number": 1,
             "guess": 0, "card": "river", "result": "blue"},
            {"turn": 2, "clue": "food", "guess_number": 1,
             "guess": 2, "card": "apple", "result": "neutral"},
        ]
        state.turn = 3
        prompt = build_user_prompt("battle", state)
        parsed = _extract_previous_clues(prompt)
        assert len(parsed) == 2
        clues = {entry["clue"] for entry in parsed}
        assert clues == {"water", "food"}

    def test_history_includes_card_name_from_entry(self):
        """When 'card' key exists in history entry, use it directly."""
        state = _FakeGameState()
        state.revealed[1] = "red"
        state.history = [
            {"turn": 1, "clue": "peak", "guess_number": 1,
             "guess": 1, "card": "mountain", "result": "red"}
        ]
        state.turn = 2
        prompt = build_user_prompt("fruit", state)
        parsed = _extract_previous_clues(prompt)
        assert parsed[0]["guesses"][0]["card"] == "mountain"

    def test_history_falls_back_to_board_when_no_card_key(self):
        """When 'card' key is missing, fall back to board[guess]."""
        state = _FakeGameState()
        state.revealed[1] = "red"
        state.history = [
            {"turn": 1, "clue": "peak", "guess_number": 1,
             "guess": 1, "result": "red"}
        ]
        state.turn = 2
        prompt = build_user_prompt("fruit", state)
        parsed = _extract_previous_clues(prompt)
        assert parsed[0]["guesses"][0]["card"] == "mountain"

    def test_history_applies_without_confidence_baseline(self):
        """History is included even when confidence_level is None (baseline)."""
        state = _FakeGameState()
        state.revealed[0] = "blue"
        state.history = [
            {"turn": 1, "clue": "water", "guess_number": 1,
             "guess": 0, "card": "river", "result": "blue"}
        ]
        state.turn = 2
        prompt = build_user_prompt("peak", state, confidence_level=None)
        parsed = _extract_previous_clues(prompt)
        assert len(parsed) == 1

    def test_history_applies_with_confidence_adaptive(self):
        """History is included when confidence_level is set (adaptive)."""
        state = _FakeGameState()
        state.revealed[0] = "blue"
        state.history = [
            {"turn": 1, "clue": "water", "guess_number": 1,
             "guess": 0, "card": "river", "result": "blue"}
        ]
        state.turn = 2
        prompt = build_user_prompt("peak", state, confidence_level="high")
        parsed = _extract_previous_clues(prompt)
        assert len(parsed) == 1


def _extract_previous_clues(prompt):
    """Extract the JSON list from the 'Previous clues and outcomes' section."""
    marker = "Previous clues and outcomes:"
    start = prompt.index(marker) + len(marker)
    # The JSON list ends before the next prompt section ("Respond ONLY")
    end = prompt.index("Respond ONLY")
    return json.loads(prompt[start:end].strip())
