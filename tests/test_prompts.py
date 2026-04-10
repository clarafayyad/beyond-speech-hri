import pytest

from interaction.prompts import SYSTEM_PROMPT, build_user_prompt


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT reasoning style instructions
# ---------------------------------------------------------------------------

class TestSystemPromptReasoningStyle:
    def test_system_prompt_contains_reasoning_style_section(self):
        assert "REASONING STYLE" in SYSTEM_PROMPT

    def test_system_prompt_high_confidence_style(self):
        assert "high confidence" in SYSTEM_PROMPT
        # High confidence should indicate decisive/short language
        assert "decisive" in SYSTEM_PROMPT or "short" in SYSTEM_PROMPT or "minimal" in SYSTEM_PROMPT

    def test_system_prompt_medium_confidence_style(self):
        assert "medium confidence" in SYSTEM_PROMPT
        # Medium confidence should mention 2 candidate options
        assert "2 candidate" in SYSTEM_PROMPT or "two candidate" in SYSTEM_PROMPT
        # Medium confidence should include an interpretation statement (implicit verification)
        assert "I think you meant" in SYSTEM_PROMPT or "interpretation" in SYSTEM_PROMPT

    def test_system_prompt_low_confidence_style(self):
        assert "low confidence" in SYSTEM_PROMPT
        # Low confidence should mention multiple hypotheses and uncertainty
        assert "hypothes" in SYSTEM_PROMPT or "2–3" in SYSTEM_PROMPT
        assert "uncertainty" in SYSTEM_PROMPT or "uncertain" in SYSTEM_PROMPT or "hesitation" in SYSTEM_PROMPT
        # Low confidence should include an interpretation statement (implicit verification)
        assert "I think you might be pointing" in SYSTEM_PROMPT

    def test_system_prompt_high_confidence_no_interpretation(self):
        assert "high confidence" in SYSTEM_PROMPT
        # High confidence should explicitly skip the interpretation statement
        assert "Do NOT include an interpretation statement" in SYSTEM_PROMPT

    def test_system_prompt_no_questions_in_medium_low(self):
        # The implicit verification must not ask questions
        assert "No questions" in SYSTEM_PROMPT or "no questions" in SYSTEM_PROMPT

    def test_system_prompt_unknown_non_adaptive_style(self):
        assert "unknown" in SYSTEM_PROMPT
        # unknown should use a fixed, non-adaptive style
        assert "non-adaptive" in SYSTEM_PROMPT or "fixed" in SYSTEM_PROMPT

    def test_system_prompt_reason_field_is_spoken_sentence(self):
        # The reason field description should mention it will be spoken aloud
        assert "say aloud" in SYSTEM_PROMPT or "spoken" in SYSTEM_PROMPT


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
