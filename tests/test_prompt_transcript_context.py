from interaction.prompts import SYSTEM_PROMPT_ADAPTIVE, build_user_prompt


class _FakeGameState:
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


def test_build_user_prompt_includes_transcribed_clue_utterance():
    state = _FakeGameState()
    prompt = build_user_prompt(
        "grow",
        state,
        confidence_level="high",
        clue_transcript="This is tricky... Grow two.",
    )
    assert 'Transcribed clue utterance: "This is tricky... Grow two."' in prompt


def test_system_prompt_adaptive_mentions_uncertainty_override():
    lower = SYSTEM_PROMPT_ADAPTIVE.lower()
    assert "important override" in lower
    assert "do not act as high" in lower


def test_build_user_prompt_marks_transcript_unavailable():
    state = _FakeGameState()
    prompt = build_user_prompt("grow", state, confidence_level="medium")
    assert 'Transcribed clue utterance: "(not available)"' in prompt
