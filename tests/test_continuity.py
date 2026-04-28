import pytest

from interaction.continuity import (
    _last_turn_performance,
    get_baseline_continuity_utterance,
    get_adaptive_continuity_utterance,
)
from interaction.game_state import BLUE, RED, NEUTRAL


# ---------------------------------------------------------------------------
# Minimal game-state stub (avoids real server / board setup)
# ---------------------------------------------------------------------------

class _FakeGameState:
    def __init__(self, turn=0, history=None, confidence_history=None):
        self.board = ["river", "mountain", "apple", "castle", "forest"]
        self.revealed = {}
        self.history = history or []
        self.confidence_history = confidence_history or []
        self.turn = turn
        self.game_over = False
        self.win = None


# ---------------------------------------------------------------------------
# Helper: _last_turn_performance
# ---------------------------------------------------------------------------

class TestLastTurnPerformance:
    def test_no_history_returns_none(self):
        gs = _FakeGameState(turn=1)
        assert _last_turn_performance(gs) is None

    def test_turn_zero_returns_none(self):
        gs = _FakeGameState(turn=0)
        assert _last_turn_performance(gs) is None

    def test_all_blue(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
            {"turn": 0, "clue": "water", "guess_number": 2, "guess": 1, "card": "mountain", "result": BLUE},
        ])
        perf = _last_turn_performance(gs)
        assert perf["all_correct"] is True
        assert perf["any_correct"] is True
        assert perf["blue_count"] == 2
        assert perf["total"] == 2

    def test_mixed_results(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
            {"turn": 0, "clue": "water", "guess_number": 2, "guess": 1, "card": "mountain", "result": RED},
        ])
        perf = _last_turn_performance(gs)
        assert perf["all_correct"] is False
        assert perf["any_correct"] is True

    def test_all_wrong(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": NEUTRAL},
        ])
        perf = _last_turn_performance(gs)
        assert perf["all_correct"] is False
        assert perf["any_correct"] is False

    def test_only_looks_at_last_turn(self):
        gs = _FakeGameState(turn=2, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
            {"turn": 1, "clue": "fire", "guess_number": 1, "guess": 1, "card": "mountain", "result": RED},
        ])
        perf = _last_turn_performance(gs)
        assert perf["all_correct"] is False
        assert perf["any_correct"] is False


# ---------------------------------------------------------------------------
# Baseline continuity utterances
# ---------------------------------------------------------------------------

class TestBaselineContinuity:
    def test_returns_none_on_turn_zero(self):
        gs = _FakeGameState(turn=0)
        assert get_baseline_continuity_utterance(gs) is None

    def test_returns_none_after_good_turn(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
        ])
        assert get_baseline_continuity_utterance(gs) is None

    def test_returns_none_after_bad_turn(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": NEUTRAL},
        ])
        assert get_baseline_continuity_utterance(gs) is None

    def test_returns_none_after_mixed_turn(self):
        gs = _FakeGameState(turn=1, history=[
            {"turn": 0, "clue": "water", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
            {"turn": 0, "clue": "water", "guess_number": 2, "guess": 1, "card": "mountain", "result": RED},
        ])
        assert get_baseline_continuity_utterance(gs) is None


# ---------------------------------------------------------------------------
# Adaptive continuity utterances
# ---------------------------------------------------------------------------

class TestAdaptiveContinuity:
    def test_returns_none_on_turn_zero(self):
        gs = _FakeGameState(turn=0)
        assert get_adaptive_continuity_utterance(gs) is None

    def test_all_correct_low_trend(self):
        gs = _FakeGameState(
            turn=2,
            history=[
                {"turn": 0, "clue": "a", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
                {"turn": 1, "clue": "b", "guess_number": 1, "guess": 1, "card": "mountain", "result": BLUE},
            ],
            confidence_history=["low", "low"],
        )
        utterance = get_adaptive_continuity_utterance(gs)
        assert isinstance(utterance, str)
        assert len(utterance) > 0

    def test_all_correct_high_trend(self):
        gs = _FakeGameState(
            turn=2,
            history=[
                {"turn": 0, "clue": "a", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
                {"turn": 1, "clue": "b", "guess_number": 1, "guess": 1, "card": "mountain", "result": BLUE},
            ],
            confidence_history=["high", "high"],
        )
        utterance = get_adaptive_continuity_utterance(gs)
        assert isinstance(utterance, str)
        assert len(utterance) > 0

    def test_no_correct_low_trend(self):
        gs = _FakeGameState(
            turn=2,
            history=[
                {"turn": 0, "clue": "a", "guess_number": 1, "guess": 0, "card": "river", "result": NEUTRAL},
                {"turn": 1, "clue": "b", "guess_number": 1, "guess": 1, "card": "mountain", "result": RED},
            ],
            confidence_history=["low", "low"],
        )
        utterance = get_adaptive_continuity_utterance(gs)
        assert isinstance(utterance, str)
        assert len(utterance) > 0

    def test_mixed_results_no_trend_returns_none(self):
        gs = _FakeGameState(
            turn=2,
            history=[
                {"turn": 0, "clue": "a", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
                {"turn": 1, "clue": "b", "guess_number": 1, "guess": 1, "card": "mountain", "result": BLUE},
                {"turn": 1, "clue": "b", "guess_number": 2, "guess": 2, "card": "apple", "result": RED},
            ],
            confidence_history=["high", "medium"],
        )
        assert get_adaptive_continuity_utterance(gs) is None

    def test_empty_confidence_history_still_returns_utterance(self):
        gs = _FakeGameState(
            turn=1,
            history=[
                {"turn": 0, "clue": "a", "guess_number": 1, "guess": 0, "card": "river", "result": BLUE},
            ],
            confidence_history=[],
        )
        utterance = get_adaptive_continuity_utterance(gs)
        assert isinstance(utterance, str)
        assert len(utterance) > 0
