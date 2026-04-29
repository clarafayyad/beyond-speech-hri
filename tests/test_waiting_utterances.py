"""Tests for robot utterances while waiting for the spymaster's clue.

Covers:
- contains_hesitation_trigger() detects stress / difficulty words
- Guesser.get_waiting_for_clue_hesitation_utterance() returns a non-empty string
- Guesser.get_waiting_for_clue_long_wait_utterance() returns a non-empty string
- GameLoop constants: LONG_WAIT_THRESHOLD_SECONDS and MAX_LONG_WAIT_REACTIONS
- GameLoop.receive_clue() long-wait escalation: first long wait → long-wait utterance;
  second long wait → continuity remark (with fallback to long-wait utterance)
"""

import ast
from pathlib import Path

import pytest

from multimodal_perception.audio.verbal_hesitation import contains_hesitation_trigger


def _get_game_loop_constant(name):
    """Read a module-level constant from game_loop.py via AST without importing
    the module (which requires hardware-level dependencies)."""
    source_path = Path(__file__).resolve().parents[1] / "interaction" / "game_loop.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    raise AssertionError(f"Could not find constant '{name}' in interaction/game_loop.py")


# ---------------------------------------------------------------------------
# contains_hesitation_trigger
# ---------------------------------------------------------------------------

class TestContainsHesitationTrigger:
    def test_stress_word_shit(self):
        assert contains_hesitation_trigger("shit") is True

    def test_stress_word_damn(self):
        assert contains_hesitation_trigger("damn") is True

    def test_stress_word_fuck(self):
        assert contains_hesitation_trigger("fuck") is True

    def test_difficulty_phrase(self):
        assert contains_hesitation_trigger("that's difficult") is True

    def test_difficulty_phrase_hard(self):
        assert contains_hesitation_trigger("this is hard") is True

    def test_meta_difficulty_tricky(self):
        assert contains_hesitation_trigger("it's tricky") is True

    def test_plain_filler_no_trigger(self):
        assert contains_hesitation_trigger("uh") is False

    def test_plain_filler_hmm_no_trigger(self):
        assert contains_hesitation_trigger("hmm") is False

    def test_empty_string_no_trigger(self):
        assert contains_hesitation_trigger("") is False

    def test_none_no_trigger(self):
        assert contains_hesitation_trigger(None) is False

    def test_normal_sentence_no_trigger(self):
        assert contains_hesitation_trigger("ocean two") is False

    def test_case_insensitive(self):
        assert contains_hesitation_trigger("SHIT") is True
        assert contains_hesitation_trigger("Difficult") is True


# ---------------------------------------------------------------------------
# GameLoop constants
# ---------------------------------------------------------------------------

class TestGameLoopConstants:
    def test_long_wait_threshold_is_20s(self):
        assert _get_game_loop_constant("LONG_WAIT_THRESHOLD_SECONDS") == 20

    def test_max_long_wait_reactions_is_2(self):
        assert _get_game_loop_constant("MAX_LONG_WAIT_REACTIONS") == 2


# ---------------------------------------------------------------------------
# receive_clue() long-wait escalation (AST-based structural check)
# ---------------------------------------------------------------------------

def _get_receive_clue_long_wait_ast():
    """Return the body of the long-wait if-block inside receive_clue().

    Looks for the assignment to ``is_first_long_wait`` to locate the correct
    branch, then returns the containing if-block body for structural checks.
    """
    source_path = Path(__file__).resolve().parents[1] / "interaction" / "game_loop.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GameLoop":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "receive_clue":
                    for n in ast.walk(item):
                        if isinstance(n, ast.If):
                            for stmt in n.body:
                                if (isinstance(stmt, ast.Assign)
                                        and any(
                                            isinstance(t, ast.Name) and t.id == "is_first_long_wait"
                                            for t in stmt.targets
                                        )):
                                    return n.body
    raise AssertionError("Could not find long-wait escalation block in receive_clue()")


class TestReceiveClueLongWaitEscalation:
    """Structural tests for the long-wait escalation logic in receive_clue()."""

    def test_is_first_long_wait_variable_assigned(self):
        """receive_clue must assign is_first_long_wait to distinguish first vs later waits."""
        body = _get_receive_clue_long_wait_ast()
        names = [
            t.id
            for stmt in body
            if isinstance(stmt, ast.Assign)
            for t in stmt.targets
            if isinstance(t, ast.Name)
        ]
        assert "is_first_long_wait" in names, (
            "receive_clue() must assign 'is_first_long_wait' inside the long-wait block"
        )

    def test_get_continuity_remark_called_on_second_long_wait(self):
        """get_continuity_remark() must appear in the else-branch of the long-wait block."""
        body = _get_receive_clue_long_wait_ast()
        for stmt in body:
            if isinstance(stmt, ast.If):
                call_names = [
                    node.func.attr
                    for node in ast.walk(ast.Module(body=stmt.orelse, type_ignores=[]))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                ]
                assert "get_continuity_remark" in call_names, (
                    "The else-branch (second+ long wait) must call get_continuity_remark()"
                )
                return
        raise AssertionError("No if/else branch found for is_first_long_wait in long-wait block")

    def test_get_long_wait_utterance_in_fallback(self):
        """get_waiting_for_clue_long_wait_utterance() must be the fallback in the else-branch."""
        body = _get_receive_clue_long_wait_ast()
        for stmt in body:
            if isinstance(stmt, ast.If):
                call_names = [
                    node.func.attr
                    for node in ast.walk(ast.Module(body=stmt.orelse, type_ignores=[]))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                ]
                assert "get_waiting_for_clue_long_wait_utterance" in call_names, (
                    "The else-branch must fall back to get_waiting_for_clue_long_wait_utterance()"
                )
                return
        raise AssertionError("No if/else branch found for is_first_long_wait in long-wait block")


# ---------------------------------------------------------------------------
# Guesser utterance methods (tested via AST to avoid hardware deps)
# ---------------------------------------------------------------------------
