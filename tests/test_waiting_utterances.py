"""Tests for robot utterances while waiting for the spymaster's clue.

Covers:
- contains_hesitation_trigger() detects stress / difficulty words
- Guesser.get_waiting_for_clue_hesitation_utterance() returns a non-empty string
- Guesser.get_waiting_for_clue_long_wait_utterance() returns a non-empty string
- GameLoop constants: LONG_WAIT_THRESHOLD_SECONDS and MAX_LONG_WAIT_REACTIONS
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
# Guesser utterance methods (tested via AST to avoid hardware deps)
# ---------------------------------------------------------------------------

def _get_method_reactions(method_name):
    """Parse guesser.py and return the list of strings in the 'reactions'
    variable of the named Guesser method."""
    source_path = Path(__file__).resolve().parents[1] / "agents" / "guesser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Guesser":
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                    for stmt in class_node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == "reactions":
                                    if isinstance(stmt.value, ast.List):
                                        return [
                                            elt.value
                                            for elt in stmt.value.elts
                                            if isinstance(elt, ast.Constant)
                                        ]
    raise AssertionError(f"Could not find '{method_name}' reactions list in agents/guesser.py")


class TestWaitingForClueHesitationUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"


class TestWaitingForClueLongWaitUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"



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
# Guesser utterance methods (tested via AST to avoid hardware deps)
# ---------------------------------------------------------------------------

def _get_method_reactions(method_name):
    """Parse guesser.py and return the list of strings in the 'reactions'
    variable of the named Guesser method."""
    source_path = Path(__file__).resolve().parents[1] / "agents" / "guesser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Guesser":
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                    for stmt in class_node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == "reactions":
                                    if isinstance(stmt.value, ast.List):
                                        return [
                                            elt.value
                                            for elt in stmt.value.elts
                                            if isinstance(elt, ast.Constant)
                                        ]
    raise AssertionError(f"Could not find '{method_name}' reactions list in agents/guesser.py")


class TestWaitingForClueHesitationUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"


class TestWaitingForClueLongWaitUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"



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
# Guesser utterance methods (tested via AST to avoid hardware deps)
# ---------------------------------------------------------------------------

def _get_method_reactions(method_name):
    """Parse guesser.py and return the list of strings in the 'reactions'
    variable of the named Guesser method."""
    source_path = Path(__file__).resolve().parents[1] / "agents" / "guesser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Guesser":
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                    for stmt in class_node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == "reactions":
                                    if isinstance(stmt.value, ast.List):
                                        return [
                                            elt.value
                                            for elt in stmt.value.elts
                                            if isinstance(elt, ast.Constant)
                                        ]
    raise AssertionError(f"Could not find '{method_name}' reactions list in agents/guesser.py")


class TestWaitingForClueHesitationUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_hesitation_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"


class TestWaitingForClueLongWaitUtterances:
    def test_returns_non_empty_list(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        assert len(reactions) > 0

    def test_all_items_are_short_strings(self):
        reactions = _get_method_reactions("get_waiting_for_clue_long_wait_utterance")
        for r in reactions:
            assert isinstance(r, str)
            assert len(r) > 0, "Utterance must not be empty"
            assert len(r) <= 80, f"Utterance too long: {r!r}"
