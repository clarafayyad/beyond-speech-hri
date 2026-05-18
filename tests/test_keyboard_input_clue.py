import ast
from pathlib import Path


def _get_game_loop_tree():
    source_path = Path(__file__).resolve().parents[1] / "interaction" / "game_loop.py"
    return ast.parse(source_path.read_text(encoding="utf-8"))


def _get_game_loop_method(name: str) -> ast.FunctionDef:
    tree = _get_game_loop_tree()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GameLoop":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"Could not find GameLoop.{name}() in interaction/game_loop.py")


class TestKeyboardClueInput:
    def test_receive_clue_reads_keyboard_input_after_failed_parsing(self):
        method = _get_game_loop_method("receive_clue")
        call_names = [
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "_read_keyboard_clue_input" in call_names

    def test_receive_clue_stores_typed_clue_for_next_parse_attempt(self):
        method = _get_game_loop_method("receive_clue")
        assigned_names = [
            target.id
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        assert "pending_keyboard_clue" in assigned_names

    def test_keyboard_reader_uses_input_prompt(self):
        method = _get_game_loop_method("_read_keyboard_clue_input")
        call_names = [
            node.func.id
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "input" in call_names
