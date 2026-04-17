import ast
from pathlib import Path


def _load_module_ast(relative_path: str) -> ast.Module:
    return ast.parse(_load_source(relative_path))


def _load_source(relative_path: str) -> str:
    source_path = Path(__file__).resolve().parents[1] / relative_path
    return source_path.read_text(encoding="utf-8")


def _get_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                    return class_node
    raise AssertionError(f"Could not find {class_name}.{method_name}")


def _get_return_dict_keys(fn_node: ast.FunctionDef) -> list[str]:
    keys = []
    for node in fn_node.body:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
    return keys


def test_important_feature_extractor_includes_transcript_in_features():
    tree = _load_module_ast("multimodal_perception/audio/important_feature_extractor.py")
    extract_method = _get_method(tree, "ImportantFeaturesExtractor", "extract")
    assert "transcript" in _get_return_dict_keys(extract_method)


def test_receive_clue_does_not_resume_recording_after_repeat_prompt():
    tree = _load_module_ast("interaction/game_loop.py")
    receive_clue = _get_method(tree, "GameLoop", "receive_clue")

    resume_calls = [
        node
        for node in ast.walk(receive_clue)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resume_recording"
    ]
    repeat_prompt_calls = [
        node
        for node in ast.walk(receive_clue)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "say"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Oh, could you repeat the clue?"
    ]

    assert len(resume_calls) == 1, "Expected exactly one resume_recording call in receive_clue"
    assert len(repeat_prompt_calls) == 1, "Expected exactly one repeat prompt utterance in receive_clue"
    repeat_prompt_line = repeat_prompt_calls[0].lineno
    resumes_after_repeat_prompt = [node for node in resume_calls if node.lineno > repeat_prompt_line]
    assert not resumes_after_repeat_prompt, "resume_recording should not be called after the repeat prompt"
