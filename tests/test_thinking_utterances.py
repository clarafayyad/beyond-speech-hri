import ast
from pathlib import Path


def _get_thinking_utterances():
    source_path = Path(__file__).resolve().parents[1] / "agents" / "guesser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Guesser":
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == "get_random_thinking":
                    for stmt in class_node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == "reactions":
                                    if isinstance(stmt.value, ast.List):
                                        return [elt.value for elt in stmt.value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError("Could not find get_random_thinking reactions list in agents/guesser.py")


def test_thinking_utterances_are_simple_thinking_signals():
    assert _get_thinking_utterances() == [
        "Hmm.",
        "Hmm…",
        "Hmm, let's see.",
        "Let's see…",
        "Okay…",
        "Alright…",
        "One moment…",
        "Hmm, let me think.",
        "Just thinking…",
        "Thinking…",
    ]
