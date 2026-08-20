from pathlib import Path


def test_main_exposes_keyboard_toggle_and_wires_it_into_interaction_conf():
    main_source = Path("src/main.py").read_text(encoding="utf-8")

    assert "use_keyboard_input = False" in main_source
    assert "keyboard_input=use_keyboard_input" in main_source
