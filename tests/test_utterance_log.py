from datetime import datetime

from interaction.utterance_log import build_utterance_log_path, format_utterance_log_line


def test_build_utterance_log_path_contains_participant_and_date(tmp_path):
    path = build_utterance_log_path(
        participant_id="p01",
        log_dir=str(tmp_path),
        now=datetime(2026, 4, 17, 10, 0, 0),
    )
    assert path.endswith("utterances_p01_20260417.txt")


def test_build_utterance_log_path_returns_none_without_participant():
    assert build_utterance_log_path(participant_id=None) is None
    assert build_utterance_log_path(participant_id="") is None


def test_format_utterance_log_line_uses_timestamp_and_speaker():
    line = format_utterance_log_line(
        speaker="spymaster",
        text="ocean 2",
        now=datetime(2026, 4, 17, 10, 30, 45),
    )
    assert line == "[2026-04-17 10:30:45] spymaster: ocean 2"
