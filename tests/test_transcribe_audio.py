import importlib
import sys
import types
from unittest.mock import Mock


def _load_transcribe_module(monkeypatch, mock_model):
    fake_whisper = types.SimpleNamespace(load_model=Mock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    module = importlib.import_module("multimodal_perception.audio.transcribe_audio")
    return importlib.reload(module), fake_whisper


def _mock_transcribe_result():
    return {
        "text": " urban ",
        "segments": [
            {
                "words": [
                    {"word": " urban ", "start": 0.0, "end": 0.2},
                ]
            }
        ],
    }


def test_uses_initial_prompt_when_set(monkeypatch):
    mock_model = Mock()
    mock_model.transcribe.return_value = _mock_transcribe_result()
    module, fake_whisper = _load_transcribe_module(monkeypatch, mock_model)

    transcriber = module.WhisperTranscriber(initial_prompt="urban triangle")
    transcript, asr_words = transcriber.transcribe_audio("audio.wav")

    fake_whisper.load_model.assert_called_once_with("base")
    mock_model.transcribe.assert_called_once_with(
        "audio.wav",
        language="en",
        fp16=False,
        word_timestamps=True,
        initial_prompt="urban triangle",
    )
    assert transcript == " urban "
    assert asr_words == [{"word": "urban", "start": 0.0, "end": 0.2}]


def test_explicit_prompt_overrides_initial_prompt(monkeypatch):
    mock_model = Mock()
    mock_model.transcribe.return_value = _mock_transcribe_result()
    module, _ = _load_transcribe_module(monkeypatch, mock_model)

    transcriber = module.WhisperTranscriber(initial_prompt="default prompt")
    transcriber.transcribe_audio("audio.wav", prompt="override prompt")

    assert mock_model.transcribe.call_args.kwargs["initial_prompt"] == "override prompt"


def test_uses_default_prompt_when_not_set(monkeypatch):
    mock_model = Mock()
    mock_model.transcribe.return_value = _mock_transcribe_result()
    module, _ = _load_transcribe_module(monkeypatch, mock_model)

    transcriber = module.WhisperTranscriber()
    transcriber.transcribe_audio("audio.wav")

    assert mock_model.transcribe.call_args.kwargs["initial_prompt"] == module.DEFAULT_INITIAL_PROMPT


def test_can_disable_prompt_with_empty_string(monkeypatch):
    mock_model = Mock()
    mock_model.transcribe.return_value = _mock_transcribe_result()
    module, _ = _load_transcribe_module(monkeypatch, mock_model)

    transcriber = module.WhisperTranscriber(initial_prompt="")
    transcriber.transcribe_audio("audio.wav")

    assert "initial_prompt" not in mock_model.transcribe.call_args.kwargs
