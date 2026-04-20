import importlib
import sys
import types


def _load_transcribe_module_with_fake_whisper(monkeypatch):
    model_holder = {}
    loaded_model_name = {}

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            model_holder["transcribe_args"] = args
            model_holder["transcribe_kwargs"] = kwargs
            return {
                "text": "urban two",
                "segments": [
                    {
                        "words": [
                            {"word": " urban", "start": 0.0, "end": 0.2},
                            {"word": " two.", "start": 0.2, "end": 0.4},
                        ]
                    }
                ],
            }

    fake_model = FakeModel()

    def load_model(name):
        loaded_model_name["name"] = name
        return fake_model

    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(load_model=load_model))
    sys.modules.pop("multimodal_perception.audio.transcribe_audio", None)
    module = importlib.import_module("multimodal_perception.audio.transcribe_audio")
    return module, loaded_model_name, model_holder


def test_whisper_transcriber_uses_small_en_model(monkeypatch):
    module, loaded_model_name, _ = _load_transcribe_module_with_fake_whisper(monkeypatch)
    module.WhisperTranscriber()
    assert loaded_model_name["name"] == "small.en"


def test_transcribe_audio_uses_more_constrained_whisper_settings(monkeypatch):
    module, _, model_holder = _load_transcribe_module_with_fake_whisper(monkeypatch)
    transcriber = module.WhisperTranscriber()

    transcript, words = transcriber.transcribe_audio("/tmp/fake.wav")

    assert transcript == "urban two"
    # FakeModel uses leading/trailing punctuation to validate clean_asr_word().
    assert words == [
        {"word": "urban", "start": 0.0, "end": 0.2},
        {"word": "two", "start": 0.2, "end": 0.4},
    ]

    kwargs = model_holder["transcribe_kwargs"]
    assert model_holder["transcribe_args"] == ("/tmp/fake.wav",)
    assert kwargs["task"] == "transcribe"
    assert kwargs["language"] == "en"
    assert kwargs["fp16"] is False
    assert kwargs["word_timestamps"] is True
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False
    assert "urban two" in kwargs["initial_prompt"]
    assert "shell two" in kwargs["initial_prompt"]
