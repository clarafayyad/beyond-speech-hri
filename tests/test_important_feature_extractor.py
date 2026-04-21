import importlib
import sys
import types


def _load_module_with_fakes(monkeypatch):
    fake_feature_extractor = types.SimpleNamespace(
        load_audio=lambda _path: ("raw-audio", 16000),
        trim_silence=lambda y: f"trim({y})",
        normalize_audio=lambda y: f"norm({y})",
        reduce_noise=lambda y, _sr: f"denoise({y})",
        extract_pause_features_vad=lambda _y, _sr: (2, 0.4, 0.9),
        pause_position_features=lambda _asr_words: (1, 1),
        count_hesitation_words=lambda _transcript: 3,
        extract_speech_rate=lambda _transcript, _duration: 2.5,
        extract_mfcc_features=lambda _y, _sr: {"mfcc_2_mean": 0.7},
        energy_features=lambda _y: (0.0, 0.2, 0.3, 0.0, 0.0),
        extract_voice_quality=lambda _path: (0.0, 0.0, 1.1),
    )
    fake_librosa = types.SimpleNamespace(get_duration=lambda y, sr: 4.0)

    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "multimodal_perception.audio.feature_extractor", fake_feature_extractor)

    sys.modules.pop("multimodal_perception.audio.important_feature_extractor", None)
    return importlib.import_module("multimodal_perception.audio.important_feature_extractor")


def test_extract_transcribes_from_loaded_audio_buffer(monkeypatch):
    module = _load_module_with_fakes(monkeypatch)
    transcribe_calls = []

    class FakeWhisper:
        def transcribe_audio(self, audio_input):
            transcribe_calls.append(audio_input)
            return "urban two", [{"word": "urban", "start": 0.0, "end": 0.2}]

    extractor = module.ImportantFeaturesExtractor(FakeWhisper())
    features = extractor.extract("/tmp/example.wav")

    assert transcribe_calls == ["raw-audio"]
    assert features["transcript"] == "urban two"
    assert features["duration"] == 4.0
