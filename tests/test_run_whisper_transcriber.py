import importlib
import os
import sys
import types


def _load_module_with_fakes(monkeypatch, transcribe_result=("urban two", []), stop_result=None):
    calls = {"transcribe_audio_path": None, "start_called": False}

    class FakeRecorder:
        def __init__(self, device_index=None, sample_rate=16000, channels=2):
            self.device_index = device_index
            self.sample_rate = sample_rate
            self.channels = channels

        def start(self):
            calls["start_called"] = True

        def stop(self):
            return stop_result

    class FakeTranscriber:
        def transcribe_audio(self, path):
            calls["transcribe_audio_path"] = path
            return transcribe_result

    monkeypatch.setitem(
        sys.modules,
        "multimodal_perception.audio.recorder",
        types.SimpleNamespace(AudioRecorder=FakeRecorder),
    )
    monkeypatch.setitem(
        sys.modules,
        "multimodal_perception.audio.transcribe_audio",
        types.SimpleNamespace(WhisperTranscriber=FakeTranscriber),
    )
    sys.modules.pop("interaction.run_whisper_transcriber", None)
    module = importlib.import_module("interaction.run_whisper_transcriber")
    return module, calls


def test_main_uses_audio_path_without_recording(monkeypatch, capsys):
    module, calls = _load_module_with_fakes(monkeypatch)
    code = module.main(["--audio-path", "/tmp/example.wav"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Transcript:" in out
    assert "urban two" in out
    assert calls["transcribe_audio_path"] == "/tmp/example.wav"
    assert calls["start_called"] is False


def test_main_records_when_no_audio_path_and_handles_empty_recording(monkeypatch, capsys):
    module, calls = _load_module_with_fakes(monkeypatch, stop_result=None)
    monkeypatch.setattr("builtins.input", lambda: "")

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "No audio captured." in out
    assert calls["start_called"] is True


def test_main_deletes_temp_file_after_recording(monkeypatch, tmp_path):
    tmp_audio = tmp_path / "sample.wav"
    tmp_audio.write_bytes(b"fake")
    module, _ = _load_module_with_fakes(monkeypatch, stop_result=str(tmp_audio))
    monkeypatch.setattr("builtins.input", lambda: "")

    code = module.main([])

    assert code == 0
    assert not os.path.exists(tmp_audio)
