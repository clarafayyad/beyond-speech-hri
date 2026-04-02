import json
import os
from datetime import datetime

from multimodal_perception.audio.dummy_classifier import DummyConfidenceClassifier
from multimodal_perception.audio.dummy_extractor import DummyFeatureExtractor
from multimodal_perception.audio.recorder import AudioRecorder

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "..", "logs")


class AudioPipeline:
    """
    Per-turn audio pipeline: record → extract features → classify confidence
    → log results.

    Parameters
    ----------
    participant_id : str
        Identifier for the current participant, used in log entries and the
        output filename.
    audio_device_index : int or None
        Index of the audio input device to record from.  ``None`` uses the
        system default.
    log_dir : str
        Directory where the JSON log file is written.
    """

    def __init__(self, participant_id, audio_device_index=None, log_dir=LOG_DIR):
        self.participant_id = participant_id
        self.recorder = AudioRecorder(device_index=audio_device_index)
        self.extractor = DummyFeatureExtractor()
        self.classifier = DummyConfidenceClassifier()

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"session_{participant_id}_{timestamp}.json")
        self._log_entries = []

    def start_recording(self):
        """Start capturing audio from the configured input device."""
        self.recorder.start()

    def stop_and_process(self, clue, turn):
        """
        Stop recording, run the feature-extraction and classification pipeline,
        persist the result to the session log, and return the predicted
        confidence level.

        Parameters
        ----------
        clue : str
            The clue word given by the spymaster this turn.
        turn : int
            Current turn number (used in the log entry).

        Returns
        -------
        str
            Predicted confidence level: ``"low"``, ``"medium"``, or ``"high"``.
        """
        audio_path = self.recorder.stop()

        features = self.extractor.extract(audio_path)
        confidence_level = self.classifier.classify(features)

        # Clean up the temporary audio file now that features have been extracted
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        entry = {
            "participant_id": self.participant_id,
            "turn": turn,
            "clue": clue,
            "features": features,
            "confidence_level": confidence_level,
        }
        self._log_entries.append(entry)
        self._save_log()

        print(f"[AudioPipeline] Turn {turn} | clue='{clue}' | confidence={confidence_level}")
        return confidence_level

    def _save_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._log_entries, f, indent=2)
