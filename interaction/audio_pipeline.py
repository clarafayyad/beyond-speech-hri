import json
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

import soundfile as sf

from multimodal_perception.model.confidence_classifier import ConfidenceClassifier
from multimodal_perception.audio.important_feature_extractor import ImportantFeaturesExtractor
from multimodal_perception.audio.recorder import AudioRecorder
from multimodal_perception.audio.transcribe_audio import WhisperTranscriber

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

    def __init__(self, participant_id: str, audio_device_index=None, log_dir=LOG_DIR):
        self.participant_id = participant_id
        self.recorder = AudioRecorder(device_index=audio_device_index)
        # create a Whisper transcriber and pass it to the extractor
        whisper = WhisperTranscriber()
        self.extractor = ImportantFeaturesExtractor(whisper)
        # construct classifier with participant so it can auto-load calibration
        self.classifier = ConfidenceClassifier(participant_id=self.participant_id)

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"session_{participant_id}_{timestamp}.json")
        self._log_entries = []

    def start_recording(self):
        """Start capturing audio from the configured input device."""
        self.recorder.start()

    def pause_recording(self):
        """Pause capturing audio (e.g., while the robot is speaking)."""
        self.recorder.pause()

    def resume_recording(self):
        """Resume capturing audio after a pause."""
        self.recorder.resume()

    def _clip_last_seconds(self, audio_path: str, seconds: int = 60) -> Optional[str]:
        """Return a path to a WAV file that contains only the last `seconds` seconds
        of `audio_path`.

        Implementation notes:
        - Uses soundfile to get duration and ffmpeg to perform the actual clip.
        - If the audio is shorter than `seconds`, returns the original path.
        - On any error (missing ffmpeg, failure), returns the original path.
        """
        if audio_path is None:
            return None

        try:
            info = sf.info(audio_path)
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            # If we can't read the file info, don't attempt clipping
            return audio_path

        if duration <= seconds:
            # No clipping needed
            return audio_path

        start_time = max(0.0, duration - float(seconds))

        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            # Use ffmpeg to clip the last `seconds` seconds. Place -ss before -i for speed.
            # Example: ffmpeg -y -ss 120 -i input.wav -ac 1 -ar 16000 out.wav
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_time}",
                    "-i",
                    audio_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    tmp_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            return tmp_path
        except Exception:
            # If ffmpeg fails for any reason, fall back to original file
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            return audio_path

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
        tuple[dict, str]
            A ``(features, confidence_level)`` pair where *features* is the
            raw feature dict and *confidence_level* is ``"low"``,
            ``"medium"``, or ``"high"``.
        """
        audio_path = self.recorder.stop()

        # Clip the last 60 seconds of the recording before extracting features.
        clipped_path = self._clip_last_seconds(audio_path, seconds=60)

        features = self.extractor.extract(clipped_path)
        _, confidence_level = self.classifier.classify(features)

        # Clean up the temporary audio files now that features have been extracted
        # Remove clipped file if it is a different temporary file
        if clipped_path is not None and clipped_path != audio_path:
            try:
                os.unlink(clipped_path)
            except OSError:
                pass

        # Remove the original recording file
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        entry = {
            "participant_id": self.participant_id,
            "turn": turn,
            "clue": clue,
            "features": _to_serializable(features),
            "confidence_level": confidence_level,
        }
        self._log_entries.append(entry)
        self._save_log()

        print(f"[AudioPipeline] Turn {turn} | clue='{clue}' | confidence={confidence_level}")
        return features, confidence_level

    def _save_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._log_entries, f, indent=2)

    def stop_recording_if_active(self):
        """Stop the recorder if it's currently active and remove the temporary
        audio file that was created.

        Returns
        -------
        Optional[str]
            Path to the temporary audio file that was removed, or ``None`` if no
            active recording was found or removal failed.
        """
        # Attempt to stop the recorder; AudioRecorder.stop() returns the path to
        # the saved file or None if nothing was recorded. It is safe to call
        # even if recording was not started.
        try:
            audio_path = self.recorder.stop()
        except Exception:
            # If stop() raises for some reason, try to defensively close the
            # underlying stream if present and then give up.
            stream = getattr(self.recorder, "_stream", None)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                try:
                    self.recorder._stream = None
                except Exception:
                    pass
            return None

        if not audio_path:
            return None

        # Remove the temporary file produced by stop()
        try:
            os.unlink(audio_path)
            return audio_path
        except Exception:
            # If we couldn't remove the file, return None to indicate cleanup
            # didn't complete.
            return None


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # catches numpy scalars (float32, int64, etc.)
        return obj.item()
    else:
        return obj
