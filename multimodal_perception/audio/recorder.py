import tempfile
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Records audio from the specified input device into a WAV file."""

    def __init__(self, device_index=None, sample_rate=16000, channels=2):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels

        self._frames = []
        self._stream = None
        # _paused_event is set when recording is paused; the audio callback
        # checks this flag in a thread-safe manner using threading.Event.
        self._paused_event = threading.Event()

    def start(self):
        """Begin recording audio in the background."""
        self._frames = []
        self._paused_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()

    def pause(self):
        """Pause recording; incoming audio frames are discarded until resumed."""
        self._paused_event.set()

    def resume(self):
        """Resume recording after a pause."""
        self._paused_event.clear()

    def _callback(self, indata, frames, time, status):
        if not self._paused_event.is_set():
            self._frames.append(indata.copy())

    def stop(self):
        """
        Stop recording and save the captured audio to a temporary WAV file.

        Returns the path to the saved file, or None if nothing was recorded.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return None

        audio = np.concatenate(self._frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, self.sample_rate)
        return tmp.name
