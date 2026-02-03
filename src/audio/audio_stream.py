import sounddevice as sd
import numpy as np

from src.audio.prosody import ProsodyService

SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 1.0)  # 1 second


class Prosody():
    def __init__(self):
        self.fluency = "unknown"
        self.arousal = "unknown"


class AudioStream():
    def __init__(self, prosody_service: ProsodyService):
        self.prosody_service = prosody_service
        self.prosody = Prosody()
        self.stream = sd.InputStream(channels=1, callback=self.audio_callback, samplerate=SAMPLE_RATE)
        self.audio_buffer = []

    def audio_callback(self, indata, frames, time, status):
        self.audio_buffer.extend(indata[:, 0])
        if len(self.audio_buffer) >= CHUNK_SIZE:
            chunk = np.array(self.audio_buffer[:CHUNK_SIZE])
            self.audio_buffer = self.audio_buffer[CHUNK_SIZE:]  # remove processed part
            self.prosody.fluency, self.prosody.arousal = self.prosody_service.process_chunk(chunk)

    def start(self):
        self.stream.start()

    def release(self):
        self.stream.stop()
        self.stream.close()
