import asyncio
import base64
import logging
import os
import hashlib
import string
import wave
from json import dumps, loads, load, dump

from enum import Enum


class TTSService(Enum):
    GOOGLE = 1
    ELEVENLABS = 2


class TTSConf:

    def __init__(self):
        pass


class GoogleTTSConf(TTSConf):

    def __init__(self, speaking_rate=1.0, google_tts_voice_name="nl-NL-Standard-D", google_tts_voice_gender="FEMALE"):
        super().__init__()
        self.speaking_rate = speaking_rate
        self.google_tts_voice_name = google_tts_voice_name
        self.google_tts_voice_gender = google_tts_voice_gender


class NaoqiTTSConf(TTSConf):
    def __init__(self):
        super().__init__()


class TTSCacher:

    def __init__(self, tts_cache_dir='tts_cache', tts_cache_map_file_name='tts_cache_map.json', subfolder_depth=2):
        self.tts_cache_dir = tts_cache_dir
        self.tts_cache_map_file = os.path.join(tts_cache_dir, tts_cache_map_file_name)
        self.subfolder_depth = subfolder_depth

        self.tts_cache = self._load_cache()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, strip, remove punctuation for consistent caching"""
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def make_tts_key(self, text: str, voice_conf: TTSConf) -> str:
        """Generate a hash key based on text + TTS parameters"""
        if isinstance(voice_conf, GoogleTTSConf):
            payload = {
                "text": self.normalize_text(text),
                'tts_service': "GOOGLE",
                "speaking_rate": voice_conf.speaking_rate,
                "setting_1": voice_conf.google_tts_voice_name,
                "setting_2": voice_conf.google_tts_voice_gender,
            }
        else:
            raise ValueError(f'Voice Conf {voice_conf} is not supported.')

        # Sort keys to ensure deterministic JSON
        canonical = dumps(payload, sort_keys=True)
        return hashlib.md5(canonical.encode("utf-8")).hexdigest()

    def save_audio_file(self, tts_key: str, audio_bytes: bytes, sample_rate: int, sample_width: int = 2, channels: int = 1):
        subfolder = os.path.join(self.tts_cache_dir, tts_key[:self.subfolder_depth])
        os.makedirs(subfolder, exist_ok=True)
        filename = os.path.join(subfolder, f"{tts_key}.wav")

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)  # 2 bytes = 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        self.tts_cache[tts_key] = filename
        self._save_cache()

    def load_audio_file(self, tts_key):
        if tts_key in self.tts_cache:
            # Cached audio exists, play it
            audio_file = self.tts_cache[tts_key]
            if os.path.exists(audio_file):
                return audio_file
            else:
                del self.tts_cache[tts_key]
        return None

    def _load_cache(self) -> dict:
        if os.path.exists(self.tts_cache_map_file):
            with open(self.tts_cache_map_file, "r") as f:
                return load(f)
        return {}

    def _save_cache(self):
        with open(self.tts_cache_map_file, "w") as f:
            dump(self.tts_cache, f, indent=2)
