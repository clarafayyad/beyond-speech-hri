import random
import re
import asyncio
import wave
from os import environ
from threading import Thread
from time import sleep, strftime

import numpy as np
from sic_framework.core import sic_logging
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.core.sic_application import SICApplication
from sic_framework.devices import Pepper
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.desktop import Desktop
from sic_framework.devices.device import SICDeviceManager
from sic_framework.services.dialogflow.dialogflow import DialogflowConf
from sic_framework.services.llm import GPTConf, GPTRequest
from dotenv import load_dotenv

from agents.stt_manager import RealTimeSTTService, DialogFlowSTTService
from agents.tts_manager import NaoqiTTSConf, TTSConf, TTSCacher, ElevenLabsTTSConf, ElevenLabsTTS


class InteractionConf:

    def __init__(self, speaking_rate=None, sleep_time=0, animated=True, max_attempts=2, amplified=False,
                 always_regenerate=False, real_time_stt=True):
        self.speaking_rate = speaking_rate
        self.sleep_time = sleep_time
        self.animated = animated
        self.max_attempts = max_attempts
        self.amplified = amplified
        self.always_regenerate = always_regenerate
        self.real_time_stt = real_time_stt

    @staticmethod
    def apply_config_defaults(config_attr, param_names):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                config = getattr(self, config_attr)
                for name in param_names:
                    if kwargs.get(name) is None:
                        kwargs[name] = getattr(config, name)
                return func(self, *args, **kwargs)

            return wrapper

        return decorator


class DialogManager:
    def __init__(self, device_manager: SICDeviceManager, dialogflow_conf: DialogflowConf,
                 tts_conf: TTSConf = NaoqiTTSConf, microphone_device=None):

        print("\n SETTING UP BASIC PROCESSING")
        # Development Logging
        self.app = SICApplication()
        self.logger = self.app.get_app_logger()
        self.app.set_log_level(sic_logging.DEBUG)  # can be DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.app.set_log_file("./logs")

        # Data logging
        self._log_queue = None
        self._log_thread = None

        # Interaction configuration
        self.interaction_conf = InteractionConf()

        # Background loop
        self.background_loop = asyncio.new_event_loop()
        self.background_thread = Thread(target=self._start_loop, daemon=True)
        self.background_thread.start()
        print('Complete')

        load_dotenv("../config/.env")

        print("\n SETTING UP OPENAI")
        try:
            # Setup GPT client
            conf = GPTConf(openai_key=environ["OPENAI_API_KEY"])
            # self.gpt = GPT(conf=conf)
        except KeyError:
            self.logger.warning("No openAI key available")
            self.gpt = None
        print('Complete')

        print("\n SETTING UP TTS")
        self.tts_conf = tts_conf
        if isinstance(self.tts_conf, ElevenLabsTTSConf):
            self.sample_rate = 22050
            self.tts = ElevenLabsTTS(elevenlabs_key=environ["ELEVENLABS_API_KEY"],
                                     voice_id=self.tts_conf.voice_id,
                                     model_id=self.tts_conf.model_id,
                                     sample_rate=self.sample_rate,
                                     speaking_rate=self.tts_conf.speaking_rate,
                                     stability=self.tts_conf.stability)
            connect_to_elevenlabs_future = asyncio.run_coroutine_threadsafe(self.tts.connect(),
                                                                            self.background_loop)
            try:
                connect_to_elevenlabs_future.result()
                asyncio.run_coroutine_threadsafe(self.tts.speak("Initializing text to speech"),
                                                 self.background_loop).result()
                print('Elevenlabs TTS activated')
            except Exception as e:
                self.logger.error("Failed to connect to elevenlabs", exc_info=e)
        elif isinstance(self.tts_conf, NaoqiTTSConf):
            pass
        else:
            raise ValueError(f"Unknown tts_conf {self.tts_conf}")
        self.tts_cacher = TTSCacher()
        print("Complete")

        print("\n SETTING UP DEVICE MANAGER")
        if microphone_device:
            print("\n Additional Microphone Device Detected")
            self.mic = microphone_device.mic
        self.device_manager = device_manager
        if isinstance(self.device_manager, Pepper):
            print("\n Device is PEPPER")
            self.speaker = self.device_manager.speaker
            if not microphone_device:
                self.mic = self.device_manager.mic
        elif isinstance(self.device_manager, Desktop):
            print("\n Device is COMPUTER")
            self.speaker = self.device_manager.speakers
            if not microphone_device:
                self.mic = self.device_manager.mic
        else:
            raise ValueError(f"DeviceManager {self.device_manager} is currently not supported")
        print("Complete")

        print("\n SETTING UP STT")
        if self.interaction_conf.real_time_stt:
            # TODO: pass mic index as a param to this func
            self.stt_service = RealTimeSTTService(mic_index=3)
        else:
            self.stt_service = DialogFlowSTTService(mic_index=self.mic, dialogflow_conf=dialogflow_conf)
        print("Complete and ready for interaction!")

    def log_writer(self, log_path):
        with open(log_path, 'a', encoding='utf-8') as f:
            while True:
                item = self._log_queue.get()
                if item is None:
                    break  # Exit signal
                f.write(item + '\n')
                f.flush()

    def log_utterance(self, speaker, text):
        if self._log_queue:
            timestamp = strftime("%Y-%m-%d %H:%M:%S")
            self._log_queue.put(f"[{timestamp}] {speaker}: {text}")

    def naoqi_say(self, text, sleep_time=None, animated=False):
        self.device_manager.tts.request(
            NaoqiTextToSpeechRequest(text, animated=animated, language='English'))

        # Sleep if requested
        if sleep_time and sleep_time > 0:
            sleep(sleep_time)

    @InteractionConf.apply_config_defaults('interaction_conf', ['speaking_rate', 'sleep_time', 'animated', 'amplified',
                                                                'always_regenerate'])
    def say(self, text, speaking_rate, sleep_time=None, animated=None, amplified=False, always_regenerate=False):
        print("Saying: ", text)
        if isinstance(self.tts_conf, NaoqiTTSConf):
            self.naoqi_say(text, sleep_time=sleep_time, animated=animated)
        elif isinstance(self.tts_conf, ElevenLabsTTSConf):
            self.elevenlabs_say(text, sleep_time=sleep_time, amplified=amplified,
                                always_regenerate=always_regenerate, chunking=True)
        else:
            raise ValueError(f'Unsupported tts_conf type: {type(self.tts_conf)}')

    def elevenlabs_say(self, text, sleep_time=None, amplified=False, always_regenerate=False, chunking=True):
        if not chunking:
            text_chunks = [text]
        else:
            text_chunks = self._split_text(text, max_len=80)

        for chunk in text_chunks:
            # Normalize and hash text
            tts_key = self.tts_cacher.make_tts_key(chunk, self.tts_conf)

            if not always_regenerate:
                audio_file = self.tts_cacher.load_audio_file(tts_key)
                if audio_file:
                    self.log_utterance(speaker='robot', text=f'{chunk} (cache)')
                    self.play_audio(audio_file, log=False)
                    continue

            # Generate new audio
            audio_bytes = self.elevenlabs_generate_chunk_audio(chunk, amplified)

            # Play audio
            self.speaker.request(AudioRequest(audio_bytes, self.sample_rate))
            self.log_utterance(speaker='robot', text=f'{chunk}')

            # Sleep if requested
            if sleep_time and sleep_time > 0:
                sleep(sleep_time)

    def play_audio(self, audio_file, amplified=False, log=True):
        with wave.open(audio_file, 'rb') as wf:
            # Get parameters
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            # Ensure format is 16-bit (2 bytes per sample)
            if sample_width != 2:
                raise ValueError("WAV file is not 16-bit audio. Sample width = {} bytes.".format(sample_width))

            audio = wf.readframes(n_frames)
            if amplified:
                audio = self._amplify_audio(audio)

            self.speaker.request(AudioRequest(audio, framerate))
            if log:
                self.log_utterance(speaker='robot', text=f'plays {audio_file}')

    def elevenlabs_generate_audio(self, text, amplified=False, renew_all=False):
        text_chunks = self._split_text(text, max_len=80)
        for chunk in text_chunks:
            if not renew_all:
                tts_key = self.tts_cacher.make_tts_key(chunk, self.tts_conf)
                if tts_key in self.tts_cacher.tts_cache:
                    continue
            self.elevenlabs_generate_chunk_audio(chunk, amplified)

    def elevenlabs_generate_chunk_audio(self, text, amplified=False):
        # Normalize and hash text
        tts_key = self.tts_cacher.make_tts_key(text, self.tts_conf)

        # ElevenLabs TTS returns bytes
        audio_bytes = asyncio.run_coroutine_threadsafe(self.tts.speak(text), self.background_loop).result()

        if audio_bytes and amplified:
            audio_bytes = self._amplify_audio(audio_bytes)

        # Save to cache file
        self.tts_cacher.save_audio_file(tts_key, audio_bytes, self.sample_rate)

        return audio_bytes

    @staticmethod
    def _split_text(text: str, max_len: int = 80, min_tail: int = 20):
        """
            Split text into natural chunks of ~max_len characters.
            - First, split by sentence boundaries (.?!)
            - Then, split long sentences further at commas or spaces
              while avoiding tiny fragments at the end.
            """
        text = text.strip()

        if len(text) <= max_len:
            return [text]

        chunks = []

        # Step 1: split at sentence boundaries, including no-space cases
        sentences = re.split(r'(?<=[,.?!])(?=\s|[A-Z])', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            while len(sentence) > max_len:
                # Try to find a good split point
                chunk = sentence[:max_len]

                # Prefer splitting at last comma or space in chunk
                break_pos = max(chunk.rfind(','), chunk.rfind(' '))

                if break_pos == -1 or break_pos < max_len // 3:
                    # fallback: just split at max_len
                    break_pos = max_len

                # Avoid leaving tiny tail
                if len(sentence) - break_pos < min_tail:
                    break_pos = len(sentence)

                chunks.append(sentence[:break_pos].strip())
                sentence = sentence[break_pos:].strip()

            if sentence:
                chunks.append(sentence)

        return chunks

    def listen(self):
        print("Listening...")
        transcript = self.stt_service.listen()
        print("Heard: ", transcript)
        return transcript

    def _start_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    @staticmethod
    def _amplify_audio(waveform_bytes, compression_strength=2.0, target_level=0.9):
        """
        Amplify audio by normalizing and applying dynamic range compression.

        :param waveform_bytes: Raw PCM audio data as bytes (int16)
        :param compression_strength: Compression strength (1.0=minimal, 2.0=moderate, 5.0=heavy)
        :param target_level: Final output level (0.0-1.0, recommend 0.9 to avoid clipping)
        :return: Processed audio as bytes (int16)
        """
        # Convert bytes to numpy array
        audio_data = np.frombuffer(waveform_bytes, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32767.0

        # Step 1: Initial normalization to [-1, 1] range
        max_val = np.max(np.abs(audio_float))
        if max_val > 0:
            audio_normalized = audio_float / max_val
        else:
            audio_normalized = audio_float

        # Step 2: Apply logarithmic compression to boost quiet parts
        sign = np.sign(audio_normalized)
        magnitude = np.abs(audio_normalized)
        compressed_magnitude = np.log1p(magnitude * compression_strength) / np.log1p(compression_strength)
        compressed_audio = sign * compressed_magnitude

        # Step 3: Final normalization and scaling to target level
        final_max = np.max(np.abs(compressed_audio))
        if final_max > 0:
            compressed_audio = compressed_audio / final_max * target_level

        # Convert back to int16 bytes
        audio_int16 = (compressed_audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def animate_thinking(self):
        if not isinstance(self.device_manager, Pepper):
            return
        thinking_animations = [
            "animations/Stand/Gestures/Thinking_1",
            "animations/Stand/Gestures/Thinking_3",
            "animations/Stand/Gestures/Thinking_4",
            "animations/Stand/Gestures/Thinking_6",
            "animations/Stand/Gestures/Thinking_8"
        ]
        self.device_manager.motion.request(NaoqiAnimationRequest(random.choice(thinking_animations)), block=False)
