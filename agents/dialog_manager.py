import time
import asyncio
from json import load
import queue
import wave
from os import environ
from os.path import abspath, join
from pathlib import Path
from threading import Thread
from time import sleep, strftime

import numpy as np
from sic_framework.core import sic_logging
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.core.sic_application import SICApplication
from sic_framework.devices import Pepper
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.desktop import Desktop
from sic_framework.devices.device import SICDeviceManager
from sic_framework.services.dialogflow.dialogflow import (
    Dialogflow,
    DialogflowConf,
    GetIntentRequest,
)
from sic_framework.services.google_tts.google_tts import (
    GetSpeechRequest,
    Text2Speech,
    Text2SpeechConf,
)
from sic_framework.services.llm.openai_gpt import GPT
from sic_framework.services.llm import GPTConf, GPTRequest
from dotenv import load_dotenv

from agents.tts_manager import NaoqiTTSConf, TTSConf, GoogleTTSConf, TTSCacher

load_dotenv("../config/.env")


class InteractionConf:

    def __init__(self, speaking_rate=None, sleep_time=0, animated=True, max_attempts=2, amplified=False,
                 always_regenerate=False):
        self.speaking_rate = speaking_rate
        self.sleep_time = sleep_time
        self.animated = animated
        self.max_attempts = max_attempts
        self.amplified = amplified
        self.always_regenerate = always_regenerate

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
                 tts_conf: TTSConf = NaoqiTTSConf, env_path=None, microphone_device=None):

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

        print('complete')

        print("\n SETTING UP OPENAI")
        # Generate your personal openai api key here: https://platform.openai.com/api-keys
        # Either add your openai key to your systems variables (and comment the next line out) or
        # create a .openai_env file in the conf/openai folder and add your key there like this:
        # OPENAI_API_KEY="your key"
        if env_path:
            load_dotenv(env_path)

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
        if isinstance(self.tts_conf, GoogleTTSConf):
            # setup the tts service
            self.tts = Text2Speech(conf=Text2SpeechConf(keyfile_json=dialogflow_conf.keyfile_json,
                                                        speaking_rate=self.tts_conf.speaking_rate))
            init_reply = self.tts.request(GetSpeechRequest(text="Ik ben aan het initializeren",
                                                           voice_name=self.tts_conf.google_tts_voice_name,
                                                           ssml_gender=self.tts_conf.google_tts_voice_gender))
            self.sample_rate = init_reply.sample_rate
            print('Google TTS activated')
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

        print("\n SETTING UP DIALOGFLOW")
        # initiate Dialogflow object
        self.dialogflow = Dialogflow(ip="localhost", conf=dialogflow_conf, input_source=self.mic)
        # flag to signal when the app should listen (i.e. transmit to dialogflow)
        self.request_id = np.random.randint(10000)
        self.dialogflow.register_callback(self._on_dialog)
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

    def google_say(self, text, speaking_rate=None, sleep_time=None, animated=None, amplified=False,
                   always_regenerate=False):
        # Generate cache key and load cached speech audio if available.
        tts_key = self.tts_cacher.make_tts_key(text, self.tts_conf)
        audio_file = self.tts_cacher.load_audio_file(tts_key)

        # If requested and available play cached speech audio
        if not always_regenerate and audio_file:
            self.log_utterance(speaker='robot', text=f'{text} (cache)')
            self.play_audio(audio_file, log=False)
        else:  # Else generate new speech audio
            reply = self.tts.request(GetSpeechRequest(
                text=text,
                voice_name=self.tts_conf.google_tts_voice_name,
                ssml_gender=self.tts_conf.google_tts_voice_gender,
                speaking_rate=speaking_rate or self.tts_conf.speaking_rate
            ))
            audio_bytes = reply.waveform
            sample_rate = reply.sample_rate

            # Amplify audio if needed
            if audio_bytes and amplified:
                audio_bytes = self._amplify_audio(audio_bytes)

            # Play audio
            self.speaker.request(AudioRequest(audio_bytes, sample_rate))
            self.log_utterance(speaker='robot', text=text)

            # Save to cache file
            self.tts_cacher.save_audio_file(tts_key, audio_bytes, sample_rate)

        # Sleep if requested
        if sleep_time and sleep_time > 0:
            sleep(sleep_time)

    def naoqi_say(self, text, sleep_time=None, animated=False):
        self.device_manager.tts.request(
            NaoqiTextToSpeechRequest(text, animated=animated, language='English'))

        # Sleep if requested
        if sleep_time and sleep_time > 0:
            sleep(sleep_time)

    @InteractionConf.apply_config_defaults('interaction_conf', ['speaking_rate', 'sleep_time', 'animated', 'amplified',
                                                                'always_regenerate'])
    def say(self, text, speaking_rate=None, sleep_time=None, animated=None, amplified=False, always_regenerate=False):
        if isinstance(self.tts_conf, NaoqiTTSConf):
            self.naoqi_say(text, sleep_time=sleep_time, animated=animated)
        elif isinstance(self.tts_conf, GoogleTTSConf):
            self.google_say(text, speaking_rate=speaking_rate, sleep_time=sleep_time, animated=animated,
                            amplified=amplified, always_regenerate=always_regenerate)
        else:
            raise ValueError(f'Unsupported tts_conf type: {type(self.tts_conf)}')

    def listen(self):
        return input("Listening, enter response")
        # try:
        #     reply = self.dialogflow.request(GetIntentRequest(self.request_id), timeout=10)
        #     if reply.response.query_result.query_text:
        #         return reply.response.query_result.query_text
        #     return None
        # except TimeoutError as e:
        #     print("Error:", e)

    def _on_dialog(self, message):
        if message.response:
            transcript = message.response.recognition_result.transcript
            print("Transcript:", transcript)
            if message.response.recognition_result.is_final:
                self.log_utterance(speaker='child', text=transcript)

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
