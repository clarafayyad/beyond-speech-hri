import numpy as np
from sic_framework.services.dialogflow import Dialogflow, GetIntentRequest

from RealtimeSTT.RealtimeSTT import AudioToTextRecorder
from abc import ABC, abstractmethod


class STTService(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def listen(self) -> str:
        """
        Blocks until speech is detected and finalized.
        Returns recognized text.
        """
        pass


class RealTimeSTTService(STTService):
    def __init__(self, mic_index):
        super().__init__()
        self.recorder = AudioToTextRecorder(
            input_device_index=mic_index,
            model="small.en",
            language="en",
            realtime_processing_pause=0.1,
            post_speech_silence_duration=2,
            use_microphone=True,
            enable_realtime_transcription=False,  # final result only
            beam_size=5,
            initial_prompt=(
                "This is a board game called Codenames. "
                "Valid phrases include colors and numbers like "
                "'animal four', 'glass two', 'ready'..."
            )
        )

    def listen(self) -> str:
        """
        Blocks until speech is detected and transcription is finalized.
        Returns recognized text (lowercased).
        """
        text = self.recorder.text()
        return text.strip().lower() if text else ""


class DialogFlowSTTService(STTService):
    def __init__(self, mic_index, dialogflow_conf):
        super().__init__()
        self.dialogflow = Dialogflow(ip="localhost", conf=dialogflow_conf, input_source=mic_index)
        self.request_id = np.random.randint(10000)

    def listen(self) -> str:
        try:
            reply = self.dialogflow.request(GetIntentRequest(self.request_id), timeout=10)
            if reply.response.query_result.query_text:
                return reply.response.query_result.query_text
            return ""
        except TimeoutError as e:
            print("Error:", e)
