from json import load
from os.path import abspath, join

from sic_framework.devices.naoqi_shared import Naoqi
from sic_framework.services.dialogflow import DialogflowConf

from agents.dialog_manager import DialogManager
from agents.llm_agent import LLMAgent


class ConversationalAgent:
    def __init__(self, device_manager, tts_conf):
        self.dialog_manager = self.build_dialog_manager(device_manager, tts_conf)
        self.llm_agent = LLMAgent()

    @staticmethod
    def build_dialog_manager(device_manager, tts_conf):
        if isinstance(device_manager, Naoqi):
            sample_rate_hertz = 16000
        else:
            sample_rate_hertz = 44100

        dialogflow_conf = DialogflowConf(
            keyfile_json=load(open("google-key.json")),
            sample_rate_hertz=sample_rate_hertz,
            language="en",
            timeout=7)

        return DialogManager(device_manager=device_manager,
                                            dialogflow_conf=dialogflow_conf,
                                            tts_conf=tts_conf,
                                            env_path=abspath(join("config", "env", ".env")))

    def prompt_llm(self, system_prompt: str, user_prompt: str) -> dict:
        return self.llm_agent.prompt_llm(system_prompt, user_prompt)

    def say(self, text):
        self.dialog_manager.say(text)

    def listen(self) -> str:
        return self.dialog_manager.listen()
