import random
from json import load
from os.path import abspath, join

from sic_framework.devices import Pepper
from sic_framework.devices.naoqi_shared import Naoqi
from sic_framework.services.dialogflow import DialogflowConf

from agents.dialog_manager import DialogManager
from agents.llm_agent import LLMAgent
from agents.pepper_tablet.display_service import PepperTabletDisplayService


class Guesser:
    def __init__(self, device_manager, tts_conf):
        self.device_manager = device_manager
        self.dialog_manager = self.build_dialog_manager(device_manager, tts_conf)
        self.llm_agent = LLMAgent()

        if isinstance(self.device_manager, Pepper):
            self.display_service = PepperTabletDisplayService(pepper=device_manager)


    @staticmethod
    def build_dialog_manager(device_manager, tts_conf):
        if isinstance(device_manager, Naoqi):
            sample_rate_hertz = 16000
        else:
            sample_rate_hertz = 44100

        dialogflow_conf = DialogflowConf(
            keyfile_json=load(open("../config/google/google-key.json")),
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
        self.dialog_manager.say(text, always_regenerate=True)

    def listen(self) -> str:
        return self.dialog_manager.listen()

    def random_red_reaction(self):
        reactions = [
            "Oh— uh… no. That’s red. Yeah… turn’s over.",
            "Hmm. Nope. Red card. That didn’t work.",
            "Ah… okay, that one’s red. Ending the turn.",
            "Oof. That’s a red card. My mistake.",
            "Yeah… no. Red team. Turn ends."
        ]
        self.say(random.choice(reactions))

    def random_blue_reaction(self):
        reactions = [
            "Oh! Yes— that’s blue!",
            "Nice… yeah, blue card. That worked.",
            "Mm-hmm! That one’s ours. Blue.",
            "Hey, okay! Blue agent found.",
            "Yes! Blue card. Good clue."
        ]
        self.say(random.choice(reactions))

    def random_neutral_reaction(self):
        reactions = [
            "Uh… okay. That’s neutral. Just a bystander.",
            "Hmm… no agent there. Neutral card.",
            "Alright… innocent bystander. Moving on.",
            "Yeah… neutral. Nothing happens.",
            "Okay. That one’s neutral. Continuing."
        ]
        self.say(random.choice(reactions))

    def random_assassin_reaction(self):
        reactions = [
            "Oh— oh no… that’s the assassin. We lost.",
            "Uh… yeah. That’s bad. Assassin card. Game over.",
            "Oh wow… okay, that was the assassin. We lose.",
            "Ah… no. Assassin. That ends the game.",
            "Mm… yeah. I found the assassin. We lost."
        ]
        self.say(random.choice(reactions))

    def display_guess(self, file_path):
        if not isinstance(self.device_manager, Pepper):
            return
        self.display_service.show_image(file_path)
