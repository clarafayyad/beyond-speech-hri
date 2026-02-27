import os.path
import random
import sys
from json import load
from PIL import Image

from sic_framework.devices import Pepper
from sic_framework.devices.naoqi_shared import Naoqi
from sic_framework.services.dialogflow import DialogflowConf

from agents.dialog_manager import DialogManager
from agents.llm_agent import LLMAgent
from agents.pepper_tablet.display_service import PepperTabletDisplayService
from agents.stt_manager import RealTimeSTTService


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
                             tts_conf=tts_conf)

    def prompt_llm(self, system_prompt: str, user_prompt: str) -> dict:
        return self.llm_agent.prompt_llm(system_prompt, user_prompt)

    def say(self, text, sleep_time=0):
        self.dialog_manager.say(text, always_regenerate=True, sleep_time=sleep_time)

    def listen(self) -> str:
        return self.dialog_manager.listen()

    def say_random_red_reaction(self):
        reactions = [
            "Oh— uh… no. That’s red. Yeah… turn’s over.",
            "Hmm. Nope. Red card. That didn’t work.",
            "Ah… okay, that one’s red. Ending the turn.",
            "Oof. That’s a red card. My mistake.",
            "Yeah… no. Red team. Turn ends."
        ]
        self.say(random.choice(reactions))

    def say_random_blue_reaction(self):
        reactions = [
            "Oh! Yes— that’s blue!",
            "Nice… yeah, blue card. That worked.",
            "Mm-hmm! That one’s ours. Blue.",
            "Hey, okay! Blue agent found.",
            "Yes! Blue card. Good clue."
        ]
        self.say(random.choice(reactions))

    def say_random_neutral_reaction(self):
        reactions = [
            "Uh… okay. That’s neutral. Just a bystander.",
            "Hmm… no agent there. Neutral card.",
            "Alright… innocent bystander. Moving on.",
            "Yeah… neutral. Nothing happens.",
            "Okay. That one’s neutral. Continuing."
        ]
        self.say(random.choice(reactions))

    def say_random_assassin_reaction(self):
        reactions = [
            "Oh— oh no… that’s the assassin. We lost.",
            "Uh… yeah. That’s bad. Assassin card. Game over.",
            "Oh wow… okay, that was the assassin. We lose.",
            "Ah… no. Assassin. That ends the game.",
            "Mm… yeah. I found the assassin. We lost."
        ]
        self.say(random.choice(reactions))

    def say_random_start_game(self):
        reactions = [
            "Alright! Let’s fire up our brains and start Codenames!",
            "Game on! I’m ready when you are.",
            "Welcome to Codenames. May our clues be clever!",
            "Let the guessing begin! I promise to think very hard.",
            "Okay team, activating game mode. Beep boop.",
            "New game starting! I have a good feeling about this one."
        ]
        self.say(random.choice(reactions))

    def say_random_human_turn(self):
        reactions = [
            "It’s your turn! Please give me a clue.",
            "Your move, spymaster. I’m listening.",
            "I’m ready for your clue. Impress me!",
            "Clue time! What do you have for me?",
            "All ears! What’s the clue?",
            "Go ahead, give me a clever hint."
        ]
        self.say(random.choice(reactions))

    def say_random_clue_not_understood(self):
        reactions = [
            "Hmm… I didn’t quite get that. Could you try again?",
            "My circuits are confused. Please repeat the clue.",
            "Sorry, I didn’t understand that clue. One more time?",
            "That clue puzzled me. Can you rephrase it?",
            "I think something went wrong in my brain. Please try again.",
            "Oops! I didn’t catch that. Another attempt, please."
        ]
        self.say(random.choice(reactions))

    def say_random_repeat_clue(self, clue, guesses):
        reactions = [
            f"Alright! The clue is {clue}, and I can guess {guesses} times.",
            f"Got it! Clue {clue}, with {guesses} guesses allowed.",
            f"I will guess based on {clue}. I have {guesses} guesses.",
            f"You said {clue}, so I get {guesses} chances. Let’s go!",
            f"Clue received: {clue}. Number of guesses: {guesses}.",
            f"Okay, {clue} it is! I may guess {guesses} times."
        ]
        self.say(random.choice(reactions))

    def say_random_game_over(self):
        reactions = [
            "That’s the end of the game!",
            "Game over! Well played.",
            "And… that concludes our game.",
            "The mission is complete. Game over.",
            "No more guesses! The game has ended.",
            "That’s it! Codenames finished."
        ]
        self.say(random.choice(reactions))

    def say_random_win_reaction(self):
        reactions = [
            "We won! Great teamwork!",
            "Victory! Excellent clues and guesses!",
            "Nice job! We make a great team.",
            "Yes! We did it!",
            "Mission successful. Well played!",
            "We won the game! High five! …Emotionally."
        ]
        self.say(random.choice(reactions))

    def say_random_loss_reaction(self):
        reactions = [
            "Ah… we lost this one. Still a good try!",
            "That’s a loss, but we played well.",
            "Oops! The other team got us this time.",
            "Sorry, we lost. Let’s do better next round!",
            "Defeat detected. But I had fun!",
            "We didn’t win, but I enjoyed playing with you."
        ]
        self.say(random.choice(reactions))

    def say_random_thinking(self):
        reactions = [
            "Hmm… let me think.",
            "Processing… please wait.",
            "Thinking very hard right now.",
            "Analyzing the board…",
            "My brain is working at maximum capacity.",
            "Give me a moment to calculate.",
            "Beep boop… thinking.",
            "This requires deep thought."
        ]
        self.say(random.choice(reactions))

    def say_random_guess(self, card_idx):
        reactions = [
            f"Hmm… I think I'll pick {card_idx}.",
            f"My guess is {card_idx}. Fingers crossed!",
            f"I choose {card_idx}! Let’s see how I did.",
            f"Alright, {card_idx} feels right to me.",
            f"Going with {card_idx} as my guess.",
            f"I have a good feeling about {card_idx}, so I pick it."
        ]
        self.say(random.choice(reactions))

    def display_guess(self, file_path):
        if isinstance(self.device_manager, Pepper):
            self.display_service.show_image(file_path)
            return

        file_path = os.path.join("../assets/cards/", file_path)
        img = Image.open(file_path)
        img.show()

    def clear_display(self):
        if not isinstance(self.device_manager, Pepper):
            return
        self.display_service.clear_display()

    def shutdown(self):
        print("🛑 Shutting down STT...")
        if not isinstance(self.dialog_manager.stt_service, RealTimeSTTService):
            return
        try:
            self.dialog_manager.stt_service.recorder.stop()
        except Exception:
            pass
        sys.exit(0)

