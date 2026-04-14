import os.path
import random
import sys
import time
from json import load
from PIL import Image

from sic_framework.devices import Pepper
from sic_framework.devices.desktop import Desktop
from sic_framework.devices.naoqi_shared import Naoqi
from sic_framework.services.dialogflow import DialogflowConf

from agents.dialog_manager import DialogManager
from agents.llm_agent import LLMAgent
from agents.pepper_tablet.display_service import PepperTabletDisplayService
from agents.stt_manager import RealTimeSTTService
from interaction.audio_pipeline import AudioPipeline

from multimodal_perception.model.confidence_classifier import CONFIDENCE_LOW, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM


class Guesser:
    def __init__(self, device_manager, tts_conf, interaction_conf=None):
        self.device_manager = device_manager
        self.dialog_manager = self.build_dialog_manager(device_manager, tts_conf, interaction_conf)
        self.llm_agent = LLMAgent()

        if isinstance(self.device_manager, Pepper):
            self.display_service = PepperTabletDisplayService(pepper=device_manager)

        self.audio_pipeline = (
            AudioPipeline(interaction_conf.participant_id, interaction_conf.external_audio_device_id)
            if interaction_conf.participant_id is not None
            else None
        )

    @staticmethod
    def build_dialog_manager(device_manager, tts_conf, interaction_conf):
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
                             interaction_conf=interaction_conf)

    def prompt_llm(self, system_prompt: str, user_prompt: str) -> dict:
        return self.llm_agent.prompt_llm(system_prompt, user_prompt)

    def say(self, text, sleep_time=0):
        self.dialog_manager.say(text, always_regenerate=True, sleep_time=sleep_time)
        if isinstance(self.dialog_manager.device_manager, Desktop):
            time.sleep(2)  # To avoid hearing its own speech as feedback

    def listen(self) -> str:
        return self.dialog_manager.listen()

    def start_recording(self):
        if self.audio_pipeline:
            self.audio_pipeline.start_recording()

    def pause_recording(self):
        """Pause the audio recording (e.g., while the robot is speaking)."""
        if self.audio_pipeline:
            self.audio_pipeline.pause_recording()

    def resume_recording(self):
        """Resume audio recording after the robot has finished speaking."""
        if self.audio_pipeline:
            self.audio_pipeline.resume_recording()

    def stop_and_process_audio(self, clue_word, turn_number):
        if self.audio_pipeline:
            return self.audio_pipeline.stop_and_process(clue_word, turn_number)
        return None, None

    def stop_recording_if_active(self):
        if self.audio_pipeline:
            self.audio_pipeline.stop_recording_if_active()

    def is_adaptive(self):
        return self.dialog_manager.interaction_conf.adaptive

    @staticmethod
    def _feature_comment(features: dict, confidence_level: str) -> str:
        """Return a short, light-hearted phrase referencing the audio feature
        that most likely drove the confidence inference, or an empty string
        when no single feature stands out.

        Only the features that are intuitively explainable in natural language
        are used: duration, verbal hesitation count, maximum pause length, and
        speech rate.  The most salient signal is reported first (in priority
        order) to keep the utterance concise.
        """
        if not features:
            return ""

        duration = features.get('duration') or 0
        pause_max = features.get('pause_max') or 0
        hesitation_count = features.get('verbal_hesitation_count') or 0
        speech_rate = features.get('speech_rate') or 0

        if confidence_level == CONFIDENCE_LOW:
            if duration > 12:
                return random.choice([
                    "Whoa, that clue took a while to arrive!",
                    "Looks like that one needed some thought!",
                    "That was quite the thinking session!",
                ])
            if hesitation_count >= 2:
                return random.choice([
                    "I caught a few 'um's and 'uh's in there!",
                    "Sounds like the clue was still brewing!",
                    "A couple of hesitations — no worries, I'm on it!",
                ])
            if pause_max > 2.5:
                return random.choice([
                    "I noticed a little pause in there!",
                    "There was a moment of mystery in that silence!",
                    "A dramatic pause — love it, but let's be careful!",
                ])
            if 0 < speech_rate < 1.5:
                return random.choice([
                    "You took it nice and slow!",
                    "Careful and measured — I respect that!",
                    "Slow and steady clue incoming!",
                ])

        elif confidence_level == CONFIDENCE_HIGH:
            if duration > 0 and hesitation_count == 0:
                return random.choice([
                    "Not a single hesitation — I like it!",
                    "Clean delivery, no fillers!",
                    "You knew exactly what to say!",
                ])
            if 0 < duration < 4:
                return random.choice([
                    "That was quick and decisive!",
                    "Straight to the point!",
                    "Quick and clear!",
                ])
            if speech_rate > 3.5:
                return random.choice([
                    "You rattled that right off!",
                    "Fast and sure — I love it!",
                ])

        return ""

    def say_confidence_level_reaction(self, confidence_level, features=None):
        comment = Guesser._feature_comment(features, confidence_level) if features else ""

        # When a feature-grounded comment is available, use it as the full
        # reaction so the utterance stays natural and non-redundant.
        if comment:
            self.say(comment)
            return

        reactions = []

        if confidence_level == CONFIDENCE_LOW:
            reactions = [
                "Hmm… you don't sound very sure.",
                "Okay… I'll be careful with this one.",
                "That sounded a bit uncertain… let's think.",
                "Alright… not super confident, I hear you.",
                "Hmm… I might need to play this safe."
            ]
        elif confidence_level == CONFIDENCE_MEDIUM:
            reactions = [
                "Okay, I think I get what you mean.",
                "Alright, that sounds reasonable.",
                "Hmm, I've got a rough idea.",
                "Okay… let's try this.",
                "Got it. I'll go with that."
            ]
        elif confidence_level == CONFIDENCE_HIGH:
            reactions = [
                "Oh, you sound confident. I like that.",
                "Alright! That was clear.",
                "Nice, that sounded very certain.",
                "Okay, I'm feeling good about this.",
                "Got it — strong signal."
            ]

        if reactions:
            self.say(random.choice(reactions))

    def say_memory_reference(self, last_turn_memory):
        """Say a natural sentence referencing the previous round when relevant."""
        if not last_turn_memory:
            return

        if last_turn_memory.get("had_correct_guess"):
            phrases = [
                "That last one was tricky too—we handled it well.",
                "The previous round was uncertain as well, but we got through it.",
                "Hmm, same uncertainty as before. We managed then—let's stay careful.",
                "We navigated that last tough clue together. Let's do it again.",
            ]
        else:
            phrases = [
                "Last round was tough too. Let's try to do better this time.",
                "Hmm, uncertain again. The last one was difficult too—let's think carefully.",
                "That previous clue gave us trouble as well. Staying cautious.",
                "Same tricky situation as before. Let's be extra careful.",
            ]
        self.say(random.choice(phrases))

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

    def say_verify_received_clue(self):
        reactions = [
            "Did I get the clue right?",
            "Is that the clue you meant?",
            "Just checking—did I hear the clue correctly?",
            "Let me confirm: is that the clue?",
            "Did I understand the clue properly?",
            "Is that correct?",
            "Got it… or did I?",
            "Did I catch the clue right?",
            "Am I interpreting the clue correctly?",
            "Just to be sure, is that the clue?"
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
            "Hmm… okay, give me a second.",
            "Alright… thinking… thinking…",
            "Let me just… pretend I know what I’m doing.",
            "Hmm. This is harder than it looks.",
            "Okay… big brain moment incoming.",
            "Wait, wait… I almost have it.",
            "Let me think this through before I embarrass myself.",
            "Okay… analyzing… but like, casually.",
            "Hmm… don’t rush me, I’m being smart.",
            "I’m thinking. It’s subtle, but it’s happening.",
            "Alright… calculating my chances of being wrong.",
            "Hmm… this could go very well or very badly.",
            "Thinking… with style.",
            "Okay… activating strategic mode.",
            "Hmm… I feel like I should know this.",
        ]
        self.say(random.choice(reactions))

    def say_random_guess(self):
        reactions = [
            "I'm considering this card.",
            "Here is the card I'm thinking about.",
            "This one looks promising.",
            "I think this might be the one.",
            "Take a look at this card.",
            "This is my current guess.",
            "I'm leaning toward this card.",
            "This card matches the clue the best.",
            "This might be the correct one.",
            "I'm pointing to this card.",
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
            self.dialog_manager.stt_service.recorder.shutdown()
        except Exception:
            pass
        sys.exit(0)

