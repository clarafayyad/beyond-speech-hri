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

from interaction.continuity import get_baseline_continuity_utterance, get_adaptive_continuity_utterance
from multimodal_perception.model.confidence_classifier import CONFIDENCE_LOW, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

LOW_DURATION_DEV_THRESHOLD = 1.0
LOW_HESITATION_DEV_THRESHOLD = 0.5
LOW_PAUSE_DEV_THRESHOLD = 0.8
LOW_SPEECH_RATE_DEV_THRESHOLD = -0.8
HIGH_HESITATION_DEV_THRESHOLD = -0.8
HIGH_DURATION_DEV_THRESHOLD = -0.8
HIGH_SPEECH_RATE_DEV_THRESHOLD = 0.8


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

    def say(self, text, sleep_time=0, always_regenerate=False):
        self.dialog_manager.say(text, always_regenerate=always_regenerate, sleep_time=sleep_time)
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

    def get_continuity_remark(self, game_state, confidence_level=None, adaptive=None):
        """Return a context-aware remark referencing previous turn performance,
        or ``None`` when nothing should be said (e.g. first turn).

        Parameters
        ----------
        game_state : GameState
            Current game state; must expose ``turn``, ``history``, and
            ``confidence_history`` attributes.
        confidence_level : str | None
            Backward-compatible signal used to decide adaptive mode when
            *adaptive* is not explicitly provided.
        adaptive : bool | None
            When ``True`` an adaptive remark is generated; when ``False`` a
            baseline remark is used.  When ``None``, behavior falls back to
            checking whether *confidence_level* is ``None``.
        """
        if adaptive is None:
            adaptive = confidence_level is not None

        if adaptive:
            return get_adaptive_continuity_utterance(game_state, confidence_level)
        return get_baseline_continuity_utterance(game_state)

    def say_continuity_remark(self, game_state, confidence_level=None):
        """Utter a context-aware remark referencing previous turn performance.

        In adaptive mode (*confidence_level* is not ``None``) the remark also
        considers recent confidence trends.  In baseline mode it references
        prior performance in a general way.  Does nothing on the first turn.
        """
        utterance = self.get_continuity_remark(game_state, confidence_level)
        if utterance:
            self.say(utterance)

    @staticmethod
    def _feature_comment(features: dict, confidence_level: str) -> str:
        """Return a short, light-hearted phrase referencing the audio feature
        that most likely drove the confidence inference, or an empty string
        when no single feature stands out.

        Only the features that are intuitively explainable in natural language
        are used: duration, verbal hesitation count, maximum pause length, and
        speech rate.  When participant-relative deviation features (``*_dev``)
        are available, they are preferred so comments only trigger for
        participant-salient signals.  The most salient signal is reported first
        (in priority order) to keep the utterance concise.
        """
        if not features:
            return ""

        duration = features.get('duration') or 0
        pause_max = features.get('pause_max') or 0
        hesitation_count = features.get('verbal_hesitation_count') or 0
        speech_rate = features.get('speech_rate') or 0
        duration_dev = features.get('duration_dev')
        pause_max_dev = features.get('pause_max_dev')
        hesitation_count_dev = features.get('verbal_hesitation_count_dev')
        speech_rate_dev = features.get('speech_rate_dev')
        has_participant_relative_deviations = any(
            value is not None
            for value in (duration_dev, pause_max_dev, hesitation_count_dev, speech_rate_dev)
        )

        def _dev_value(metric):
            return metric if metric is not None else 0

        if confidence_level == CONFIDENCE_LOW:
            if has_participant_relative_deviations:
                if _dev_value(duration_dev) > LOW_DURATION_DEV_THRESHOLD:
                    return random.choice([
                        "Whoa, that clue took a while to arrive!",
                        "Looks like that one needed some thought!",
                        "That was quite the thinking session!",
                    ])
                if _dev_value(hesitation_count_dev) > LOW_HESITATION_DEV_THRESHOLD:
                    return random.choice([
                        "I caught a few 'um's and 'uh's in there!",
                        "Sounds like the clue was still brewing!",
                        "A couple of hesitations — no worries, I'm on it!",
                    ])
                if _dev_value(pause_max_dev) > LOW_PAUSE_DEV_THRESHOLD:
                    return random.choice([
                        "I noticed a little pause in there!",
                        "There was a moment of mystery in that silence!",
                        "A dramatic pause — love it, but let's be careful!",
                    ])
                if _dev_value(speech_rate_dev) < LOW_SPEECH_RATE_DEV_THRESHOLD:
                    return random.choice([
                        "You took it nice and slow!",
                        "Careful and measured — I respect that!",
                        "Slow and steady clue incoming!",
                    ])
                return ""

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
            if has_participant_relative_deviations and _dev_value(hesitation_count_dev) < HIGH_HESITATION_DEV_THRESHOLD:
                return random.choice([
                    "Not a single hesitation — I like it!",
                    "Clean delivery, no fillers!",
                    "You knew exactly what to say!",
                ])
            if has_participant_relative_deviations and _dev_value(duration_dev) < HIGH_DURATION_DEV_THRESHOLD:
                return random.choice([
                    "That was quick and decisive!",
                    "Straight to the point!",
                    "Quick and clear!",
                ])
            if has_participant_relative_deviations and _dev_value(speech_rate_dev) > HIGH_SPEECH_RATE_DEV_THRESHOLD:
                return random.choice([
                    "You rattled that right off!",
                    "Fast and sure — I love it!",
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

    def get_confidence_level_reaction(self, confidence_level, features=None):
        """Return the text of a confidence-level reaction, or ``None``.

        Parameters
        ----------
        confidence_level : str | None
            One of ``CONFIDENCE_LOW``, ``CONFIDENCE_MEDIUM``,
            ``CONFIDENCE_HIGH``, or ``None``.  Returns ``None`` when the
            level is ``None`` or unrecognised.
        features : dict | None
            Optional audio-feature dict.  When a notable feature is found,
            a feature-grounded comment replaces the generic reaction.
        """
        comment = Guesser._feature_comment(features, confidence_level) if features else ""

        if comment:
            return comment

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
                "Alright! That was very clear!",
                "Nice! That sounded super certain!",
                "Okay! Strong signal, I like it!",
                "Got it! That felt really confident!"
            ]

        return random.choice(reactions) if reactions else None

    def say_confidence_level_reaction(self, confidence_level, features=None):
        text = self.get_confidence_level_reaction(confidence_level, features)
        if text:
            self.say(text)

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
            "We won! Great teamwork! See you next time!",
            "Victory! Excellent clues and guesses! Catch you later!",
            "Nice job! We make a great team. Until next time!",
            "Yes! We did it! Talk soon!",
            "Mission successful. Well played! Goodbye for now!",
            "We won the game! High five! …Emotionally. See you later!"
        ]
        self.say(random.choice(reactions))

    def say_random_loss_reaction(self):
        reactions = [
            "Ah… we lost this one. Still a good try! See you next time!",
            "That’s a loss, but we played well. Catch you later!",
            "Oops! The other team got us this time. Until next time!",
            "Sorry, we lost. Let’s do better next round! Talk soon!",
            "Defeat detected. But I had fun! Goodbye for now!",
            "We didn’t win, but I enjoyed playing with you. See you later!"
        ]
        self.say(random.choice(reactions))

    def get_random_thinking(self):
        """Return the text of a random thinking utterance.

        Always returns a non-empty str selected from a fixed pool of
        filler phrases.
        """
        reactions = [
            "Hmm.",
            "Hmm…",
            "Hmm, let's see.",
            "Let's see…",
            "Okay…",
            "Alright…",
            "One moment…",
            "Hmm, let me think.",
            "Just thinking…",
            "Thinking…",
        ]
        return random.choice(reactions)

    def say_random_thinking(self):
        self.say(self.get_random_thinking())

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
        self.dialog_manager.shutdown_logging()
        if not isinstance(self.dialog_manager.stt_service, RealTimeSTTService):
            return
        try:
            self.dialog_manager.stt_service.recorder.shutdown()
        except Exception:
            pass
        sys.exit(0)
