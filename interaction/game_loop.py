import time

from sic_framework.devices.desktop import Desktop

from agents.guesser import Guesser
from interaction.audio_pipeline import AudioPipeline
from interaction.game_state import GameState
from interaction.turn_manager import TurnManager
from interaction.utils import parse_clue


class GameLoop:
    def __init__(self, guesser: Guesser, game_state: GameState, max_turns=5):
        self.guesser = guesser
        self.game_state = game_state
        self.max_turns = max_turns
        self.turn_manager = TurnManager(guesser, game_state)

    def play(self):
        # TODO: introduce robot
        self.guesser.say_random_start_game()
        self.guesser.say("I'm waiting for you to place the red cards, let me know when you're ready.", sleep_time=0.5)

        # Wait for the spymaster to place the initial red cards
        response = self.guesser.listen()
        red_cards_placed = self.game_state.are_initial_red_cards_placed()
        while not response or response == "" or "ready" not in response or not red_cards_placed:
            print("Spymaster not ready yet, waiting...")
            response = self.guesser.listen()
            red_cards_placed = self.game_state.are_initial_red_cards_placed()

        # Start recording for the first turn after initial red cards are placed
        self.guesser.start_recording()

        while not self.game_state.game_over and self.game_state.turn < self.max_turns:
            print(f"Playing Turn {self.game_state.turn}")
            self.guesser.pause_recording()
            self.guesser.say_random_human_turn()
            self.guesser.resume_recording()

            clue_word, num = self.receive_clue()

            # Stop recording immediately after the clue is received and classify
            features, confidence_level = self.guesser.stop_and_process_audio(clue_word, self.game_state.turn)

            # In non-adaptive mode the confidence is still logged but not used
            # to adjust the robot's verbal behavior.
            adaptive_confidence = confidence_level if self.guesser.is_adaptive() else None
            adaptive_features = features if self.guesser.is_adaptive() else None
            self.turn_manager.play_turn(clue_word, num, adaptive_confidence, adaptive_features)

            if not self.game_state.game_over:
                self.guesser.say("Go ahead, place a red card.")
                input("Press enter after red card is placed.")
                # Start recording for the next turn after the red card is placed
                self.guesser.start_recording()

        if not self.game_state.game_over:
            self.guesser.say_random_game_over()

        if self.game_state.win is True:
            self.guesser.say_random_win_reaction()
        elif self.game_state.win is False:
            self.guesser.say_random_loss_reaction()

        self.guesser.stop_recording_if_active()

    def receive_clue(self) -> tuple[str, int]:
        while True:
            # --- Listen until we get some input ---
            raw_clue = self.guesser.listen()
            while not raw_clue:
                print("No input detected from listener; listening again")
                raw_clue = self.guesser.listen()

            # --- Try to parse the clue ---
            try:
                clue_word, num = parse_clue(raw_clue)
            except Exception:
                # Pause recording while the robot responds, then resume so the
                # next clue attempt is captured.
                self.guesser.pause_recording()
                self.guesser.say_random_clue_not_understood()
                self.guesser.resume_recording()
                continue  # restart from listening

            # --- Pause recording before confirmation: the verification exchange
            #     (robot repeating the clue, user saying yes/no, robot asking to
            #     repeat) should not be included in the confidence analysis. ---
            self.guesser.pause_recording()
            self.guesser.say_random_repeat_clue(clue_word, num)
            self.guesser.say_verify_received_clue()

            feedback = self.guesser.listen()
            if self.is_clue_well_received(feedback):
                # Recording stays paused; the caller will stop and process it.
                return clue_word, num

            # --- Not confirmed → ask to repeat and resume recording for the
            #     next clue attempt. ---
            self.guesser.say("Oh, could you repeat the clue?")
            self.guesser.resume_recording()

    @staticmethod
    def is_clue_well_received(feedback: str) -> bool:
        """
        Returns True only if the spymaster clearly confirms
        the clue was understood correctly.
        """
        if not feedback:
            return False

        t = feedback.lower().strip()

        # Strong acceptance signals
        accept_phrases = [
            "yes", "yeah", "yep", "correct", "right", "exactly",
            "that's right", "you got it", "perfect", "ok", "okay",
            "sounds good"
        ]

        # Strong rejection / correction signals
        reject_phrases = [
            "no", "nope", "not", "wrong", "incorrect",
            "repeat", "again", "say it again",
            "didn't", "did not", "don't", "do not",
            "wait", "hold on", "sorry"
        ]

        # If rejection appears anywhere → False
        if any(p in t for p in reject_phrases):
            return False

        # Clear acceptance → True
        if any(p in t for p in accept_phrases):
            return True

        # Everything else is treated as not well received
        return False
