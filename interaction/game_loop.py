import time

from sic_framework.devices.desktop import Desktop

from agents.guesser import Guesser
from interaction.audio_pipeline import AudioPipeline
from interaction.experiment_logger import ExperimentLogger
from interaction.game_state import GameState
from interaction.turn_manager import TurnManager
from interaction.utils import parse_clue
from multimodal_perception.audio.verbal_hesitation import FILLERS, contains_hesitation_trigger
from multimodal_perception.model.confidence_classifier import CONFIDENCE_MEDIUM

# Number of silent retries before asking the user to repeat
GRACE_PERIOD_RETRIES = 2
# Seconds to wait between silent grace-period retries
GRACE_PERIOD_WAIT_SECONDS = 1.0
# Seconds of silence before the robot says a long-wait utterance (adaptive only)
LONG_WAIT_THRESHOLD_SECONDS = 20
# Maximum number of long-wait reactions per clue-reception attempt
MAX_LONG_WAIT_REACTIONS = 2


class GameLoop:
    def __init__(self, guesser: Guesser, game_state: GameState, max_turns=5,
                 participant_id=None, is_adaptive=False, board=None, key_map=None):
        self.guesser = guesser
        self.game_state = game_state
        self.max_turns = max_turns
        self.turn_manager = TurnManager(guesser, game_state)

        pid = participant_id or ""
 
        self.experiment_logger = ExperimentLogger(
            participant_id=pid,
            is_adaptive=is_adaptive,
            board=board if board is not None else game_state.board,
            key_map=key_map,
        )

    def play(self):
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
            turn_start = time.time()
            self.guesser.pause_recording()

            if self.game_state.turn > 0:
                continuity_text = self.guesser.get_continuity_remark(self.game_state,  adaptive=self.guesser.is_adaptive())
                if continuity_text:
                    self.guesser.say(continuity_text)

            self.guesser.say_random_human_turn()
            self.guesser.resume_recording()

            clue_word, num = self.receive_clue()

            # Stop recording immediately after the clue is received and classify
            features = None
            confidence_level = None
            if self.guesser.is_adaptive():
                self.guesser.say_random_sounds()  # to fill the silent gap while processing audio and make it feel more natural
                print("Processing audio for confidence level classification...")
                features, confidence_level = self.guesser.stop_and_process_audio(clue_word, self.game_state.turn)

            current_turn = self.game_state.turn
            # Default to medium confidence for the first turn, because it usually takes a moment for the spymaster to give a clue
            # which could lead to low confidence predictions that don't reflect the user's true confidence level.
            if current_turn == 0 and self.guesser.is_adaptive():
                confidence_level = CONFIDENCE_MEDIUM

            turn_result = self.turn_manager.play_turn(clue_word, num, confidence_level, features)
            turn_duration = time.time() - turn_start

            self.experiment_logger.log_turn(
                turn=current_turn,
                clue_word=clue_word,
                clue_number=num,
                features=features,
                confidence_level=confidence_level,
                guesses=turn_result["guesses"],
                outcomes=turn_result["outcomes"],
                score=turn_result["score"],
                turn_duration_s=turn_duration,
            )

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
        receive_start = time.time()
        long_wait_count = 0
        hesitation_said = False
        failed_attempts = 0
        while True:
            # --- Listen until we get some non-empty input ---
            raw_clue = self.guesser.listen()
            while not raw_clue:
                print("No input detected from listener; listening again")
                # Say a long-wait utterance (up to MAX_LONG_WAIT_REACTIONS times)
                # if the spymaster has been silent for a while (adaptive only).
                if (self.guesser.is_adaptive()
                        and long_wait_count < MAX_LONG_WAIT_REACTIONS
                        and time.time() - receive_start > LONG_WAIT_THRESHOLD_SECONDS * (long_wait_count + 1)):
                    is_first_long_wait = long_wait_count == 0
                    long_wait_count += 1
                    self.guesser.pause_recording()
                    if is_first_long_wait:
                        utterance = self.guesser.get_waiting_for_clue_long_wait_utterance()
                    else:
                        utterance = (
                            self.guesser.get_continuity_remark(self.game_state, adaptive=self.guesser.is_adaptive())
                            or self.guesser.get_waiting_for_clue_long_wait_utterance()
                        )
                    self.guesser.say(utterance)
                    self.guesser.resume_recording()
                raw_clue = self.guesser.listen()

            # --- Ignore utterances that are only filler/hesitation words ---
            if self._is_filler_only(raw_clue):
                # React to stress / difficulty words in the filler (adaptive only,
                # at most once per clue-reception attempt).
                if (self.guesser.is_adaptive()
                        and not hesitation_said
                        and contains_hesitation_trigger(raw_clue)):
                    hesitation_said = True
                    continuity_utterance = self.guesser.get_continuity_remark(self.game_state, adaptive=self.guesser.is_adaptive())
                    if continuity_utterance:
                        self.guesser.pause_recording()
                        self.guesser.say(continuity_utterance)
                        self.guesser.resume_recording()
                else:
                    print(f"Filler-only input detected ('{raw_clue}'); listening again silently")
                continue

            # --- Try to parse the clue ---
            try:
                clue_word, num = parse_clue(raw_clue)
            except Exception:
                failed_attempts += 1
                if failed_attempts <= GRACE_PERIOD_RETRIES:
                    # Grace period: the user may still be formulating their
                    # clue.  Wait briefly and try again without interrupting.
                    # If the utterance contains hesitation/difficulty words,
                    # react empathetically (adaptive condition only,
                    # at most once per clue-reception attempt).
                    if (self.guesser.is_adaptive()
                            and not hesitation_said
                            and contains_hesitation_trigger(raw_clue)):
                        hesitation_said = True
                        continuity_utterance = self.guesser.get_continuity_remark(self.game_state, adaptive=self.guesser.is_adaptive())
                        if continuity_utterance:
                            self.guesser.pause_recording()
                            self.guesser.say(continuity_utterance)
                            self.guesser.resume_recording()
                    else:
                        print(
                            f"Could not parse clue (attempt {failed_attempts}/{GRACE_PERIOD_RETRIES}); "
                            "waiting silently before retrying"
                        )
                    time.sleep(GRACE_PERIOD_WAIT_SECONDS)
                    continue

                # Grace period exhausted → ask the user to repeat
                failed_attempts = 0
                self.guesser.pause_recording()
                self.guesser.say_random_clue_not_understood()
                self.guesser.resume_recording()
                continue  # restart from listening

            failed_attempts = 0  # reset on successful parse

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
    def _is_filler_only(text: str) -> bool:
        """
        Returns True if the transcribed text contains only filler / hesitation
        words (e.g. "uh", "hmm", "um") and no meaningful content.  Used to
        avoid interrupting users who are still thinking out loud.
        """
        # Strip punctuation from each token before checking so that
        # transcriptions like "um," or "uh." are still detected as fillers.
        tokens = [t.strip(".,!?;:'\"") for t in text.lower().split()]
        tokens = [t for t in tokens if t]
        return len(tokens) > 0 and all(t in FILLERS for t in tokens)

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
