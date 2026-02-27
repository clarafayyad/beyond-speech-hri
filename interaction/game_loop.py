from agents.guesser import Guesser
from interaction.turn_manager import TurnManager
from interaction.utils import parse_clue


class GameLoop:
    def __init__(self, guesser: Guesser, game_state, max_turns=5):
        self.guesser = guesser
        self.game_state = game_state
        self.max_turns = max_turns
        self.turn_manager = TurnManager(guesser, game_state)

    def play(self):
        # TODO: introduce robot
        self.guesser.say_random_start_game()
        self.guesser.say("I'm waiting for you to place the red cards, let me know when you're ready.", sleep_time=0.5)

        response = self.guesser.listen()
        while not response or response == "" or "ready" not in response:
            print("Spymaster not ready yet, waiting...")
            response = self.guesser.listen()

        while not self.game_state.game_over and self.game_state.turn < self.max_turns:
            print(f"Playing Turn {self.game_state.turn}")
            self.guesser.say_random_human_turn()

            clue_word, num = self.receive_clue()
            self.turn_manager.play_turn(clue_word, num)

            if not self.game_state.game_over:
                self.guesser.say("Go ahead, place a red card.")
                input("Press enter after red card is placed.")

        if not self.game_state.game_over:
            self.guesser.say_random_game_over()

        if self.game_state.win is True:
            self.guesser.say_random_win_reaction()
        elif self.game_state.win is False:
            self.guesser.say_random_loss_reaction()

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
                self.guesser.say_random_clue_not_understood()
                continue  # restart from listening

            # --- Confirm understanding ---
            self.guesser.say_random_repeat_clue(clue_word, num)
            self.guesser.say("Did I get the clue right?")

            feedback = self.guesser.listen()
            if self.is_clue_well_received(feedback):
                return clue_word, num

            # --- Not confirmed → ask to repeat and loop ---
            self.guesser.say("Oh, could you repeat?")

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