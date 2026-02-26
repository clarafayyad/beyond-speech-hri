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

            clue_received = False
            while not clue_received:
                raw_clue = self.guesser.listen()
                while raw_clue is None or raw_clue == "":
                    print("No input detected from listener; listening again")
                    raw_clue = self.guesser.listen()
                try:
                    clue_word, num = parse_clue(raw_clue)
                except Exception:
                    self.guesser.say_random_clue_not_understood()
                    continue
                clue_received = True

            self.guesser.say_random_repeat_clue(clue_word, num)
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