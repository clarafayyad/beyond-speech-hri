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
        self.guesser.say(self.guesser.random_start_game())

        while not self.game_state.game_over and self.game_state.turn < self.max_turns:
            self.guesser.say(self.guesser.random_human_turn())
            raw_clue = self.guesser.listen()
            while raw_clue is None or raw_clue == "":
                print("No input detected from listener; listening again")
                raw_clue = self.guesser.listen()

            try:
                clue_word, num = parse_clue(raw_clue)
            except Exception:
                self.guesser.say(self.guesser.random_clue_not_understood())
                continue

            self.guesser.say(self.guesser.random_repeat_clue(clue_word, num))
            self.turn_manager.play_turn(clue_word, num)

        if not self.game_state.game_over:
            self.guesser.say(self.guesser.random_game_over())

        if self.game_state.win is True:
            self.guesser.say(self.guesser.random_win_reaction())
        elif self.game_state.win is False:
            self.guesser.say(self.guesser.random_loss_reaction())