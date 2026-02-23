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
        # TODO: optional: explain the game to the player before starting
        self.guesser.say("Let's start the game.")

        while not self.game_state.game_over and self.game_state.turn < self.max_turns:
            self.guesser.say("Waiting for your clue...")
            raw_clue = self.guesser.listen()

            try:
                clue_word, num = parse_clue(raw_clue)
            except Exception:
                self.guesser.say("I did not understand the clue. Please try again.")
                continue

            self.guesser.say(f"I will make up to {num} guesses.")
            self.turn_manager.play_turn(clue_word, num)

        if not self.game_state.game_over:
            self.guesser.say("The game is over.")

        if self.game_state.win is True:
            self.guesser.say("We won. Good job.")
        elif self.game_state.win is False:
            self.guesser.say("Sorry. We lost.")