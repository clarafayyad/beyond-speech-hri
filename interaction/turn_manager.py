import time

from agents.guesser import Guesser
from interaction.prompts import SYSTEM_PROMPT, build_user_prompt
from interaction.game_state import RED, BLUE, NEUTRAL, ASSASSIN, TOTAL_BLUE, TOTAL_RED


class TurnManager:
    def __init__(self, guesser: Guesser, game_state):
        self.guesser = guesser
        self.game_state = game_state

    def make_guess(self, clue_word):
        response = self.guesser.prompt_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(clue_word, self.game_state)
        )

        guess_idx = response["guess_index"]
        self.guesser.say(f"I choose card {guess_idx}.")
        self.guesser.display_guess(self.game_state.board[guess_idx])
        return guess_idx

    def get_feedback(self, guess_idx):
        while guess_idx not in self.game_state.revealed:
            print("Waiting for feedback...")
            time.sleep(0.5)

        return self.game_state.revealed[guess_idx]

    def play_turn(self, clue_word, max_guesses):
        guesses = 0

        while guesses < max_guesses and not self.game_state.game_over:
            self.guesser.dialog_manager.animate_thinking()
            self.guesser.say_random_thinking()

            guess_idx = self.make_guess(clue_word)
            result = self.get_feedback(guess_idx)

            self.game_state.revealed[guess_idx] = result
            self.game_state.history.append({
                "turn": self.game_state.turn,
                "clue": clue_word,
                "guess_number": guesses + 1,
                "guess": guess_idx,
                "result": result
            })

            guesses += 1

            if result == ASSASSIN:
                self.guesser.say_random_assassin_reaction()
                self.game_state.game_over = True
                self.game_state.win = False
                return

            if result == RED:
                self.guesser.say_random_red_reaction()
                break

            if result == BLUE:
                self.guesser.say_random_blue_reaction()
                continue

            if result == NEUTRAL:
                self.guesser.say_random_neutral_reaction()
                continue

        if self.guessed_all_blue_cards():
            self.game_state.game_over = True
            self.game_state.win = True
        elif self.placed_all_red_cards():
            self.game_state.game_over = True
            self.game_state.win = False

        self.game_state.turn += 1
        self.guesser.clear_display()

    def guessed_all_blue_cards(self):
        blue_revealed = sum(1 for color in self.game_state.revealed.values() if color == BLUE)
        return blue_revealed == TOTAL_BLUE

    def placed_all_red_cards(self):
        red_placed = sum(1 for color in self.game_state.revealed.values() if color == RED)
        return red_placed == TOTAL_RED
