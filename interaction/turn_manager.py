import time

from agents.guesser import Guesser
from interaction.prompts import SYSTEM_PROMPT, build_user_prompt
from interaction.game_state import RED, BLUE, NEUTRAL, ASSASSIN, TOTAL_BLUE, TOTAL_RED
from multimodal_perception.model.confidence_classifier import CONFIDENCE_LOW


class TurnManager:
    def __init__(self, guesser: Guesser, game_state):
        self.guesser = guesser
        self.game_state = game_state
        self.last_turn_memory = None  # minimal 1-turn memory: stores previous turn metadata

    def make_guess(self, clue_word, confidence_level=None):
        response = self.guesser.prompt_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(clue_word, self.game_state, confidence_level)
        )

        guess_idx = response["guess_index"]
        self.guesser.display_guess(self.game_state.board[guess_idx])
        
        reason = response.get("reason", "")
        if reason:
            self.guesser.say(reason)
        else:
            self.guesser.say_random_guess()

        return guess_idx

    def get_feedback(self, guess_idx):
        while guess_idx not in self.game_state.revealed:
            print("Waiting for feedback...")
            time.sleep(0.5)

        return self.game_state.revealed[guess_idx]

    def _should_say_memory_reference(self, confidence_level):
        """Return True when a memory callback is contextually relevant.

        Currently triggers when both the previous turn and the current turn
        carry low confidence, signalling a repeated uncertain situation that
        the robot can acknowledge naturally.
        """
        if self.last_turn_memory is None:
            return False
        return (
            confidence_level == CONFIDENCE_LOW
            and self.last_turn_memory["confidence"] == CONFIDENCE_LOW
        )

    def _build_turn_memory(self, confidence_level, results):
        """Return a minimal metadata dict for the turn that just finished."""
        return {
            "confidence": confidence_level,
            "had_correct_guess": BLUE in results,
        }

    def play_turn(self, clue_word, max_guesses, confidence_level=None, features=None):
        if self._should_say_memory_reference(confidence_level):
            self.guesser.say_memory_reference(self.last_turn_memory)

        self.guesser.say_confidence_level_reaction(confidence_level, features)

        guesses = 0

        while guesses < max_guesses and not self.game_state.game_over:
            self.guesser.dialog_manager.animate_thinking()
            self.guesser.say_random_thinking()

            guess_idx = self.make_guess(clue_word, confidence_level)
            result = self.get_feedback(guess_idx)

            self.game_state.revealed[guess_idx] = result
            self.game_state.history.append({
                "turn": self.game_state.turn,
                "clue": clue_word,
                "guess_number": guesses + 1,
                "guess": guess_idx,
                "card": self.game_state.board[guess_idx],
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

        # Save this turn's metadata for optional use at the start of the next turn
        turn_results = [
            entry["result"]
            for entry in self.game_state.history
            if entry["turn"] == self.game_state.turn
        ]
        self.last_turn_memory = self._build_turn_memory(confidence_level, turn_results)

        self.game_state.turn += 1
        self.guesser.clear_display()

    def guessed_all_blue_cards(self):
        blue_revealed = sum(1 for color in self.game_state.revealed.values() if color == BLUE)
        return blue_revealed == TOTAL_BLUE

    def placed_all_red_cards(self):
        red_placed = sum(1 for color in self.game_state.revealed.values() if color == RED)
        return red_placed == TOTAL_RED - 1
