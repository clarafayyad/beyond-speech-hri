import time

from agents.guesser import Guesser
from interaction.prompts import SYSTEM_PROMPT, INNER_DISCUSSION_SYSTEM_PROMPT, build_user_prompt, build_inner_discussion_prompt
from interaction.game_state import RED, BLUE, NEUTRAL, ASSASSIN, TOTAL_BLUE, TOTAL_RED
from multimodal_perception.model.confidence_classifier import CONFIDENCE_LOW


class TurnManager:
    def __init__(self, guesser: Guesser, game_state):
        self.guesser = guesser
        self.game_state = game_state
        self.last_turn_memory = None  # minimal 1-turn memory: stores previous turn metadata
        self.inner_discussion_used = False  # inner discussion may only fire once per game

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

    def _should_trigger_inner_discussion(self, confidence_level):
        """Return True when the 2-agent inner discussion should be triggered.

        Conditions:
        - Confidence level is LOW.
        - The inner discussion has not yet been used in this game (once per game).
        """
        return confidence_level == CONFIDENCE_LOW and not self.inner_discussion_used

    def _run_inner_discussion(self, clue_word):
        """Simulate a short 2-agent consultation and return the chosen guess index.

        The robot voices Player 1's question, then Clara's (Player 2) reply, then
        commits to the final decision.  Marks the discussion as used so it cannot
        fire again this game.

        Returns the guess_index from the discussion, or None if the LLM response
        is invalid (caller should fall back to make_guess).
        """
        self.inner_discussion_used = True

        response = self.guesser.prompt_llm(
            system_prompt=INNER_DISCUSSION_SYSTEM_PROMPT,
            user_prompt=build_inner_discussion_prompt(clue_word, self.game_state),
        )

        player1_line = response.get("player1_line", "")
        player2_line = response.get("player2_line", "")
        final_decision = response.get("final_decision", "")
        guess_idx = response.get("guess_index")

        # Validate that the returned index is an unrevealed card
        valid_indices = [
            idx for idx in range(len(self.game_state.board))
            if idx not in self.game_state.revealed
        ]
        if guess_idx not in valid_indices:
            return None

        if player1_line:
            self.guesser.say(player1_line)
        if player2_line:
            self.guesser.say_as_clara(player2_line)
        if final_decision:
            self.guesser.say(final_decision)

        self.guesser.display_guess(self.game_state.board[guess_idx])

        return guess_idx

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

    def play_turn(self, clue_word, max_guesses, confidence_level=None):
        if self._should_say_memory_reference(confidence_level):
            self.guesser.say_memory_reference(self.last_turn_memory)

        self.guesser.say_confidence_level_reaction(confidence_level)

        guesses = 0

        while guesses < max_guesses and not self.game_state.game_over:
            self.guesser.dialog_manager.animate_thinking()
            self.guesser.say_random_thinking()

            # On the first guess of the turn, trigger the inner discussion if
            # this is a low-confidence turn and it hasn't been used yet.
            if guesses == 0 and self._should_trigger_inner_discussion(confidence_level):
                self.guesser.say_inner_discussion_intro()
                guess_idx = self._run_inner_discussion(clue_word)
                if guess_idx is None:
                    # Fallback: LLM returned an invalid index
                    guess_idx = self.make_guess(clue_word, confidence_level)
            else:
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
