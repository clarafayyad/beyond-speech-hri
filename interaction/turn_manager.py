import time

from agents.guesser import Guesser
from interaction.prompts import SYSTEM_PROMPT, build_user_prompt
from interaction.game_state import RED, BLUE, NEUTRAL, ASSASSIN, TOTAL_BLUE, TOTAL_RED


def _count_blue(outcomes):
    """Return the number of blue (correct) outcomes in a list."""
    return sum(1 for o in outcomes if o == BLUE)


class TurnManager:
    def __init__(self, guesser: Guesser, game_state):
        self.guesser = guesser
        self.game_state = game_state

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

    def play_turn(self, clue_word, max_guesses, confidence_level=None, features=None):
        # Say exactly one pre-guess utterance, chosen by a simple fallback:
        #   Turn 0 (no history): confidence reaction → thinking filler
        #   Later turns: continuity remark → confidence reaction → thinking filler
        confidence_text = self.guesser.get_confidence_level_reaction(confidence_level, features)
        continuity_text = self.guesser.get_continuity_remark(self.game_state, confidence_level)

        if self.game_state.turn == 0:
            utterance = confidence_text or self.guesser.get_random_thinking()
        else:
            utterance = continuity_text or confidence_text or self.guesser.get_random_thinking()

        self.game_state.confidence_history.append(confidence_level)

        self.guesser.say(utterance)

        guesses = 0
        turn_guesses = []
        turn_outcomes = []

        while guesses < max_guesses and not self.game_state.game_over:
            self.guesser.dialog_manager.animate_thinking()

            # On subsequent guesses, say a thinking filler since the
            # pre-turn utterance was only spoken before the first guess.
            if guesses > 0:
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

            turn_guesses.append(self.game_state.board[guess_idx])
            turn_outcomes.append(result)
            guesses += 1

            if result == ASSASSIN:
                self.guesser.say_random_assassin_reaction()
                self.game_state.game_over = True
                self.game_state.win = False
                score = _count_blue(turn_outcomes)
                return {"guesses": turn_guesses, "outcomes": turn_outcomes, "score": score}

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

        score = _count_blue(turn_outcomes)
        return {"guesses": turn_guesses, "outcomes": turn_outcomes, "score": score}

    def guessed_all_blue_cards(self):
        blue_revealed = sum(1 for color in self.game_state.revealed.values() if color == BLUE)
        return blue_revealed == TOTAL_BLUE

    def placed_all_red_cards(self):
        red_placed = sum(1 for color in self.game_state.revealed.values() if color == RED)
        return red_placed == TOTAL_RED - 1
