from agents.conversational_agent import ConversationalAgent
from interaction.prompts import SYSTEM_PROMPT, build_user_prompt
from interaction.utils import normalize_feedback


class TurnManager:
    def __init__(self, agent: ConversationalAgent, game_state):
        self.agent = agent
        self.game_state = game_state

    def make_guess(self, clue_word):
        response = self.agent.prompt_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(clue_word, self.game_state)
        )

        guess_idx = response["guess_index"]
        self.agent.say(f"I choose card {guess_idx}.")
        return guess_idx

    def get_feedback(self):
        self.agent.say("Please tell me the result.")
        feedback = normalize_feedback(self.agent.listen())

        if feedback is None:
            self.agent.say("Please say blue, red, neutral, or assassin.")
            return self.get_feedback()

        return feedback

    def play_turn(self, clue_word, max_guesses):
        guesses = 0

        while guesses < max_guesses and not self.game_state.game_over:
            guess_idx = self.make_guess(clue_word)
            result = self.get_feedback()

            self.game_state.revealed[guess_idx] = result
            self.game_state.history.append({
                "turn": self.game_state.turn,
                "clue": clue_word,
                "guess_number": guesses + 1,
                "guess": guess_idx,
                "result": result
            })

            guesses += 1

            if result == "assassin":
                self.agent.say("I chose the assassin. We lose.")
                self.game_state.game_over = True
                self.game_state.win = False
                return

            if result == "red":
                self.agent.say("That was red. Turn ends.")
                break

            if result == "blue":
                self.agent.say("Blue card. Continuing.")
                continue

            if result == "neutral":
                self.agent.say("Neutral card. Continuing.")
                continue

        self.game_state.turn += 1