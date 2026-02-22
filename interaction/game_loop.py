from agents.conversational_agent import ConversationalAgent
from interaction.turn_manager import TurnManager
from interaction.utils import parse_clue


class GameLoop:
    def __init__(self, agent: ConversationalAgent, game_state, max_turns=6):
        self.agent = agent
        self.game_state = game_state
        self.max_turns = max_turns
        self.turn_manager = TurnManager(agent, game_state)

    def play(self):
        self.agent.say("Let's start the game.")

        while not self.game_state.game_over and self.game_state.turn < self.max_turns:
            self.agent.say("Please give me a clue.")
            raw_clue = self.agent.listen()

            try:
                clue_word, num = parse_clue(raw_clue)
            except Exception:
                self.agent.say("I did not understand the clue. Please try again.")
                continue

            self.agent.say(f"I will make up to {num} guesses.")
            self.turn_manager.play_turn(clue_word, num)

        if not self.game_state.game_over:
            self.agent.say("The game is over.")

        if self.game_state.win is True:
            self.agent.say("We won. Good job.")
        elif self.game_state.win is False:
            self.agent.say("Sorry. We lost.")