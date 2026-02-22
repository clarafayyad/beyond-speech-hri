import json

from codenames.game import CodenamesGame
from interaction.game_state import GameState
from interaction.game_loop import GameLoop
from interaction.llm_agent import Agent

game = CodenamesGame()
agent = Agent()

game_state = GameState(
    board=game.board,
    card_descriptions=json.load(open("../assets/card_descriptions.json"))
)

loop = GameLoop(agent, game_state)
loop.play()