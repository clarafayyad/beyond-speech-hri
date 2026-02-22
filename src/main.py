import json

from sic_framework.devices import Pepper

from agents.tts_manager import NaoqiTTSConf
from codenames.game import CodenamesGame
from interaction.game_state import GameState
from interaction.game_loop import GameLoop
from agents.conversational_agent import ConversationalAgent

device_manager = Pepper(ip='192.168.1.85')
# device_manager = Desktop(speakers_conf=SpeakersConf(sample_rate=22050))
tts_conf = NaoqiTTSConf()
agent = ConversationalAgent(device_manager, tts_conf)

game = CodenamesGame()
game_state = GameState(
    board=game.board,
    card_descriptions=json.load(open("../assets/card_descriptions.json"))
)

loop = GameLoop(agent, game_state)
loop.play()
