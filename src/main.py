import json

from sic_framework.devices import Pepper
from sic_framework.devices.common_desktop.desktop_speakers import SpeakersConf
from sic_framework.devices.desktop import Desktop

from agents.tts_manager import ElevenLabsTTSConf
from codenames.game import CodenamesGame
from interaction.game_state import GameState
from interaction.game_loop import GameLoop
from agents.guesser import Guesser

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Configurations & Conversational Agent Setup
    # device_manager = Pepper(ip='10.0.0.148')
    device_manager = Desktop(speakers_conf=SpeakersConf(sample_rate=22050))
    tts_conf = ElevenLabsTTSConf(voice_id='yO6w2xlECAQRFP6pX7Hw', stability=0.8)
    guesser = Guesser(device_manager, tts_conf)

    # Build Game
    game = CodenamesGame()
    game_state = GameState(
        board=game.board,
        card_descriptions=json.load(open("../assets/card_descriptions.json"))
    )

    input("Press Enter to start the game")

    # Start the interaction
    loop = GameLoop(guesser, game_state)
    loop.play()

    # Shutdown
    guesser.shutdown()

