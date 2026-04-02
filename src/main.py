from sic_framework.devices import Pepper
from sic_framework.devices.common_desktop.desktop_speakers import SpeakersConf
from sic_framework.devices.desktop import Desktop

from agents.dialog_manager import InteractionConf
from agents.tts_manager import ElevenLabsTTSConf
from interaction.game import CodenamesGame
from interaction.game_state import GameState
from interaction.game_loop import GameLoop
from agents.guesser import Guesser

if __name__ == "__main__":

    # Configurations & Conversational Agent Setup
    device_manager = Pepper(ip='10.0.0.148')
    # device_manager = Desktop(speakers_conf=SpeakersConf(sample_rate=22050))
    # tts_conf = ElevenLabsTTSConf(voice_id='yO6w2xlECAQRFP6pX7Hw', stability=0.8)
    tts_conf = ElevenLabsTTSConf(voice_id='EXAVITQu4vr4xnSDxMaL', stability=0.2)
    int_conf = InteractionConf(real_time_stt=False)

    guesser = Guesser(device_manager, tts_conf, int_conf)

    # Build Game
    game = CodenamesGame(config_number=2)
    game_state = GameState(board=game.board)

    input("Press Enter to start the game")

    # Participant and audio configuration
    participant_id = "P01"
    # Set audio_device_index to the index of the external microphone.
    # Run `python -m sounddevice` to list available devices and their indices.
    audio_device_index = 3  # index of the external audio input device

    # Start the interaction
    loop = GameLoop(guesser, game_state, participant_id=participant_id, audio_device_index=audio_device_index)
    loop.play()

    # Shutdown
    guesser.shutdown()

