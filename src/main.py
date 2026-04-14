from sic_framework.devices import Pepper
from sic_framework.devices.common_desktop.desktop_microphone import MicrophoneConf
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
    participant_id = "1"

    # External audio device index is used for speech recording + feature extraction
    # Device manager's (e.g. Pepper, or Desktop) mic is used for speech recognition
    external_audio_device_index = 1
    # device_manager = Pepper(ip='10.0.0.148')
    device_manager = Desktop(speakers_conf=SpeakersConf(sample_rate=22050), mic_conf=MicrophoneConf(device_index=2))
    tts_conf = ElevenLabsTTSConf(voice_id='EXAVITQu4vr4xnSDxMaL', stability=0.2)
    int_conf = InteractionConf(real_time_stt=False, external_audio_device_id=external_audio_device_index,
                               participant_id=participant_id,
                               adaptive=True)  # Set adaptive=False for non-adaptive (baseline) condition

    guesser = Guesser(device_manager, tts_conf, int_conf)

    # Build Game
    game = CodenamesGame(config_number=2)
    game_state = GameState(board=game.board)

    input("Press Enter to start the game")

    # Start the interaction
    condition = "adaptive" if int_conf.adaptive else "baseline"
    loop = GameLoop(guesser, game_state,
                    participant_id=participant_id,
                    condition=condition,
                    key_map=game.map)
    loop.play()

    # Shutdown
    guesser.shutdown()

