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

# Configurations
participant_id = "1"
is_adaptive = False
stt_mic_device_index = 2  # This should be the robot's microphone (or desktop mic if not using the robot)
audio_features_mic_device_index = 1  # This should be the external mic that the participant is wearing
board_config_number = 1
robot_ip = '10.0.0.148'


def run():
    # Conversational Agent Setup
    mic_conf = MicrophoneConf(device_index=stt_mic_device_index)
    # device_manager = Pepper(ip=robot_ip)
    device_manager = Desktop(speakers_conf=SpeakersConf(sample_rate=22050), mic_conf=mic_conf)
    tts_conf = ElevenLabsTTSConf(voice_id='EXAVITQu4vr4xnSDxMaL', stability=0.2)
    int_conf = InteractionConf(
        real_time_stt=False,
        external_audio_device_id=audio_features_mic_device_index,
        participant_id=participant_id,
        adaptive=is_adaptive
    )

    guesser = Guesser(device_manager, tts_conf, int_conf)

    # Build Game
    game = CodenamesGame(config_number=board_config_number)
    game_state = GameState(board=game.board)

    input("Press Enter to start the game")

    # Start the interaction
    loop = GameLoop(
        guesser,
        game_state,
        participant_id=participant_id,
        is_adaptive=is_adaptive,
        board=game.board_id,
        key_map=game.map_name,
    )
    loop.play()

    # Shutdown
    guesser.shutdown()


if __name__ == "__main__":
    run()
