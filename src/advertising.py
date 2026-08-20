from sic_framework.devices import Pepper
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest

from agents.dialog_manager import InteractionConf
from agents.tts_manager import ElevenLabsTTSConf
from agents.guesser import Guesser

if __name__ == "__main__":
    external_audio_device_index = 1
    device_manager = Pepper(ip='10.0.0.148')
    tts_conf = ElevenLabsTTSConf(voice_id='EXAVITQu4vr4xnSDxMaL', stability=0.2)
    int_conf = InteractionConf(real_time_stt=False, external_audio_device_id=external_audio_device_index, adaptive=True)

    guesser = Guesser(device_manager, tts_conf, int_conf)

    device_manager.autonomous.request(NaoBasicAwarenessRequest(False, tracking_mode="Head"))
    # time.sleep(2)  # Give the robot a moment to start tracking

    guesser.dialog_manager.animate_hello()
    guesser.say("Hey you!")
    guesser.dialog_manager.animate_random()
    guesser.say("Yes you.")
    guesser.dialog_manager.animate_random()
    guesser.say("I’m Ada, and I’m looking for a teammate!")
    guesser.dialog_manager.animate_random()
    guesser.say("Let’s play Codenames together and have some fun!")
    guesser.dialog_manager.animate_random()
    guesser.say("You'll earn a gift card based on your score!")
    guesser.dialog_manager.animate_random()
    guesser.say("Click on the link below to join me!")
    guesser.dialog_manager.animate_random()
    guesser.say("I'll see ya there!")
    guesser.dialog_manager.animate_bye()

    guesser.shutdown()