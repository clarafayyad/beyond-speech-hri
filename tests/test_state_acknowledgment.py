from interaction.state_acknowledgment import (
    detect_additional_audio_states,
    get_additional_state_acknowledgment,
)


def test_detects_hesitation_from_audio_cues():
    features = {
        "transcript": "river",
        "duration": 3.0,
        "pause_max": 3.0,
        "speech_rate": 2.5,
        "verbal_hesitation_count": 0,
    }
    assert detect_additional_audio_states(features) == ["hesitation"]


def test_detects_hesitation_from_speech_rate():
    features = {
        "transcript": "river",
        "duration": 3.0,
        "pause_max": 0.2,
        "speech_rate": 1.0,
        "verbal_hesitation_count": 0,
    }
    assert detect_additional_audio_states(features) == ["hesitation"]


def test_detects_risk_from_transcript():
    features = {
        "transcript": "this one is risky, watch out for assassin",
        "duration": 3.0,
        "pause_max": 0.2,
        "speech_rate": 2.5,
        "verbal_hesitation_count": 0,
    }
    assert detect_additional_audio_states(features) == ["risk"]


def test_detects_hesitation_and_risk_together():
    features = {
        "transcript": "maybe this is tricky",
        "duration": 15.0,
        "pause_max": 3.0,
        "speech_rate": 1.0,
        "verbal_hesitation_count": 3,
    }
    assert detect_additional_audio_states(features) == ["hesitation", "risk"]


def test_additional_state_acknowledgment_empty_when_no_states():
    features = {
        "transcript": "river",
        "duration": 3.0,
        "pause_max": 0.2,
        "speech_rate": 2.5,
        "verbal_hesitation_count": 0,
    }
    assert get_additional_state_acknowledgment(features) == ""
