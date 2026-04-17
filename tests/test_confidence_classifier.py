from multimodal_perception.model.confidence_classifier import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ConfidenceClassifier,
)


def test_long_duration_alone_does_not_force_low_confidence():
    classifier = ConfidenceClassifier(participant_id='1')
    features = {
        'participant_id': '1',
        'duration': 40,
        'pause_max': 1.0,
        'speech_rate': 0.4,
        'mfcc_2_mean': 20,
        'verbal_hesitation_count': 0,
        'hnr': 16,
        'energy_std': 0.087,
        'pause_mid_speech': 0,
        'pause_count': 5,
    }

    _, label = classifier.classify(features)

    assert label in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}


def test_long_duration_with_other_low_signals_can_still_be_low():
    classifier = ConfidenceClassifier(participant_id='1')
    features = {
        'participant_id': '1',
        'duration': 40,
        'pause_max': 4.0,
        'speech_rate': 0.2,
        'mfcc_2_mean': 5,
        'verbal_hesitation_count': 4,
        'hnr': 10,
        'energy_std': 0.1,
        'pause_mid_speech': 3,
        'pause_count': 20,
    }

    _, label = classifier.classify(features)

    assert label == CONFIDENCE_LOW
