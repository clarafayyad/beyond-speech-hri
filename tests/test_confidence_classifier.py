import numpy as np

from multimodal_perception.model.confidence_classifier import (
    CONFIDENCE_HIGH,
    CONFIDENCE_THRESHOLD,
    ConfidenceClassifier,
)


def test_classify_returns_none_below_threshold():
    classifier = ConfidenceClassifier()
    lead_prob = CONFIDENCE_THRESHOLD - 0.01
    remainder = 1.0 - lead_prob
    classifier.probs = lambda _features: np.array(
        [lead_prob, remainder / 2, remainder / 2]
    )

    _probs, label = classifier.classify({})

    assert label is None


def test_classify_keeps_label_at_threshold():
    classifier = ConfidenceClassifier()
    classifier.probs = lambda _features: np.array(
        [CONFIDENCE_THRESHOLD, (1 - CONFIDENCE_THRESHOLD) / 2, (1 - CONFIDENCE_THRESHOLD) / 2]
    )

    _probs, label = classifier.classify({})

    assert label == CONFIDENCE_HIGH
