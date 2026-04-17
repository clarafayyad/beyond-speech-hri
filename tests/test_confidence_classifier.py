import numpy as np

from interaction.run_calibration import normalize_self_report
from multimodal_perception.model.confidence_classifier import ConfidenceClassifier


def test_adjust_with_self_report_blends_probs():
    clf = ConfidenceClassifier()
    probs = np.array([0.2, 0.5, 0.3])

    adjusted = clf.adjust_with_self_report(probs, "HIGH", alpha=0.3)

    np.testing.assert_allclose(adjusted, np.array([0.44, 0.35, 0.21]))
    np.testing.assert_allclose(np.sum(adjusted), 1.0)


def test_classify_uses_self_report_when_provided():
    clf = ConfidenceClassifier()
    clf.probs = lambda _: np.array([0.2, 0.7, 0.1])  # type: ignore[method-assign]

    probs, label = clf.classify({"self_report": "high", "self_report_alpha": 0.8})

    assert label == "high"
    np.testing.assert_allclose(probs, np.array([0.84, 0.14, 0.02]))


def test_classify_ignores_invalid_self_report():
    clf = ConfidenceClassifier()
    clf.probs = lambda _: np.array([0.1, 0.2, 0.7])  # type: ignore[method-assign]

    probs, label = clf.classify({"self_report": "unknown"})

    assert label == "medium"
    np.testing.assert_allclose(probs, np.array([0.1, 0.2, 0.7]))


def test_normalize_self_report():
    assert normalize_self_report("LOW") == "low"
    assert normalize_self_report("med") == "medium"
    assert normalize_self_report("h") == "high"
    assert normalize_self_report("invalid") is None
