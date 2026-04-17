import numpy as np
import pandas as pd

from interaction.run_calibration import normalize_self_report
from multimodal_perception.model.confidence_classifier import ConfidenceClassifier


def test_adjust_with_self_report_blends_probs_with_prior_vector():
    clf = ConfidenceClassifier()
    probs = np.array([0.2, 0.5, 0.3])

    adjusted = clf.adjust_with_self_report(probs, np.array([1.0, 0.0, 0.0]), alpha=0.3)

    np.testing.assert_allclose(adjusted, np.array([0.44, 0.35, 0.21]))
    np.testing.assert_allclose(np.sum(adjusted), 1.0)


def test_classify_uses_participant_self_report_prior():
    clf = ConfidenceClassifier()
    clf.probs = lambda _: np.array([0.1, 0.2, 0.7])  # type: ignore[method-assign]
    clf.calibration = {
        "participants": {"p1": {}},
        "global": {},
        "self_report_priors": {"participants": {"p1": np.array([0.0, 1.0, 0.0])}, "global": None},
    }

    probs, label = clf.classify({"participant_id": "p1", "self_report_alpha": 0.8})

    assert label == "low"
    np.testing.assert_allclose(probs, np.array([0.02, 0.84, 0.14]))


def test_classify_ignores_runtime_self_report_feature():
    clf = ConfidenceClassifier()
    clf.probs = lambda _: np.array([0.2, 0.6, 0.2])  # type: ignore[method-assign]
    clf.calibration = {
        "participants": {"p2": {}},
        "global": {},
        "self_report_priors": {"participants": {"p2": np.array([0.0, 1.0, 0.0])}, "global": None},
    }

    probs, label = clf.classify({"participant_id": "p2", "self_report": "high", "self_report_alpha": 1.0})

    assert label == "low"
    np.testing.assert_allclose(probs, np.array([0.0, 1.0, 0.0]))


def test_load_calibration_derives_self_report_prior_per_participant():
    clf = ConfidenceClassifier()
    df = pd.DataFrame(
        {
            "participant_id": ["1", "1", "1", "2"],
            "duration": [1.0, 2.0, 3.0, 4.0],
            "self_report": ["high", "low", "low", "medium"],
        }
    )

    clf._load_calibration_from_df(df)

    np.testing.assert_allclose(
        clf._get_self_report_prior_for_participant("1"),
        np.array([1.0 / 3.0, 2.0 / 3.0, 0.0]),
    )


def test_normalize_self_report():
    assert normalize_self_report("LOW") == "low"
    assert normalize_self_report("med") == "medium"
    assert normalize_self_report("h") == "high"
    assert normalize_self_report("invalid") is None
