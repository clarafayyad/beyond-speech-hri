import numpy as np
import pytest

from multimodal_perception.model.confidence_classifier import (
    ConfidenceClassifier,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
)


class _PatchedClassifier(ConfidenceClassifier):
    """ConfidenceClassifier subclass that injects fixed softmax probabilities."""

    def __init__(self, fixed_probs):
        # Bypass __init__ to avoid loading calibration files from disk.
        self.W = np.zeros((3, 15))
        self.b = np.zeros(3)
        self.calibration = None
        self.fallback_to_global = True
        self._fixed_probs = np.array(fixed_probs, dtype=float)

    def probs(self, features: dict) -> np.ndarray:
        return self._fixed_probs


class TestClassifyProbabilityCalibration:
    """Tests for the probability-based calibration rule in classify()."""

    # ------------------------------------------------------------------
    # Rule 1: max_prob >= 0.6 → keep the model's predicted class
    # ------------------------------------------------------------------

    def test_high_confidence_keeps_predicted_class_high(self):
        # probs order: [high, low, medium]
        clf = _PatchedClassifier([0.70, 0.20, 0.10])
        _, label = clf.classify({})
        assert label == CONFIDENCE_HIGH

    def test_high_confidence_keeps_predicted_class_low(self):
        clf = _PatchedClassifier([0.20, 0.70, 0.10])
        _, label = clf.classify({})
        assert label == CONFIDENCE_LOW

    def test_high_confidence_keeps_predicted_class_medium(self):
        clf = _PatchedClassifier([0.15, 0.15, 0.70])
        _, label = clf.classify({})
        assert label == CONFIDENCE_MEDIUM

    # ------------------------------------------------------------------
    # Rule 2: max_prob < 0.66, model split between low and high extremes
    # ------------------------------------------------------------------

    def test_split_between_extremes_overrides_to_medium(self):
        # max_prob < 0.66; low and high both > medium; |low - high| < 0.1
        # e.g. [high=0.42, low=0.45, medium=0.13] → max=0.45 < 0.66
        clf = _PatchedClassifier([0.42, 0.45, 0.13])
        _, label = clf.classify({})
        assert label == CONFIDENCE_MEDIUM

    def test_split_between_extremes_equal_probs_overrides_to_medium(self):
        # Equal low and high (|diff| = 0 < 0.1), both > medium
        clf = _PatchedClassifier([0.40, 0.40, 0.20])
        _, label = clf.classify({})
        assert label == CONFIDENCE_MEDIUM

    def test_split_between_extremes_diff_at_boundary_not_overridden(self):
        # |low - high| = 0.1, which is NOT < 0.1 (strict), so no override
        clf = _PatchedClassifier([0.35, 0.45, 0.20])
        # diff = 0.10, not < 0.10 → keep argmax (low=0.45)
        _, label = clf.classify({})
        assert label == CONFIDENCE_LOW

    def test_medium_dominates_no_override(self):
        # medium is highest → not both low and high > medium, no override
        clf = _PatchedClassifier([0.30, 0.30, 0.40])
        _, label = clf.classify({})
        assert label == CONFIDENCE_MEDIUM

    def test_no_split_high_alone_dominates(self):
        # low < medium, so split condition not met → keep high
        clf = _PatchedClassifier([0.55, 0.10, 0.35])
        _, label = clf.classify({})
        assert label == CONFIDENCE_HIGH

    # ------------------------------------------------------------------
    # Overlap zone: 0.60 <= max_prob < 0.66 with split condition
    # ------------------------------------------------------------------

    def test_overlap_zone_split_condition_overrides(self):
        # max_prob = 0.63 is >= 0.6 (rule 1 says keep) but also < 0.66,
        # and split condition met → rule 2 overrides to medium
        clf = _PatchedClassifier([0.63, 0.62, 0.05])
        # diff = |0.62 - 0.63| = 0.01 < 0.1; both > medium → override
        _, label = clf.classify({})
        assert label == CONFIDENCE_MEDIUM

    def test_overlap_zone_split_not_met_keeps_original(self):
        # max_prob = 0.63, but split diff too large → keep original class
        clf = _PatchedClassifier([0.63, 0.27, 0.10])
        # diff = |0.27 - 0.63| = 0.36 >= 0.1 → keep high
        _, label = clf.classify({})
        assert label == CONFIDENCE_HIGH

    # ------------------------------------------------------------------
    # max_prob >= 0.66 → rule 2 never fires, always keep predicted class
    # ------------------------------------------------------------------

    def test_above_066_no_override_even_if_near_split(self):
        # max_prob = 0.66; rule 2 condition (< 0.66) not met → keep original
        clf = _PatchedClassifier([0.66, 0.20, 0.14])
        _, label = clf.classify({})
        assert label == CONFIDENCE_HIGH

    # ------------------------------------------------------------------
    # Return value: probs array is always returned unchanged
    # ------------------------------------------------------------------

    def test_classify_returns_probs_array(self):
        fixed = [0.42, 0.45, 0.13]
        clf = _PatchedClassifier(fixed)
        probs, _ = clf.classify({})
        np.testing.assert_array_almost_equal(probs, fixed)
