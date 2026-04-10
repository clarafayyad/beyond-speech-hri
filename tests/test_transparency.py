"""Tests for the transparency-in-state-acknowledgment feature.

The robot should briefly explain the reasoning behind its confidence inference
by referencing the most notable audio feature, e.g. 'That took too long.'
"""

import pytest
from unittest.mock import MagicMock

from agents.guesser import Guesser
from multimodal_perception.model.confidence_classifier import (
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_HIGH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_guesser_with_real_methods(*method_names):
    """Return a MagicMock Guesser with the listed real methods restored."""
    guesser = MagicMock(spec=Guesser)
    for name in method_names:
        method = getattr(Guesser, name)
        setattr(guesser, name, method.__get__(guesser, type(guesser)))
    return guesser


# ---------------------------------------------------------------------------
# Guesser._feature_comment
# ---------------------------------------------------------------------------

class TestFeatureComment:
    """Unit tests for the static _feature_comment helper."""

    # --- low confidence triggers ---

    def test_long_duration_triggers_low_confidence_comment(self):
        features = {'duration': 20, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert comment != ""

    def test_high_hesitation_triggers_low_confidence_comment(self):
        features = {'duration': 5, 'verbal_hesitation_count': 3, 'pause_max': 0, 'speech_rate': 2.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert comment != ""

    def test_long_pause_triggers_low_confidence_comment(self):
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 3.0, 'speech_rate': 2.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert comment != ""

    def test_slow_speech_rate_triggers_low_confidence_comment(self):
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 1.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert comment != ""

    # --- high confidence triggers ---

    def test_no_hesitation_triggers_high_confidence_comment(self):
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.5}
        comment = Guesser._feature_comment(features, CONFIDENCE_HIGH)
        assert comment != ""

    def test_short_duration_triggers_high_confidence_comment(self):
        # When hesitation_count > 0 the "no hesitation" branch is skipped;
        # a short duration should still produce a comment.
        features = {'duration': 2.5, 'verbal_hesitation_count': 1, 'pause_max': 0, 'speech_rate': 2.5}
        comment = Guesser._feature_comment(features, CONFIDENCE_HIGH)
        assert comment != ""

    def test_fast_speech_rate_triggers_high_confidence_comment(self):
        # Hesitation_count=1 skips "no hesitation"; duration=5 skips "short".
        features = {'duration': 5, 'verbal_hesitation_count': 1, 'pause_max': 0, 'speech_rate': 4.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_HIGH)
        assert comment != ""

    # --- medium confidence: no feature comment expected ---

    def test_medium_confidence_always_returns_empty(self):
        # Even with extreme feature values, medium confidence yields no comment.
        features = {'duration': 25, 'verbal_hesitation_count': 5, 'pause_max': 4.0, 'speech_rate': 0.5}
        comment = Guesser._feature_comment(features, CONFIDENCE_MEDIUM)
        assert comment == ""

    # --- no-comment cases ---

    def test_unremarkable_features_return_empty_for_low_confidence(self):
        # All features within normal ranges → no notable signal.
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 1.0, 'speech_rate': 2.5}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert comment == ""

    def test_none_features_returns_empty(self):
        assert Guesser._feature_comment(None, CONFIDENCE_LOW) == ""

    def test_empty_dict_returns_empty(self):
        assert Guesser._feature_comment({}, CONFIDENCE_LOW) == ""

    # --- priority ordering: duration beats hesitation for low confidence ---

    def test_duration_takes_priority_over_hesitation_for_low(self):
        # Both duration (>12) and hesitation (>=2) are notable; the comment
        # should reference duration because it is checked first.
        features = {'duration': 15, 'verbal_hesitation_count': 3, 'pause_max': 0, 'speech_rate': 2.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        duration_phrases = [
            "Ow no, that took too long.",
            "Hmm, you spent a while on that one.",
            "That took quite some time.",
        ]
        assert comment in duration_phrases

    # --- return type is always a string ---

    def test_return_type_is_string_when_comment_present(self):
        features = {'duration': 20, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.0}
        comment = Guesser._feature_comment(features, CONFIDENCE_LOW)
        assert isinstance(comment, str)

    def test_return_type_is_string_when_no_comment(self):
        comment = Guesser._feature_comment(None, CONFIDENCE_LOW)
        assert isinstance(comment, str)


# ---------------------------------------------------------------------------
# Guesser.say_confidence_level_reaction  (with features)
# ---------------------------------------------------------------------------

class TestSayConfidenceLevelReactionWithFeatures:

    def _make_guesser(self):
        return _make_guesser_with_real_methods("say_confidence_level_reaction", "_feature_comment")

    def test_says_something_for_low_confidence_with_notable_features(self):
        guesser = self._make_guesser()
        features = {'duration': 20, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.0}
        guesser.say_confidence_level_reaction(CONFIDENCE_LOW, features)
        guesser.say.assert_called_once()
        phrase = guesser.say.call_args[0][0]
        assert isinstance(phrase, str) and len(phrase) > 0

    def test_says_something_for_low_confidence_without_features(self):
        guesser = self._make_guesser()
        guesser.say_confidence_level_reaction(CONFIDENCE_LOW, None)
        guesser.say.assert_called_once()

    def test_says_something_for_high_confidence_with_notable_features(self):
        guesser = self._make_guesser()
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.5}
        guesser.say_confidence_level_reaction(CONFIDENCE_HIGH, features)
        guesser.say.assert_called_once()

    def test_feature_comment_prepended_to_reaction_for_low_confidence(self):
        guesser = self._make_guesser()
        # Long duration (>12) always produces a comment for low confidence.
        features = {'duration': 20, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.0}
        guesser.say_confidence_level_reaction(CONFIDENCE_LOW, features)
        phrase = guesser.say.call_args[0][0]
        # The combined phrase should be longer than any single reaction string.
        assert len(phrase) > 30

    def test_no_feature_comment_gives_shorter_phrase(self):
        guesser = self._make_guesser()
        # No notable features → only the confidence reaction phrase is spoken.
        features = {'duration': 5, 'verbal_hesitation_count': 0, 'pause_max': 1.0, 'speech_rate': 2.5}
        guesser.say_confidence_level_reaction(CONFIDENCE_LOW, features)
        phrase_no_comment = guesser.say.call_args[0][0]

        guesser2 = self._make_guesser()
        features_notable = {'duration': 20, 'verbal_hesitation_count': 0, 'pause_max': 0, 'speech_rate': 2.0}
        guesser2.say_confidence_level_reaction(CONFIDENCE_LOW, features_notable)
        phrase_with_comment = guesser2.say.call_args[0][0]

        assert len(phrase_with_comment) > len(phrase_no_comment)

    def test_says_nothing_for_none_confidence_level(self):
        guesser = self._make_guesser()
        guesser.say_confidence_level_reaction(None, None)
        guesser.say.assert_not_called()

    def test_backward_compatible_call_without_features(self):
        """Calling say_confidence_level_reaction with just confidence_level still works."""
        guesser = self._make_guesser()
        guesser.say_confidence_level_reaction(CONFIDENCE_HIGH)
        guesser.say.assert_called_once()
