from interaction.confidence_decision import adjusted_guess_limit


def test_unknown_label_uses_requested_guess_count():
    assert adjusted_guess_limit(3, None, None) == 3


def test_medium_label_is_more_cautious_without_certainty():
    assert adjusted_guess_limit(3, "medium", None) == 2


def test_low_label_is_conservative_without_certainty():
    assert adjusted_guess_limit(3, "low", None) == 1


def test_high_certainty_increases_aggressiveness():
    assert adjusted_guess_limit(3, "medium", 0.85) == 3


def test_low_certainty_reduces_guess_budget():
    assert adjusted_guess_limit(3, "high", 0.40) == 2


def test_very_low_certainty_can_stop_turn():
    assert adjusted_guess_limit(1, "low", 0.30) == 0
