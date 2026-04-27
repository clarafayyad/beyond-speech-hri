import random

from interaction.game_state import BLUE


def _last_turn_performance(game_state):
    """Summarise the most recent completed turn's guess outcomes.

    Returns a dict with *blue_count*, *total*, *all_correct* and
    *any_correct*, or ``None`` when there is no previous turn.
    """
    if not game_state.history:
        return None

    last_turn = game_state.turn - 1
    guesses = [h for h in game_state.history if h["turn"] == last_turn]

    if not guesses:
        return None

    blue_count = sum(1 for g in guesses if g["result"] == BLUE)
    return {
        "blue_count": blue_count,
        "total": len(guesses),
        "all_correct": all(g["result"] == BLUE for g in guesses),
        "any_correct": blue_count > 0,
    }


def get_baseline_continuity_utterance(game_state):
    """Baseline mode does not include continuity utterances; always returns ``None``."""
    return None


def get_adaptive_continuity_utterance(game_state):
    """Return an adaptive continuity utterance only when low confidence or
    hesitation was detected in recent turns, or ``None`` otherwise.

    Continuity remarks are reserved for turns where the spymaster showed signs
    of uncertainty (low confidence trend), so that the robot only comments when
    there is a meaningful signal to react to.
    """
    if game_state.turn == 0:
        return None

    perf = _last_turn_performance(game_state)
    if perf is None:
        return None

    recent = game_state.confidence_history[-2:] if game_state.confidence_history else []
    low_trend = len(recent) > 0 and all(c == "low" for c in recent)

    if not low_trend:
        return None

    # All branches below are reached only when low_trend is True.
    if perf["all_correct"]:
        reactions = [
            "Those last ones were tough, but we still managed!",
            "Even with some uncertainty, we got through!",
            "It felt tricky, but our guesses were spot on!",
            "Not easy clues, but we still pulled it off!",
            "We guessed right even when it wasn't clear. Nice!",
        ]
    elif not perf["any_correct"]:
        reactions = [
            "Those were really tricky rounds. Let's regroup.",
            "It's been uncertain and tough. Let's refocus.",
            "Hard to read the clues lately. Let's try fresh.",
            "The signals have been unclear, but we can turn it around.",
            "Rough stretch with uncertain clues. New opportunity now!",
        ]
    else:
        # Mixed results with low confidence — not a clear pattern worth commenting on.
        return None

    return random.choice(reactions)
