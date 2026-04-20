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
    """Return a non-adaptive continuity utterance, or ``None`` on turn 0."""
    if game_state.turn == 0:
        return None

    perf = _last_turn_performance(game_state)
    if perf is None:
        return None

    if perf["all_correct"]:
        reactions = [
            "We nailed every guess last round, we've got this!",
            "All correct last time! Let's keep it going.",
            "Last round went perfectly. Let's do it again!",
            "Every guess was right last round. Nice!",
            "Great round before this one. Let's keep the momentum!",
        ]
    elif perf["any_correct"]:
        # Mixed results aren't a clear pattern worth commenting on;
        # let the fallback chain pick a different utterance type.
        return None
    else:
        reactions = [
            "Last round was rough, but let's bounce back!",
            "We've had better rounds, but I'm still optimistic!",
            "Okay, last round didn't go our way. Fresh start!",
            "Tough luck before this round, but we've got another chance!",
            "Last round was a setback, but we're not giving up!",
        ]

    return random.choice(reactions)


def get_adaptive_continuity_utterance(game_state, confidence_level):
    """Return an adaptive continuity utterance, or ``None`` on turn 0."""
    if game_state.turn == 0:
        return None

    perf = _last_turn_performance(game_state)
    if perf is None:
        return None

    recent = game_state.confidence_history[-2:] if game_state.confidence_history else []
    low_trend = len(recent) > 0 and all(c == "low" for c in recent)
    high_trend = len(recent) > 0 and all(c == "high" for c in recent)

    if perf["all_correct"] and low_trend:
        reactions = [
            "Those last ones were tough, but we still managed!",
            "Even with some uncertainty, we got through!",
            "It felt tricky, but our guesses were spot on!",
            "Not easy clues, but we still pulled it off!",
            "We guessed right even when it wasn't clear. Nice!",
        ]
    elif perf["all_correct"] and high_trend:
        reactions = [
            "We've been confident and correct. Let's keep it up!",
            "Strong clues, strong guesses. Great teamwork!",
            "Everything clicked last round. More of that, please!",
            "We're in sync! Confidence is high and so are results!",
            "Clear signals and correct answers. Let's continue!",
        ]
    elif perf["any_correct"] and low_trend:
        # Mixed results aren't a clear pattern worth commenting on.
        return None
    elif not perf["any_correct"] and low_trend:
        reactions = [
            "Those were really tricky rounds. Let's regroup.",
            "It's been uncertain and tough. Let's refocus.",
            "Hard to read the clues lately. Let's try fresh.",
            "The signals have been unclear, but we can turn it around.",
            "Rough stretch with uncertain clues. New opportunity now!",
        ]
    elif perf["all_correct"]:
        reactions = [
            "Last round went well! Let's keep going.",
            "Good results before. We're on track!",
            "Nice guesses last time. Let's stay sharp!",
            "We've been doing well. Keep it up!",
            "Great teamwork last round!",
        ]
    elif perf["any_correct"]:
        # Mixed results aren't a clear pattern worth commenting on.
        return None
    else:
        reactions = [
            "Last round didn't go great, but here we go!",
            "Tough round before. Let's make this one count!",
            "That was a setback. Time to recover!",
            "We missed last time, but I'm ready for this!",
            "Fresh round, fresh chances. Let's do this!",
        ]

    return random.choice(reactions)
