def adjusted_guess_limit(requested_guesses, confidence_level=None, certainty=None):
    """Return certainty-adaptive guess budget for the current turn."""
    try:
        requested = int(requested_guesses)
    except Exception:
        requested = 0
    requested = max(0, requested)

    if confidence_level == "low":
        base = 1 if requested > 0 else 0
    elif confidence_level == "medium":
        base = max(1, requested - 1) if requested > 0 else 0
    else:
        base = requested

    if certainty is None:
        return base

    try:
        certainty = float(certainty)
    except Exception:
        return base

    certainty = min(1.0, max(0.0, certainty))
    if certainty >= 0.8:
        return min(requested + 1, base + 1)
    if certainty < 0.5:
        return max(0, base - 1)
    return base
