import random


_RISK_KEYWORDS = (
    "risky",
    "dangerous",
    "watch out",
    "careful",
    "assassin",
    "not sure",
    "uncertain",
)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def detect_additional_audio_states(features: dict | None) -> list[str]:
    """Detect non-confidence conversational states from extracted audio features."""
    if not features:
        return []

    hesitation_count = _safe_float(features.get("verbal_hesitation_count"))
    pause_max = _safe_float(features.get("pause_max"))
    speech_rate = _safe_float(features.get("speech_rate"))
    duration = _safe_float(features.get("duration"))

    disfluency_score_raw = features.get("disfluency_score")
    disfluency_score = (
        _safe_float(disfluency_score_raw)
        if disfluency_score_raw is not None
        else None
    )
    transcript = str(features.get("transcript") or "").lower()

    hesitation_detected = (
        hesitation_count >= 2
        or pause_max >= 2.5
        or (0 < speech_rate < 1.5)
        or (disfluency_score is not None and disfluency_score >= 0.70)
    )

    risk_detected = any(keyword in transcript for keyword in _RISK_KEYWORDS) or (
        duration >= 12 and hesitation_detected
    )

    states = []
    if hesitation_detected:
        states.append("hesitation")
    if risk_detected:
        states.append("risk")
    return states


def get_additional_state_acknowledgment(features: dict | None) -> str:
    """Return a short utterance acknowledging additional inferred states."""
    states = detect_additional_audio_states(features)
    if not states:
        return ""

    if "risk" in states and "hesitation" in states:
        return random.choice([
            "I hear some hesitation, and this clue sounds a bit risky.",
            "That clue sounded cautious and a little risky to me.",
            "I sense hesitation here — I'll treat this one carefully.",
        ])

    if "risk" in states:
        return random.choice([
            "This clue sounds a little risky — I'll be extra careful.",
            "I’m sensing some risk in that clue, so I’ll play this safe.",
            "That one feels a bit dangerous; I’ll proceed carefully.",
        ])

    if "hesitation" in states:
        return random.choice([
            "I can hear a bit of hesitation in that clue.",
            "That clue sounded a little hesitant to me.",
            "I picked up some hesitation there — I’ll slow down.",
        ])

    return ""
