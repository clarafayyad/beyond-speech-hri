import random
import re


_GENERAL_RISK_KEYWORDS = (
    "risky",
    "dangerous",
    "watch out",
    "careful",
    "not sure",
    "uncertain",
)
_DOMAIN_RISK_KEYWORDS = (
    # In Codenames, "assassin" is an explicit high-risk signal.
    "assassin",
)
_RISK_KEYWORD_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(keyword)}\b")
    for keyword in (_GENERAL_RISK_KEYWORDS + _DOMAIN_RISK_KEYWORDS)
)

HESITATION_COUNT_THRESHOLD = 2
LONG_PAUSE_THRESHOLD = 2.5
SLOW_SPEECH_RATE_THRESHOLD = 1.5
DISFLUENCY_THRESHOLD = 0.70
RISK_DURATION_THRESHOLD = 12


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _contains_risk_keyword(text: str) -> bool:
    return any(pattern.search(text) for pattern in _RISK_KEYWORD_PATTERNS)


def detect_additional_audio_states(features: dict | None) -> list[str]:
    """Detect non-confidence conversational states from extracted audio features.

    Parameters
    ----------
    features : dict | None
        Optional feature dictionary with keys such as ``transcript``,
        ``verbal_hesitation_count``, ``pause_max``, ``speech_rate``,
        ``duration``, and optional ``disfluency_score``.

    Returns
    -------
    list[str]
        Zero or more state labels from ``{"hesitation", "risk"}``.
    """
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
    has_speech_rate = features.get("speech_rate") is not None
    transcript = str(features.get("transcript") or "").lower()

    hesitation_detected = (
        hesitation_count >= HESITATION_COUNT_THRESHOLD
        or pause_max >= LONG_PAUSE_THRESHOLD
        or (has_speech_rate and speech_rate < SLOW_SPEECH_RATE_THRESHOLD)
        or (disfluency_score is not None and disfluency_score >= DISFLUENCY_THRESHOLD)
    )

    risk_detected = _contains_risk_keyword(transcript) or (
        duration >= RISK_DURATION_THRESHOLD and hesitation_detected
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
