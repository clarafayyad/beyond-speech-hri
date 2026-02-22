import re

LOW_HESITATION = 1      # filled pauses, thinking sounds
MEDIUM_HESITATION = 2   # uncertainty / hedging language
HIGH_HESITATION = 3     # explicit difficulty / risk statements

MAX_TIER_SUM = 8        # caps influence (≈ 2 strong signals)

FILLERS = {
    "uh": LOW_HESITATION,
    "um": LOW_HESITATION,
    "uhm": LOW_HESITATION,
    "erm": LOW_HESITATION,
    "er": LOW_HESITATION,
    "hmm": LOW_HESITATION,
    "mmm": LOW_HESITATION,
}

UNCERTAINTY_PHRASES = {
    "maybe": MEDIUM_HESITATION,
    "i think": MEDIUM_HESITATION,
    "kind of": MEDIUM_HESITATION,
    "sort of": MEDIUM_HESITATION,
    "probably": MEDIUM_HESITATION,
    "not sure": MEDIUM_HESITATION,
    "i guess": MEDIUM_HESITATION,
    "hopefully": MEDIUM_HESITATION,
    "i hope": MEDIUM_HESITATION,
}

META_DIFFICULTY_PHRASES = {
    # Direct difficulty
    "this is hard": HIGH_HESITATION,
    "this is difficult": HIGH_HESITATION,
    "this is tricky": HIGH_HESITATION,

    # Risk awareness
    "this is dangerous": HIGH_HESITATION,
    "this is risky": HIGH_HESITATION,
    "this might be risky": HIGH_HESITATION,
    "this is ambiguous": HIGH_HESITATION,
    "ambiguous": HIGH_HESITATION,
    "not clear": HIGH_HESITATION,

    # Apologies / hedging responsibility
    "sorry": HIGH_HESITATION,
    "i apologize": HIGH_HESITATION,

    # Game-specific uncertainty
    "i don't love this": HIGH_HESITATION,
    "this is a stretch": HIGH_HESITATION,
    "this is a reach": HIGH_HESITATION,
    "i'm not confident": HIGH_HESITATION,

    # Assassin awareness
    "watch out for the assassin": HIGH_HESITATION,
    "watch out": HIGH_HESITATION,
    "careful": HIGH_HESITATION,
}

STRESS_WORDS = {
    "fuck": MEDIUM_HESITATION,
    "shit": MEDIUM_HESITATION,
    "damn": LOW_HESITATION,
}


def get_verbal_hesitation_score(transcript):
    """
    Returns a normalized hesitation score ∈ [0, 1]
    Tier-based, Codenames-specific, interpretable.
    """
    if not transcript:
        return 0.0

    text = transcript.lower()
    tokens = re.findall(r"\b\w+\b", text)

    score = 0

    # Token-level fillers
    for token in tokens:
        score += FILLERS.get(token, 0)

    # Phrase-level uncertainty
    for phrase, tier in UNCERTAINTY_PHRASES.items():
        if phrase in text:
            score += tier

    # Strong meta-statements
    for phrase, tier in META_DIFFICULTY_PHRASES.items():
        if phrase in text:
            score += tier

    # Stress words
    for phrase, tier in STRESS_WORDS.items():
        if phrase in text:
            score += tier

    # Normalize + cap
    return min(1.0, score / MAX_TIER_SUM)
