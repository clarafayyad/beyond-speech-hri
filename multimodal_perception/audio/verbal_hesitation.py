import re

FILLERS = {
        "uh", "um", "hmm", "ah", "uhmmmmmm", "uhm", "uhmm", "uhmmmm", "uhmmmmm", "hm", "hmmm", "hmmmm", "...",
        # original
        "okay", "alright", "oh", "mm", "mhmmm", "mm-hmm",  # basic hesitations
        "you know", "actually", "let's see", "so", "then", "right", "all right", "oh my god", "yeah", "okay, okay",
        "let me see", "i'll say", "i'm going to say", "i think", "i'm just going to", "i'm not at all", "shoot",
        "great", "focus",
        "huh", "er", "eh", "wow", "gee", "yikes", "oops", "fuck", "shit"  # additional interjections
                                                                  "well", "like", "kind of", "sort of", "I guess",
        "maybe", "possibly", "probably", "I mean",  # thinking fillers
        "let's do", "let's go", "alright then", "okay then", "oh no", "oh yeah", "oh okay",  # discourse markers
        "hmmm", "mmkay", "mmhmm", "mmm", "aha", "woah",  # sound-based fillers
        "you might", "so yeah", "you see", "so uh", "and then", "then uh"  # mixed hesitation phrases
}

UNCERTAINTY_PHRASES = {
    "maybe", "i think", "kind of", "sort of", "probably",
    "not sure", "i guess", "hopefully", "i hope"
}

META_DIFFICULTY_PHRASES = {
    "this is hard",  "this is hard", "this is difficult", "this is tricky", "hard", "tough", "difficult", "tricky",
    "this is dangerous", "this is risky", "this might be risky", "dangerous", "risky",
    "this is ambiguous", "ambiguous", "not clear",
    "sorry", "i apologize",
    "i don't love this", "this is a stretch", "this is a reach",
    "i'm not confident",
    "watch out for the assassin", "watch out", "careful",
}

STRESS_WORDS = {
    "fuck", "shit", "damn"
}


def contains_hesitation_trigger(transcript: str) -> bool:
    """Return True if the transcript contains a stress word or meta-difficulty
    phrase that warrants a short acknowledgment from the robot while waiting
    for the clue.

    Examples that return True: "shit", "that's difficult", "this is hard".
    """
    if not transcript:
        return False
    text = transcript.lower()
    for word in STRESS_WORDS:
        if word in text:
            return True
    for phrase in META_DIFFICULTY_PHRASES:
        if phrase in text:
            return True
    return False


def count_hesitation_words(transcript):
    """
    Returns the total number of hesitation words/phrases in the transcript.
    """
    if not transcript:
        return 0

    text = transcript.lower()
    tokens = re.findall(r"\b\w+\b", text)

    count = 0

    # Count fillers (token-level)
    for token in tokens:
        if token in FILLERS:
            count += 1

    # Count phrase-level matches (can count multiple occurrences)
    for phrase in UNCERTAINTY_PHRASES:
        count += text.count(phrase)

    for phrase in META_DIFFICULTY_PHRASES:
        count += text.count(phrase)

    for phrase in STRESS_WORDS:
        count += text.count(phrase)

    return count