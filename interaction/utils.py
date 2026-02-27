import re

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "to": 2,
    "too": 2,
    "three": 3,
    "four": 4,
    "for": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

STOPWORDS = {
    "uh", "um", "uhm", "erm",
    "ok", "okay", "okk",
    "let", "me", "see",
    "this", "is", "so", "very", "really",
    "hard", "wow",
    "bla", "blah",
    "my", "the", "a", "an",
    "clue", "would", "be", "i", "say"
}

MAX_GUESS_NUMBER = 8

def parse_clue(clue_text: str):
    if not clue_text or not clue_text.strip():
        raise ValueError("Empty clue.")

    # Normalize
    text = clue_text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]

    # 1) Find last number
    number = None
    number_idx = None

    for i, tok in enumerate(tokens):
        if tok.isdigit():
            number = int(tok)
            number_idx = i
        elif tok in NUMBER_WORDS:
            number = NUMBER_WORDS[tok]
            number_idx = i

    if number is None or number <= 0 or number > MAX_GUESS_NUMBER:
        raise ValueError("No valid number found.")

    # 2) Find nearest content word BEFORE the number
    clue_word = None
    for j in range(number_idx - 1, -1, -1):
        tok = tokens[j]
        if tok not in STOPWORDS:
            clue_word = tok
            break

    if clue_word is None:
        raise ValueError("No clue word found before number.")

    return clue_word, number
