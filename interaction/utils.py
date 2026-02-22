def parse_clue(clue_text: str):
    parts = clue_text.strip().split()
    if len(parts) < 2:
        raise ValueError("Clue must be a word followed by a number.")

    clue_word = " ".join(parts[:-1])
    num = int(parts[-1])
    return clue_word, num


def normalize_feedback(feedback: str):
    feedback = feedback.lower().strip()
    if feedback == "innocent":
        return "neutral"
    if feedback not in {"blue", "red", "neutral", "assassin"}:
        return None
    return feedback