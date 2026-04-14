"""Utility for selecting the most relevant pre-guess utterances.

The robot can generate several candidate utterances before making a guess
(e.g. continuity remark, confidence reaction, thinking filler).  Rather
than saying *all* of them — which slows down the interaction — this module
provides a priority-based selection function that keeps only the most
relevant ones.
"""

# Maximum number of utterances the robot says before displaying a guess.
MAX_PRE_GUESS_UTTERANCES = 1


def select_utterances(candidates, max_count=MAX_PRE_GUESS_UTTERANCES):
    """Pick the most relevant utterances from prioritised candidates.

    Parameters
    ----------
    candidates : list[tuple[int, str | None]]
        ``(priority, text)`` pairs.  Lower priority number means higher
        relevance.  Entries where *text* is ``None`` or empty are ignored.
    max_count : int
        Maximum number of utterances to return.

    Returns
    -------
    list[str]
        The selected utterance texts, ordered by relevance (most relevant
        first), containing at most *max_count* items.
    """
    valid = [(p, t) for p, t in candidates if t]
    valid.sort(key=lambda x: x[0])
    return [t for _, t in valid[:max_count]]
