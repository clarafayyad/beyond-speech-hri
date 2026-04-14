"""Experiment CSV logger – records per-turn measures for every game."""

import csv
import json
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(_HERE, "..", "logs")

# Column order in the output CSV
FIELDNAMES = [
    "participant_id",
    "condition",
    "turn",
    "board",
    "key_map",
    "clue_word",
    "clue_number",
    "features",
    "confidence_level",
    "guesses",
    "outcomes",
    "score",
    "turn_duration_s",
]


class ExperimentLogger:
    """Append one row per turn to a participant-specific CSV file.

    Parameters
    ----------
    participant_id : str
        Identifier for the current participant.
    condition : bool
        Whether the experiment uses the adaptive condition (``True``) or
        baseline (``False``).  Stored as ``"adaptive"`` / ``"baseline"``.
    board : list[str]
        The game board (list of card filenames).
    key_map : dict | list | None
        The colour-assignment map for the board.
    log_dir : str
        Directory where the CSV file is written.
    """

    def __init__(self, participant_id, is_adaptive, board, key_map=None,
                 log_dir=DEFAULT_LOG_DIR):
        self.participant_id = participant_id
        self.condition = "adaptive" if condition else "baseline"
        self.board = board
        self.key_map = key_map

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            log_dir, f"experiment_{participant_id}_{timestamp}.csv"
        )

        # Write header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    def log_turn(self, turn, clue_word, clue_number, features, confidence_level,
                 guesses, outcomes, score, turn_duration_s):
        """Append a single turn row to the CSV.

        Parameters
        ----------
        turn : int
            Zero-based turn index.
        clue_word : str
            The clue word given by the spymaster.
        clue_number : int
            Number of guesses allowed for the clue.
        features : dict | None
            Extracted audio features (serialised as JSON string).
        confidence_level : str | None
            Inferred confidence level (``"low"`` / ``"medium"`` / ``"high"``).
        guesses : list[str]
            Card names guessed this turn, in order.
        outcomes : list[str]
            Corresponding colour outcome for each guess (``"blue"``, ``"red"``,
            ``"neutral"``, ``"assassin"``).
        score : int
            Number of blue (correct) guesses this turn.
        turn_duration_s : float
            Wall-clock duration of the turn in seconds.
        """
        row = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "turn": turn,
            "board": json.dumps(self.board),
            "key_map": json.dumps(self.key_map) if self.key_map is not None else "",
            "clue_word": clue_word,
            "clue_number": clue_number,
            "features": json.dumps(features) if features else "",
            "confidence_level": confidence_level or "",
            "guesses": json.dumps(guesses),
            "outcomes": json.dumps(outcomes),
            "score": score,
            "turn_duration_s": round(turn_duration_s, 2),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)
