import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(_HERE, "..", "logs")


def build_utterance_log_path(participant_id, log_dir=DEFAULT_LOG_DIR, now=None):
    """Return log path for utterances, grouped by participant and date."""
    if participant_id is None or str(participant_id).strip() == "":
        return None

    os.makedirs(log_dir, exist_ok=True)
    dt = now or datetime.now()
    date_stamp = dt.strftime("%Y%m%d")
    return os.path.join(log_dir, f"utterances_{participant_id}_{date_stamp}.txt")


def format_utterance_log_line(speaker, text, now=None):
    dt = now or datetime.now()
    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] {speaker}: {text}"
