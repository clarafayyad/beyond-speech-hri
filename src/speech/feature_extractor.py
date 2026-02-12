import librosa
import numpy as np
import re

from src.speech.disfluency import get_disfluency
from src.speech.transcribe_audio import transcribe_audio
from src.speech.verbal_hesitation import get_verbal_hesitation_score

# -------- CONFIG --------
STOPWORDS = {"this", "that", "is", "but", "and", "so", "well", "already"}
NUMBER_TOKENS = {"1", "2", "3", "4", "5", "6", "7", "8",
                 "one", "two", "three", "four", "five", "six", "seven", "eight"}

SILENCE_THRESHOLD = 0.02  # RMS threshold for silence
MIN_SILENCE_DUR = 0.3  # seconds
HOP_LENGTH = 512
MAX_CLUE_LATENCY = 5.0
SPEECH_THRESHOLD = 0.04
MIN_SPEECH_DUR = 0.1

MAX_PAUSE = 2.0  # seconds
MAX_NUM_PAUSES = 5  # more than 5 long pauses saturates score


# ------------------------


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def get_clue_latency(y=None, sr=None, asr_words=None):
    """
    Compute clue latency as the first clue word (non-stopword before number).
    Falls back to RMS onset if ASR fails.
    """
    if asr_words:
        for i, w in enumerate(asr_words):
            token = w["word"].lower()
            if token in NUMBER_TOKENS:
                # Backtrack to first non-stopword
                for j in range(i - 1, -1, -1):
                    prev = asr_words[j]["word"].lower()
                    if prev not in STOPWORDS:
                        clue_word = asr_words[j]["word"]
                        number_word = w["word"]
                        print(f"Detected clue: '{clue_word} {number_word}' at {asr_words[j]['start']:.2f}s")
                        return asr_words[j]["start"]

    # Fallback to RMS speech onset
    if y is not None and sr is not None:
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        speech = rms > SPEECH_THRESHOLD
        min_frames = int(MIN_SPEECH_DUR * sr / HOP_LENGTH)
        for i in range(len(speech)):
            if speech[i] and i + min_frames < len(speech) and np.all(speech[i:i + min_frames]):
                return times[i]

    return 0.0


def get_clue_latency_score(y, sr=None, asr_words=None):
    latency = get_clue_latency(y, sr, asr_words)
    return clamp(latency / MAX_CLUE_LATENCY)


def get_pause_info(y, sr):
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

    silent = rms < SILENCE_THRESHOLD
    pause_durations = []
    current_pause = 0.0

    for i in range(1, len(silent)):
        if silent[i]:
            current_pause += times[i] - times[i - 1]
        else:
            if current_pause >= MIN_SILENCE_DUR:
                pause_durations.append(current_pause)
            current_pause = 0.0

    if current_pause >= MIN_SILENCE_DUR:
        pause_durations.append(current_pause)

    longest_pause = max(pause_durations) if pause_durations else 0.0
    return longest_pause, pause_durations


def get_pause_score(y, sr):
    longest_pause, pause_durations = get_pause_info(y, sr)
    total_pause_time = sum(pause_durations)

    # Normalize features
    longest_score = clamp(longest_pause / MAX_PAUSE)
    num_score = clamp(len(pause_durations) / MAX_NUM_PAUSES)

    # Weighted combination
    pause_score = 0.5 * longest_score + 0.5 * num_score
    return clamp(pause_score)


def get_speaking_score(y, sr, total_pause_time, asr_words=None):
    """
    Compute a normalized speaking score based on:
      1. Speaking ratio (time speaking vs total)
      2. Words per second (WPS) during clue

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sample rate.
    total_pause_time : float
        Sum of all detected pauses (seconds).
    asr_words : list of dicts, optional
        Word-level ASR output [{"word": str, "start": float, "end": float}, ...]

    Returns
    -------
    float
        Speaking score ∈ [0,1]. Higher → slower speech / more hesitation.
    """

    # --- 1. Speaking ratio ---
    total_duration = librosa.get_duration(y=y, sr=sr)
    speaking_time = max(0.0, total_duration - total_pause_time)
    speaking_ratio = speaking_time / total_duration if total_duration > 0 else 0.0
    ratio_score = 1.0 - speaking_ratio  # slower speaking → higher score
    ratio_score = clamp(ratio_score)

    # --- 2. Words per second (WPS) ---
    wps_score = 0.0
    if asr_words:
        # Clean ASR words
        cleaned_words = [w for w in asr_words if w["word"].lower() not in STOPWORDS]
        if cleaned_words:
            start = cleaned_words[0]["start"]
            end = cleaned_words[-1]["end"]
            duration = max(end - start, 0.01)  # avoid div by zero
            wps = len(cleaned_words) / duration
            MAX_WPS = 3.0  # words/sec that maps to 0 difficulty
            wps_score = clamp(1.0 - min(wps, MAX_WPS) / MAX_WPS)

    # --- 3. Combined score ---
    speaking_score = 0.5 * ratio_score + 0.5 * wps_score
    return speaking_score


def extract_speech_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y = librosa.util.normalize(y)

    transcript, asr_words = transcribe_audio(audio_path)

    longest_pause, pause_durations = get_pause_info(y, sr)
    total_pause_time = sum(pause_durations)

    return {
        "clue_latency_score": get_clue_latency_score(y, sr, asr_words),
        "pause_score": get_pause_score(y, sr),
        "verbal_hesitation_score": get_verbal_hesitation_score(transcript),
        "speaking_score": get_speaking_score(y, sr, total_pause_time, asr_words),
        "disfluency_score": get_disfluency(transcript)
    }


if __name__ == "__main__":
    features = extract_speech_features("test_audio/uncertain_clue_2.wav")
    print(features)
