from multimodal_perception.speech.feature_extractor import extract_speech_features


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_difficulty_score(features):
    """
    Weighted difficulty score ∈ [0, 1]

    Feature contributions:
      - Clue latency: strong signal
      - Pause score: moderate signal
      - Verbal hesitation: dominant signal
      - Disfluency: supporting signal
      - Speaking score: moderate signal (slow / hesitant speech)
    """
    score = 0.0

    latency_score = clamp(features.get("clue_latency_score", 0.0))
    pause_score = clamp(features.get("pause_score", 0.0))
    hesitation_score = clamp(features.get("verbal_hesitation_score", 0.0))
    disfluency_score = clamp(features.get("disfluency_score", 0.0))
    speaking_score = clamp(features.get("speaking_score", 0.0))

    # --- Core signals ---
    score += 0.30 * latency_score  # latency: strong indicator
    score += 0.20 * pause_score  # pauses: moderate

    # --- Verbal hesitation dominates disfluency ---
    score += 0.25 * hesitation_score  # hesitation: dominant
    score += 0.15 * disfluency_score  # disfluency: supporting

    # --- Speaking score (ratio + words/sec) ---
    score += 0.10 * speaking_score  # moderate contribution

    return clamp(score)


def label_difficulty(score):
    if score < 0.33:
        return "easy"
    elif score < 0.66:
        return "neutral"
    else:
        return "difficult"


def classify_clue_difficulty(audio_path):
    """
    End-to-end difficulty classification.

    extract_speech_features is expected to return:
        - hesitation_count
        - disfluency_score (optional, ∈ [0,1])
    """
    features = extract_speech_features(audio_path)
    score = compute_difficulty_score(features)
    label = label_difficulty(score)

    return {
        "difficulty_label": label,
        "difficulty_score": score,
        "features": features
    }


if __name__ == "__main__":
    result = classify_clue_difficulty("speech/test_audio/uncertain_clue_2.wav")
    print(result)
