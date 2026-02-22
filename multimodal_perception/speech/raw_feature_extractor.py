import os
import csv
import librosa
from multimodal_perception.speech.disfluency import get_disfluency
from multimodal_perception.speech.transcribe_audio import transcribe_audio
from multimodal_perception.speech.verbal_hesitation import get_verbal_hesitation_score

# -------- CONFIG --------
STOPWORDS = {"this", "that", "is", "but", "and", "so", "well", "already"}
NUMBER_TOKENS = {"1", "2", "3", "4", "5", "6", "7", "8",
                 "one", "two", "three", "four", "five", "six", "seven", "eight"}

SILENCE_THRESHOLD = 0.02
MIN_SILENCE_DUR = 0.3
HOP_LENGTH = 512
SPEECH_THRESHOLD = 0.04
MIN_SPEECH_DUR = 0.1

# ------------------------

def get_clue_latency(y=None, sr=None, asr_words=None):
    if asr_words:
        for i, w in enumerate(asr_words):
            token = w["word"].lower()
            if token in NUMBER_TOKENS:
                for j in range(i - 1, -1, -1):
                    prev = asr_words[j]["word"].lower()
                    if prev not in STOPWORDS:
                        return asr_words[j]["start"]
    if y is not None and sr is not None:
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        speech = rms > SPEECH_THRESHOLD
        min_frames = int(MIN_SPEECH_DUR * sr / HOP_LENGTH)
        for i in range(len(speech)):
            if speech[i] and i + min_frames < len(speech) and all(speech[i:i + min_frames]):
                return times[i]
    return 0.0


def get_pause_info(y, sr):
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=HOP_LENGTH)
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
    return pause_durations, longest_pause


def get_words_per_second(asr_words):
    if not asr_words:
        return 0.0
    cleaned_words = [w for w in asr_words if w["word"].lower() not in STOPWORDS]
    if not cleaned_words:
        return 0.0
    start = cleaned_words[0]["start"]
    end = cleaned_words[-1]["end"]
    duration = max(end - start, 0.01)
    return len(cleaned_words) / duration


def extract_raw_speech_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y = librosa.util.normalize(y)
    transcript, asr_words = transcribe_audio(audio_path)
    clue_latency = get_clue_latency(y, sr, asr_words)
    pause_durations, longest_pause = get_pause_info(y, sr)
    total_pause_time = sum(pause_durations)
    total_duration = librosa.get_duration(y=y, sr=sr)
    speaking_time = max(0.0, total_duration - total_pause_time)
    wps = get_words_per_second(asr_words)
    verbal_hesitations = get_verbal_hesitation_score(transcript)
    disfluencies = get_disfluency(transcript)

    return {
        "audio_file": os.path.basename(audio_path),
        "clue_latency_sec": clue_latency,
        "pause_durations_sec": pause_durations,
        "longest_pause_sec": longest_pause,
        "total_pause_time_sec": total_pause_time,
        "total_duration_sec": total_duration,
        "speaking_time_sec": speaking_time,
        "words_per_second": wps,
        "verbal_hesitation_count": verbal_hesitations,
        "disfluency_count": disfluencies
    }


def process_audio_folder(folder_path, output_csv="../data/audio_features.csv"):
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    all_features = []

    for wav_file in wav_files:
        path = os.path.join(folder_path, wav_file)
        print(f"Processing: {wav_file}")
        features = extract_raw_speech_features(path)
        # Flatten pause durations to string for CSV
        features["pause_durations_sec"] = ";".join(f"{p:.3f}" for p in features["pause_durations_sec"])
        all_features.append(features)

    # Write to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_features[0].keys())
        writer.writeheader()
        writer.writerows(all_features)

    print(f"CSV saved to {output_csv}")


if __name__ == "__main__":
    folder_path = "../../assets/audio"
    process_audio_folder(folder_path)
