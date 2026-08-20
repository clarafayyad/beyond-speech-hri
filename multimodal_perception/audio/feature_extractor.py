import os
import re
import pandas as pd
import librosa
import whisper
import subprocess
import tempfile
import webrtcvad
from wordfreq import zipf_frequency
import parselmouth
import numpy as np
from scipy.stats import linregress

from multimodal_perception.audio.disfluency import DisfluencyDetector
from multimodal_perception.audio.transcribe_audio import WhisperTranscriber
from multimodal_perception.audio.verbal_hesitation import count_hesitation_words, FILLERS

AUDIO_FOLDER = "../../assets/audio/pilot"
INPUT_CSV = "../data/pilot.csv"
OUTPUT_CSV = "../data/audio_features.csv"

STOPWORDS = {"this", "that", "is", "but", "and", "so", "well", "already"}
NUMBER_TOKENS = {"1", "2", "3", "4", "5", "6", "7", "8",
                 "one", "two", "to", "too", "three", "four", "for", "five", "six", "seven", "eight"}

SILENCE_THRESHOLD = 0.02
MIN_SILENCE_DUR = 0.3
HOP_LENGTH = 512
SPEECH_THRESHOLD = 0.04
MIN_SPEECH_DUR = 0.1

# load whisper model
whisper = WhisperTranscriber()
disfluency_detector = DisfluencyDetector()


def load_audio(path):
    y, sr = librosa.load(path, sr=16000)
    return y, sr


def normalize_audio(y, target_db=-20):
    """ Loudness Normalization """
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        return y
    scalar = 10 ** (target_db / 20) / rms
    return y * scalar


def reduce_noise(y, sr):
    noise_sample = y[:int(0.5 * sr)]  # first 0.5s as noise
    noise_profile = np.mean(np.abs(noise_sample))
    y_denoised = np.where(np.abs(y) < noise_profile, 0, y)
    return y_denoised


def trim_silence(y):
    yt, _ = librosa.effects.trim(y, top_db=30)
    return yt


def convert_to_wav(input_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        tmp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return tmp_path


def extract_pause_features(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    threshold = np.percentile(rms, 20)

    silent = rms < threshold
    pauses = []

    current = 0
    for s in silent:
        if s:
            current += 1
        else:
            if current > 0:
                pauses.append(current)
                current = 0

    if len(pauses) == 0:
        return 0, 0, 0

    pauses = np.array(pauses) * (512 / sr)

    return len(pauses), pauses.mean(), pauses.max()


def extract_pause_features_vad(y, sr):
    vad = webrtcvad.Vad(2)

    frame_duration = 30  # ms
    frame_size = int(sr * frame_duration / 1000)

    audio_int16 = (y * 32768).astype(np.int16)
    bytes_audio = audio_int16.tobytes()

    step = frame_size * 2
    speech_flags = []

    for i in range(0, len(bytes_audio), step):
        frame = bytes_audio[i:i + step]
        if len(frame) < step:
            frame += b'\0' * (step - len(frame))
        speech_flags.append(vad.is_speech(frame, sr))

    frame_time = frame_duration / 1000.0

    pauses = []
    current = 0

    for is_speech in speech_flags:
        if not is_speech:
            current += 1
        else:
            if current > 0:
                pauses.append(current * frame_time)
                current = 0

    if len(pauses) == 0:
        return 0, 0, 0

    pauses = np.array(pauses)

    return len(pauses), pauses.mean(), pauses.max()


def pause_position_features(asr_words):
    if not asr_words or len(asr_words) < 2:
        return 0, 0

    pauses_before_clue = 0
    pauses_mid = 0

    for i in range(1, len(asr_words)):
        gap = asr_words[i]["start"] - asr_words[i - 1]["end"]

        if gap > 0.3:  # pause threshold
            # before clue word (first meaningful word)
            if i == 1:
                pauses_before_clue += 1
            else:
                pauses_mid += 1

    return pauses_before_clue, pauses_mid


def extract_speech_rate(transcript, duration):
    words = transcript.split()

    if duration == 0:
        return 0

    return len(words) / duration


def count_fillers(transcript):
    count = 0

    words = transcript.lower().split()

    for w in words:
        if w in FILLERS:
            count += 1

    return count


def repetition_count(transcript):
    words = transcript.lower().split()

    count = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            count += 1

    return count


def extract_pitch_features(audio_path, end_window=0.5):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()

    times = pitch.xs()
    frequencies = pitch.selected_array['frequency']
    voiced_mask = frequencies > 0

    if np.sum(voiced_mask) == 0:
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_slope': 0.0,
            'pitch_rise_end': 0.0,
            'pitch_range': 0.0,
            'pitch_p25': 0.0,
            'pitch_p75': 0.0
        }

    voiced_times = times[voiced_mask]
    voiced_freqs = frequencies[voiced_mask]

    # clean
    mask = (voiced_freqs > 50) & (voiced_freqs < 400)
    voiced_freqs = voiced_freqs[mask]
    voiced_times = voiced_times[mask]

    mean_pitch = np.mean(voiced_freqs)
    std_pitch = np.std(voiced_freqs)
    pitch_range = np.max(voiced_freqs) - np.min(voiced_freqs)

    p25 = np.percentile(voiced_freqs, 25)
    p75 = np.percentile(voiced_freqs, 75)

    slope, _, _, _, _ = linregress(voiced_times[:len(voiced_freqs)], voiced_freqs)

    end_mask = voiced_times > voiced_times[-1] - end_window
    pitch_end = np.mean(voiced_freqs[end_mask])
    pitch_rise_end = pitch_end - mean_pitch

    return mean_pitch, std_pitch, slope, pitch_rise_end, pitch_range, p25, p75


def extract_mfcc_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = mfcc_mean[i]
        features[f"mfcc_{i + 1}_std"] = mfcc_std[i]

    return features


def extract_voice_quality(audio_path):
    snd = parselmouth.Sound(audio_path)

    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_value = parselmouth.praat.call(hnr, "Get mean", 0, 0)

    return jitter, shimmer, hnr_value


def energy_features(y):
    rms = librosa.feature.rms(y=y)[0]
    return np.mean(rms), np.std(rms), np.max(rms) - np.min(rms), np.percentile(rms, 25), np.percentile(rms, 75)


def clue_word_frequency(clue):
    words = clue.lower().strip().split()

    if len(words) == 0:
        return 0

    clue_word = words[0]

    return zipf_frequency(clue_word, 'en')


def meta_comment_score(transcript):
    meta_comment_patterns = [
        r"\bI think\b",
        r"\bthis is tough\b",
        r"\blet'?s see\b",
        r"\bokay\b",
        r"\bconcentrate\b",
        r"\bI'm going to say\b",
        r"\bfocus\b",
        r"\bnot sure\b",
        r"\bshoot\b",
        r"\bthank god\b",
        r"\bI guess\b",
        r"\bI'm just going to\b"
    ]

    # Split transcript into sentences (rough approximation)
    sentences = re.split(r'[.!?]\s*', transcript.lower())

    # Count sentences that match meta-comment patterns
    meta_count = 0
    for sentence in sentences:
        for pattern in meta_comment_patterns:
            if re.search(pattern, sentence):
                meta_count += 1
                break  # Count each sentence only once

    total_sentences = len([s for s in sentences if s.strip()])
    if total_sentences == 0:
        return 0.0
    return meta_count / total_sentences


def get_clue_latencies(asr_words=None, y=None, sr=None):
    """
    Compute both:
    - clue_latency: start time of the clue (first meaningful word)
    - clue_number_latency: time from clue start to the number word

    Args:
        asr_words (list of dict): [{"word": str, "start": float, "end": float}, ...]
        y (np.ndarray, optional): audio signal (fallback if ASR not available)
        sr (int, optional): sampling rate of audio

    Returns:
        tuple: (clue_latency, clue_number_latency) in seconds
    """
    clue_start = None
    number_start = None

    if asr_words:
        for i, w in enumerate(asr_words):
            token = w["word"].lower()
            if token in NUMBER_TOKENS:
                number_start = w["start"]
                # Find clue start as last non-stopword before number
                for j in range(i - 1, -1, -1):
                    prev = asr_words[j]["word"].lower()
                    if prev not in STOPWORDS:
                        clue_start = asr_words[j]["start"]
                        break
                break
        if clue_start is None:
            # fallback: last word in ASR
            clue_start = asr_words[-1]["start"]

    # fallback to energy-based detection if ASR not provided
    elif y is not None and sr is not None:
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        speech = rms > SPEECH_THRESHOLD
        min_frames = int(MIN_SPEECH_DUR * sr / HOP_LENGTH)
        for i in range(len(speech)):
            if speech[i] and i + min_frames < len(speech) and all(speech[i:i + min_frames]):
                clue_start = times[i]
                break
        number_start = clue_start  # fallback, unknown number
    else:
        clue_start = 0.0
        number_start = 0.0

    clue_number_latency = (number_start - clue_start) if (clue_start is not None and number_start is not None) else 0.0
    return clue_start, clue_number_latency


def count_syllables(word):
    """Simple syllable estimation: count vowels groups in a word"""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_vowel:
            count += 1
        prev_char_vowel = is_vowel
    return max(1, count)


def get_speech_ratio_and_articulation(y, sr, transcript):
    audio_bytes = (y * 32768).astype(np.int16).tobytes()

    transcript_words = transcript.lower().strip().split()

    vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3
    frame_duration = 30  # ms
    frame_size = int(sr * frame_duration / 1000) * 2  # 16-bit samples
    speech_flags = []

    for i in range(0, len(audio_bytes), frame_size):
        frame = audio_bytes[i:i + frame_size]
        if len(frame) < frame_size:
            frame += b'\0' * (frame_size - len(frame))
        speech_flags.append(vad.is_speech(frame, sr))

    frame_time = frame_duration / 1000.0
    speech_time = sum(speech_flags) * frame_time
    total_time = len(y) / sr
    speech_ratio = speech_time / total_time if total_time > 0 else 0.0

    if transcript_words:
        total_syllables = sum(count_syllables(w) for w in transcript_words)
        articulation_rate = total_syllables / speech_time if speech_time > 0 else 0.0
    else:
        # fallback: assume 1 word/sec
        articulation_rate = len(y) / sr / speech_time if speech_time > 0 else 0.0

    return speech_ratio, articulation_rate


def process_clue(row):
    clue_id = row["clue_id"]
    clue = row["Clue"]

    m4a_file = os.path.join(AUDIO_FOLDER, f"clue_{clue_id:04d}.m4a")
    audio_file = convert_to_wav(m4a_file)

    y, sr = load_audio(audio_file)
    y = trim_silence(y)
    y = normalize_audio(y)
    y = reduce_noise(y, sr)

    # whisper transcription
    transcript, asr_words = whisper.transcribe_audio(audio_file)

    # features
    duration = librosa.get_duration(y=y, sr=sr)
    pause_count, pause_mean, pause_max = extract_pause_features_vad(y, sr)
    pause_before, pause_mid = pause_position_features(asr_words)
    speech_rate = extract_speech_rate(transcript, duration)
    filler_count = count_fillers(transcript)
    repetition = repetition_count(transcript)
    pitch_mean, pitch_std, pitch_slope, pitch_rise_end, pitch_range, pitch_p25, pitch_p75 = extract_pitch_features(
        audio_file)
    mfcc_features = extract_mfcc_features(y, sr)
    jitter, shimmer, hnr = extract_voice_quality(audio_file)
    energy_mean, energy_std, energy_range, energy_p25, energy_p75 = energy_features(y)
    word_freq = clue_word_frequency(clue)
    meta_score = meta_comment_score(transcript)
    clue_latency, clue_number_latency = get_clue_latencies(asr_words, y, sr)
    disfluency = disfluency_detector.get_disfluency(transcript)
    verbal_hesitation_count = count_hesitation_words(transcript)
    speech_ratio, articulation_rate = get_speech_ratio_and_articulation(y, sr, transcript)

    return {
        # === Metadata / identifiers ===
        "clue_id": clue_id,
        "confidence": row["Confidence"],
        "difficulty": row["Difficulty"],
        "transcript": transcript,
        "clue_word_frequency": word_freq,

        # === Timing / latency ===
        "duration": duration,
        "clue_latency": clue_latency,
        "clue_number_latency": clue_number_latency,
        "speech_rate": speech_rate,
        "speech_ratio": speech_ratio,
        "articulation_rate": articulation_rate,

        # === Speech fluency / disfluencies ===
        "pause_count": pause_count,
        "pause_mean": pause_mean,
        "pause_max": pause_max,
        "pause_before_clue": pause_before,
        "pause_mid_speech": pause_mid,
        "filler_count": filler_count,
        "repetition_count": repetition,
        "disfluency": disfluency,
        "verbal_hesitation_count": verbal_hesitation_count,
        "meta_comment_presence": meta_score,

        # === Prosody / pitch ===
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_slope": pitch_slope,
        "pitch_rise_end": pitch_rise_end,
        "pitch_range": pitch_range,
        "pitch_p25": pitch_p25,
        "pitch_p75": pitch_p75,
        **mfcc_features,

        # === Energy / voice quality ===
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_range": energy_range,
        "energy_p25": energy_p25,
        "energy_p75": energy_p75,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr
    }


def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []

    for _, row in df.iterrows():

        try:
            features = process_clue(row)
            rows.append(features)
            print("processed clue", row["clue_id"])

        except Exception as e:
            print("error with clue", row["clue_id"], e)

    out = pd.DataFrame(rows)

    out.to_csv(OUTPUT_CSV, index=False)

    print("saved features to", OUTPUT_CSV)


if __name__ == "__main__":
    main()
