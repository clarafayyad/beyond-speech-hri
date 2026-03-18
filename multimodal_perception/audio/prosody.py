import librosa
import numpy as np
from collections import deque

class ProsodyService:
    def __init__(self, smoothing_window=5, sr=16000):
        self.sr = sr
        self.buffer = deque(maxlen=smoothing_window)
        self.current_features = {"fluency":"unknown","arousal":"unknown"}

    def process_chunk(self, audio_chunk):
        """
        Process a chunk of audio and return coarse prosody cues:
        - fluency: hesitant vs fluent
        - arousal: low vs high
        """

        # --- Feature extraction ---
        rms = np.mean(librosa.feature.rms(y=audio_chunk)[0])
        rms_std = np.std(librosa.feature.rms(y=audio_chunk)[0])

        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            f0_mean = np.nanmean(f0)
            f0_std = np.nanstd(f0)
        except:
            f0_mean, f0_std = 0, 0

        # Speaking rate via onset detection
        onsets = librosa.onset.onset_detect(y=audio_chunk, sr=self.sr)
        rate = len(onsets) / (len(audio_chunk)/self.sr)

        # Silence fraction
        silence_thresh = 0.02
        silence_frac = np.mean(librosa.feature.rms(y=audio_chunk)[0] < silence_thresh)

        # Spectral flatness (breathiness / instability)
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio_chunk))

        print(f"RMS: {rms:.4f}, RMS_std: {rms_std:.4f}, F0_std: {f0_std:.2f}, Rate: {rate:.2f}, Silence: {silence_frac:.2f}, Flatness: {flatness:.2f}")

        # --- Heuristic scoring for fluency / hesitation ---
        hesitancy_score = 0

        if rms < 0.03 or rms_std > 0.02:
            hesitancy_score += 1
        if f0_std < 20:
            hesitancy_score += 1
        if rate < 2.5:
            hesitancy_score += 1
        if silence_frac > 0.2:
            hesitancy_score += 1
        if flatness > 0.3:
            hesitancy_score += 1

        fluency = "hesitant" if hesitancy_score >= 3 else "fluent"

        # --- Simple arousal estimation ---
        arousal = "high" if rms > 0.05 or f0_std > 30 else "low"

        # --- Temporal smoothing ---
        self.buffer.append({"fluency":fluency, "arousal":arousal})
        self.current_features = self._smoothed_features()

        return self.current_features["fluency"], self.current_features["arousal"]

    def _smoothed_features(self):
        # Majority vote over buffer
        fluency = max(set([b["fluency"] for b in self.buffer]), key=[b["fluency"] for b in self.buffer].count)
        arousal = max(set([b["arousal"] for b in self.buffer]), key=[b["arousal"] for b in self.buffer].count)
        return {"fluency":fluency, "arousal":arousal}
