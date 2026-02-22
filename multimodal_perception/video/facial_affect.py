import time
from collections import deque
from deepface import DeepFace

EMOTION_TO_VALENCE = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fear": "negative",
    "disgust": "negative"
}

class FacialAffectService:
    def __init__(self, analyze_every_n_frames=10, smoothing_window=5):
        self.frame_count = 0
        self.last_analysis_time = 0
        self.analyze_every_n_frames = analyze_every_n_frames
        self.valence_buffer = deque(maxlen=smoothing_window)
        self.current_valence = "neutral"

    def process_frame(self, frame):
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % self.analyze_every_n_frames != 0:
            return self.current_valence

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="retinaface"
            )

            dominant_emotion = result[0]["dominant_emotion"]
            valence = EMOTION_TO_VALENCE.get(dominant_emotion, "neutral")

            self.valence_buffer.append(valence)
            self.current_valence = self._smoothed_valence()

        except Exception as e:
            # Keep last known valence
            pass

        return self.current_valence

    def _smoothed_valence(self):
        # Majority vote over window
        return max(
            set(self.valence_buffer),
            key=self.valence_buffer.count
        )
