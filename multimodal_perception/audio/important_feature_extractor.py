import librosa
import numpy as np
import pandas as pd
import os

from multimodal_perception.audio import feature_extractor


# Fixed calibration folder (relative to this module)
_CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'calibration_phase'))


class ImportantFeaturesExtractor:
    """Extracts audio features."""

    def __init__(self, whisper):
        self.whisper = whisper

    def extract(self, audio_path: str) -> dict:
        """Extract features from `audio_path` and return raw base features only."""

        # load + preprocess
        y, sr = feature_extractor.load_audio(audio_path)
        y = feature_extractor.trim_silence(y)
        y = feature_extractor.normalize_audio(y)
        y = feature_extractor.reduce_noise(y, sr)

        # transcription
        transcript, asr_words = self.whisper.transcribe_audio(audio_path)

        # compute raw features
        duration = librosa.get_duration(y=y, sr=sr)
        pause_count, _, pause_max = feature_extractor.extract_pause_features_vad(y, sr)
        _, pause_mid = feature_extractor.pause_position_features(asr_words)
        verbal_hesitation_count = feature_extractor.count_hesitation_words(transcript)
        speech_rate = feature_extractor.extract_speech_rate(transcript, duration)
        mfcc_features = feature_extractor.extract_mfcc_features(y, sr)
        mfcc_2 = mfcc_features.get("mfcc_2_mean", 0)
        _, energy_std, energy_range, _, _ = feature_extractor.energy_features(y)
        _, _, hnr = feature_extractor.extract_voice_quality(audio_path)

        return {
            'transcript': transcript,
            'duration': duration,
            'pause_max': pause_max,
            'speech_rate': speech_rate,
            'mfcc_2_mean': mfcc_2,
            'verbal_hesitation_count': verbal_hesitation_count,
            'hnr': hnr,
            'energy_std': energy_std,
            'pause_mid_speech': pause_mid,
            'pause_count': pause_count,
        }