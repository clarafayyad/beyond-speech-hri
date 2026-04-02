import librosa
import numpy as np
import pandas as pd

import feature_extractor


class ImportantFeaturesExtractor:
    """
    A feature extractor that processes recorded audio and extracts a set of important features relevant for confidence classification.
    It can optionally normalize features per-participant when provided with a small dataset (pandas DataFrame) for that participant.
    """

    def __init__(self, whisper, participant_id, participant_df: pd.DataFrame = None):
        self.whisper = whisper
        self.participant_id = participant_id
        # participant_stats: feature_name -> (mean, std)
        self.participant_stats = {}

        # TODO: load participant_df from dedicated calibration-phase folder
        if participant_df is not None:
            # compute statistics for this participant
            self._compute_participant_stats(participant_df)

    def _compute_participant_stats(self, df: pd.DataFrame):
        """Compute mean and std per numeric feature for the configured participant.
        The function only keeps numeric columns and excludes obvious ID/meta columns.
        """
        if "participant_id" not in df.columns:
            return

        # numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {"clue_id", "participant_id", "confidence", "difficulty"}
        features = [c for c in numeric_cols if c not in exclude_cols]

        df_p = df[df["participant_id"] == self.participant_id]
        if df_p.shape[0] == 0:
            # no data for this participant
            return

        for f in features:
            x = df_p[f].dropna()
            if x.size == 0:
                mean = 0.0
                std = 0.0
            else:
                mean = float(x.mean())
                std = float(x.std())
                if np.isnan(mean):
                    mean = 0.0
                if np.isnan(std):
                    std = 0.0

            self.participant_stats[f] = (mean, std)

    def update_participant_dataframe(self, df: pd.DataFrame):
        """Public method to (re)compute stats if a new participant dataframe becomes available."""
        self._compute_participant_stats(df)

    def _safe_zscore(self, value, mean, std):
        """Apply safe z-score: if std == 0 or nan, return value - mean (centering) instead of dividing."""
        if std == 0 or np.isnan(std):
            return value - mean
        return (value - mean) / std

    def extract(self, audio_path):
        """
        Extract a set of important features from the given audio file.

        Returns a dictionary which contains:
         - raw_<feature>: the raw computed value
         - <feature>: the per-participant normalized value (safe z-score)
         - <feature>_dev: the deviation (raw - participant_mean)

        If no participant stats are available for a feature, normalization falls back to centering (value - mean) with mean=0.
        """

        y, sr = feature_extractor.load_audio(audio_path)
        y = feature_extractor.trim_silence(y)
        y = feature_extractor.normalize_audio(y)
        y = feature_extractor.reduce_noise(y, sr)

        transcript, asr_words = self.whisper.transcribe_audio(audio_path)

        duration = librosa.get_duration(y=y, sr=sr)
        pause_count, _, pause_max = feature_extractor.extract_pause_features_vad(y, sr)
        _, pause_mid = feature_extractor.pause_position_features(asr_words)
        verbal_hesitation_count = feature_extractor.count_hesitation_words(transcript)
        speech_rate = feature_extractor.extract_speech_rate(transcript, duration)
        mfcc_features = feature_extractor.extract_mfcc_features(y, sr)
        # TODO: check mfcc attributes and verify mfcc_2_mean is the correct one to use here
        mfcc_2 = mfcc_features.get("mfcc_2_mean", 0)
        _, energy_std, energy_range, _, _ = feature_extractor.energy_features(y)
        _, _, hnr = feature_extractor.extract_voice_quality(audio_path)

        # raw features we compute
        raw_features = {
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

        result = {}

        # For each raw feature compute raw_, normalized (safe z-score) and dev
        for fname, raw_val in raw_features.items():
            mean, std = self.participant_stats.get(fname, (0.0, 0.0))
            dev = raw_val - mean
            z = self._safe_zscore(raw_val, mean, std)

            result[f'raw_{fname}'] = raw_val
            result[f'{fname}'] = z
            result[f'{fname}_dev'] = dev

        # keep some legacy keys that may be expected by downstream code
        # Fill any explicitly expected fields that were present before
        # ...existing code...

        # Build the final dictionary similar to earlier return structure but populated
        final = {
            'duration': result['duration'],
            'pause_max': result['pause_max'],
            'verbal_hesitation_count_dev': result.get('verbal_hesitation_count_dev', 0),
            'duration_dev': result.get('duration_dev', 0),
            'speech_rate': result['speech_rate'],
            'mfcc_2_mean': result['mfcc_2_mean'],
            'speech_rate_dev': result.get('speech_rate_dev', 0),
            'verbal_hesitation_count': result['verbal_hesitation_count'],
            'pause_count_dev': result.get('pause_count_dev', 0),
            'hnr_dev': result.get('hnr_dev', 0),
            'energy_range_dev': 0,
            'hnr': result['hnr'],
            'energy_std': result['energy_std'],
            'energy_std_dev': result.get('energy_std_dev', 0),
            'pause_mid_speech': result['pause_mid_speech'],
            # include raw versions in case caller wants them
            'raw_features': {k: v for k, v in result.items() if k.startswith('raw_')}
        }

        return final

