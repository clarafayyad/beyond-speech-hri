import librosa
import numpy as np
import pandas as pd
import os

import feature_extractor


# Fixed calibration folder (relative to this module)
_CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'calibration-phase'))


class ImportantFeaturesExtractor:
    """Extracts audio features and normalizes them per-participant using calibration data.

    Behavior:
    - The constructor accepts a transcriber and a participant_id only.
    - Calibration data is discovered automatically using a fixed convention:
        1) if env var CALIBRATION_DIR points to a CSV file -> use it
        2) else look for a file named one of: f"participant_{id}.csv", f"{id}.csv", f"calibration_{id}.csv" inside `_CALIB_FOLDER`
    - At extraction time the extractor computes per-feature mean/std from: calibration rows for the participant (if present) + accumulated live rows + current sample.
    - The current sample is normalized using those stats and appended to the in-session live buffer.
    """

    def __init__(self, whisper, participant_id: str):
        self.whisper = whisper
        self.participant_id = str(participant_id)

        # calibration raw dataframe (if available). Do not precompute stats here.
        self._cal_df = self._load_calibration_for_participant()

        # live in-session numeric rows for this participant (pandas DataFrame)
        self._live_df = pd.DataFrame()

        # current participant stats cached after last extract
        self.participant_stats = {}

    def _load_calibration_for_participant(self) -> pd.DataFrame | None:
        """Locate and load a single CSV for this participant using a simple naming convention."""
        # priority 1: explicit file via env var
        env = os.environ.get('CALIBRATION_DIR')
        if env:
            env = os.path.expanduser(env)
            if os.path.isfile(env) and env.lower().endswith('.csv'):
                try:
                    return pd.read_csv(env)
                except Exception:
                    return None
            # if env is a directory, fallthrough to look for participant file inside it
            if os.path.isdir(env):
                for name in (f"participant_{self.participant_id}.csv", f"{self.participant_id}.csv", f"calibration_{self.participant_id}.csv"):
                    path = os.path.join(env, name)
                    if os.path.isfile(path):
                        try:
                            return pd.read_csv(path)
                        except Exception:
                            return None
        # priority 2: dedicated folder with strict filenames
        for name in (f"participant_{self.participant_id}.csv", f"{self.participant_id}.csv", f"calibration_{self.participant_id}.csv"):
            path = os.path.join(_CALIB_FOLDER, name)
            if os.path.isfile(path):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return None
        return None

    def load_calibration_file(self, file_path: str) -> pd.DataFrame:
        """Load a single calibration CSV (no side effects).
        Use `update_participant_dataframe()` to set it as active calibration data.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        return pd.read_csv(file_path)

    def update_participant_dataframe(self, df: pd.DataFrame):
        """Set or replace the calibration dataframe for this participant (no precomputation).
        Stats are always computed at extraction time by combining calibration + live + current sample.
        """
        self._cal_df = df
        self.participant_stats = {}

    def _safe_zscore(self, value, mean, std):
        if std == 0 or np.isnan(std):
            return value - mean
        return (value - mean) / std

    def extract(self, audio_path: str) -> dict:
        """Extract features from `audio_path`, normalize them per-participant using calibration + live data.

        Returns a dict with raw_<feature>, <feature> (normalized), and <feature>_dev keys.
        """
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

        raw = {
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

        # current numeric row
        current_df = pd.DataFrame([raw]).apply(pd.to_numeric, errors='coerce')

        # build combined data: calibration (participant filtered if applicable) + live + current
        frames = []
        if self._cal_df is not None:
            cal = self._cal_df
            cal_p = cal[cal['participant_id'] == self.participant_id] if 'participant_id' in cal.columns else cal
            cols = [c for c in raw.keys() if c in cal_p.columns]
            if cols:
                cal_sel = cal_p[cols].copy()
                cal_sel = cal_sel.apply(pd.to_numeric, errors='coerce')
                frames.append(cal_sel)
        if not self._live_df.empty:
            frames.append(self._live_df)
        frames.append(current_df)

        combined = pd.concat(frames, ignore_index=True, sort=False)

        # compute stats per feature present in combined
        for f in raw.keys():
            if f in combined.columns:
                x = combined[f].dropna()
                if x.size:
                    mean = float(x.mean())
                    std = float(x.std())
                    if np.isnan(mean):
                        mean = 0.0
                    if np.isnan(std):
                        std = 0.0
                else:
                    mean, std = 0.0, 0.0
            else:
                mean, std = 0.0, 0.0
            self.participant_stats[f] = (mean, std)

        # compute normalized outputs
        result = {}
        for fname, raw_val in raw.items():
            mean, std = self.participant_stats.get(fname, (0.0, 0.0))
            dev = raw_val - mean
            z = self._safe_zscore(raw_val, mean, std)
            result[f'raw_{fname}'] = raw_val
            result[f'{fname}'] = z
            result[f'{fname}_dev'] = dev

        # append current numeric row to live buffer
        row = current_df.copy()
        if self._live_df.empty:
            self._live_df = row.reset_index(drop=True)
        else:
            for c in row.columns:
                if c not in self._live_df.columns:
                    self._live_df[c] = np.nan
            self._live_df = pd.concat([self._live_df, row], ignore_index=True, sort=False)

        # build legacy-friendly final dict
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
            'raw_features': {k: v for k, v in result.items() if k.startswith('raw_')}
        }

        return final

