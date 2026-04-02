import math
import os
import numpy as np
import pandas as pd

CONFIDENCE_LEVELS = ["low", "medium", "high"]

SELECTED_FEATURES = [
    'duration', 'pause_max', 'verbal_hesitation_count_dev', 'duration_dev',
    'speech_rate', 'mfcc_2_mean', 'speech_rate_dev', 'verbal_hesitation_count',
    'pause_count_dev', 'hnr_dev', 'energy_range_dev', 'hnr', 'energy_std',
    'energy_std_dev', 'pause_mid_speech',
]

DEFAULT_WEIGHTS = {
    'duration': -0.724751, 'pause_max': -0.363689,
    'verbal_hesitation_count_dev': -0.224249, 'duration_dev': 0.011265,
    'speech_rate': 0.203491, 'mfcc_2_mean': 0.128271, 'speech_rate_dev': 0.263720,
    'verbal_hesitation_count': 0.480902, 'pause_count_dev': -0.016484,
    'hnr_dev': -0.182924, 'energy_range_dev': 0.154083, 'hnr': 0.712520,
    'energy_std': 0.138211, 'energy_std_dev': 0.001496, 'pause_mid_speech': 0.238308,
}

PARTICIPANT_COL = 'participant_id'
BASE_FEATURES = sorted({f[:-4] if f.endswith('_dev') else f for f in SELECTED_FEATURES})
_CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'calibration-phase'))


class ConfidenceClassifier:
    def __init__(self, participant_id: str | None = None):
        self.weights = DEFAULT_WEIGHTS.copy()
        self.intercept = 0.0
        for f in SELECTED_FEATURES:
            self.weights.setdefault(f, 0.0)
        self._weight_vector = np.array([self.weights[f] for f in SELECTED_FEATURES], dtype=float)
        self.calibration = None
        self.fallback_to_global = True
        self._calib_folder = _CALIB_FOLDER
        if participant_id is not None:
            self._load_calibration_for_participant(participant_id)

    def _candidate_filenames_for_participant(self, pid: str) -> list:
        return [f"participant_{pid}.csv", f"{pid}.csv", f"calibration_{pid}.csv"]

    def _load_calibration_for_participant(self, participant_id: str):
        pid = str(participant_id)
        for fn in self._candidate_filenames_for_participant(pid):
            path = os.path.join(self._calib_folder, fn)
            if os.path.isfile(path):
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if PARTICIPANT_COL in df.columns:
                    df_p = df[df[PARTICIPANT_COL].astype(str) == pid]
                    if df_p.empty:
                        self._load_calibration_from_df(df)
                    else:
                        self._load_calibration_from_df(df_p)
                else:
                    self._load_calibration_from_df(df, participant_id=pid)
                return
        return

    def load_calibration_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self._load_calibration_from_df(df)

    def _load_calibration_from_df(self, df: pd.DataFrame, participant_id: str | None = None):
        if PARTICIPANT_COL in df.columns and participant_id is None:
            present = [f for f in BASE_FEATURES if f in df.columns]
            grp = df.groupby(PARTICIPANT_COL)[present]
            means = grp.mean()
            stds = grp.std(ddof=0)
            global_means = df[present].mean()
            global_stds = df[present].std(ddof=0)
            calib = {'participants': {}, 'global': {}}
            for pid, row in means.iterrows():
                pid = str(pid)
                calib['participants'][pid] = {}
                for feat in present:
                    m = float(row[feat]) if not pd.isna(row[feat]) else 0.0
                    s = float(stds.loc[pid, feat]) if (feat in stds.columns and not pd.isna(stds.loc[pid, feat])) else 0.0
                    calib['participants'][pid][feat] = {'mean': m, 'std': s}
            for feat in present:
                gm = float(global_means[feat]) if not pd.isna(global_means[feat]) else 0.0
                gs = float(global_stds[feat]) if not pd.isna(global_stds[feat]) else 0.0
                calib['global'][feat] = {'mean': gm, 'std': gs}
            self.calibration = calib
            return
        present = [f for f in BASE_FEATURES if f in df.columns]
        if not present:
            return
        means = df[present].mean()
        stds = df[present].std(ddof=0)
        pid = str(participant_id) if participant_id is not None else 'unknown'
        calib = {'participants': {}, 'global': {}}
        calib['participants'][pid] = {}
        for feat in present:
            m = float(means[feat]) if not pd.isna(means[feat]) else 0.0
            s = float(stds[feat]) if not pd.isna(stds[feat]) else 0.0
            calib['participants'][pid][feat] = {'mean': m, 'std': s}
            calib['global'][feat] = {'mean': m, 'std': s}
        self.calibration = calib

    def _get_stats_for_participant(self, participant_id):
        if self.calibration is None:
            return None
        pid = str(participant_id) if participant_id is not None else None
        if pid in self.calibration['participants']:
            return self.calibration['participants'][pid]
        return self.calibration['global'] if self.fallback_to_global else None

    def _prepare_features_from_calibration(self, features: dict) -> dict:
        if self.calibration is None:
            return dict(features)
        pid = features.get(PARTICIPANT_COL) or features.get('participant')
        stats = self._get_stats_for_participant(pid)
        if stats is None:
            return dict(features)
        out = dict(features)
        for feat in BASE_FEATURES:
            s = stats.get(feat)
            if not s:
                continue
            raw = out.get(feat, 0.0)
            try:
                raw = float(raw)
            except Exception:
                raw = 0.0
            mean = s.get('mean', 0.0)
            std = s.get('std', 0.0)
            dev = raw - mean
            val = dev / std if (std and not np.isnan(std) and std != 0.0) else dev
            out[feat] = val
            out[f"{feat}_dev"] = dev
        return out

    def _features_to_vector(self, features: dict) -> np.ndarray:
        feats = self._prepare_features_from_calibration(features)
        vec = np.zeros(len(SELECTED_FEATURES), dtype=float)
        for i, f in enumerate(SELECTED_FEATURES):
            v = feats.get(f)
            if v is None and f.startswith('raw_'):
                alt = f[4:]
                v = feats.get(alt)
            try:
                vec[i] = float(v) if v is not None else 0.0
            except Exception:
                vec[i] = 0.0
        return vec

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)

    def score(self, features: dict) -> float:
        x = self._features_to_vector(features)
        return float(self._sigmoid(float(np.dot(self._weight_vector, x) + self.intercept)))

    def classify(self, features: dict) -> float:
        return self.score(features)

    def classify_label(self, features: dict, thresholds=(0.33, 0.66)) -> str:
        s = self.score(features)
        if s < thresholds[0]:
            return CONFIDENCE_LEVELS[0]
        if s >= thresholds[1]:
            return CONFIDENCE_LEVELS[2]
        return CONFIDENCE_LEVELS[1]
