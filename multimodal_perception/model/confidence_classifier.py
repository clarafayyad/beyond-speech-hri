import os
import numpy as np
import pandas as pd

CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"

SELECTED_FEATURES = [
    'duration', 'pause_max', 'verbal_hesitation_count_dev', 'duration_dev',
    'speech_rate', 'mfcc_2_mean', 'speech_rate_dev', 'verbal_hesitation_count',
    'pause_count_dev', 'hnr_dev', 'energy_range_dev', 'hnr', 'energy_std',
    'energy_std_dev', 'pause_mid_speech',
]

PARTICIPANT_COL = 'participant_id'
BASE_FEATURES = sorted({f[:-4] if f.endswith('_dev') else f for f in SELECTED_FEATURES})
_CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'calibration_phase'))
LONG_DURATION_THRESHOLD = 12
LOW_HESITATION_THRESHOLD = 2
LOW_PAUSE_MAX_THRESHOLD = 2.5


class ConfidenceClassifier:
    def __init__(self, participant_id: str | None = None):
        self.W = np.array([
            [-0.724750756, -0.363689272, -0.224248714, 0.011265274, 0.203491125,
             0.128271115, 0.263719552, 0.480902402, -0.0164840947, -0.182924018,
             0.154083429, 0.712520228, 0.138211138, 0.00149581084, 0.238307982],

            [0.0943432199, 0.330132382, -0.403525833, 0.0636794746, -0.0457295757,
             -0.338656068, -0.00710226521, -0.367435712, -0.000203146175,
             -0.0532499204, 0.0703433529, -0.441640927, -0.164922016,
             0.000166715718, -0.140789667],

            [0.630407536, 0.0335568905, 0.627774547, -0.0749447486, -0.157761549,
             0.210384953, -0.256617287, -0.113466691, 0.0166872409,
             0.236173938, -0.224426782, -0.270879301, 0.0267108778,
             -0.00166252656, -0.0975183146]
        ])

        self.b = np.array([0.23688999, -0.31508903, 0.07819905])

        self.calibration = None
        self.fallback_to_global = True
        self._calib_folder = _CALIB_FOLDER
        if participant_id is not None:
            self._load_calibration_for_participant(participant_id)

    @staticmethod
    def _candidate_filenames_for_participant(pid: str) -> list:
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
            # Normalize group index types to strings so later lookups using
            # str(participant_id) won't raise KeyError when original index is int.
            try:
                means.index = means.index.map(str)
                stds.index = stds.index.map(str)
            except Exception:
                # If mapping fails for any reason, continue without crashing;
                # we'll handle missing keys safely below.
                pass
            global_means = df[present].mean()
            global_stds = df[present].std(ddof=0)
            calib = {'participants': {}, 'global': {}}
            for pid, row in means.iterrows():
                pid = str(pid)
                calib['participants'][pid] = {}
                for feat in present:
                    m = float(row[feat]) if not pd.isna(row[feat]) else 0.0
                    s = 0.0
                    # Access std safely; stds may not contain the participant key
                    # (or the column), so guard against KeyError and NaN.
                    if feat in stds.columns:
                        try:
                            s_val = stds.loc[pid, feat]
                            if not pd.isna(s_val):
                                s = float(s_val)
                        except (KeyError, TypeError):
                            # fallback to 0.0 when lookup fails
                            s = 0.0
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
    def _softmax(z):
        z = z - np.max(z)  # stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def probs(self, features: dict) -> np.ndarray:
        x = self._features_to_vector(features)
        logits = np.dot(self.W, x) + self.b  # shape (3,)
        return self._softmax(logits)

    @staticmethod
    def _none_to_zero(value):
        return 0 if value is None else value

    @staticmethod
    def _duration_only_low_confidence(features: dict, label: str) -> bool:
        """Detect when long duration is the only strong low-confidence cue.

        A low prediction is treated as duration-driven when duration is long
        but hesitation and pause length are still in non-low ranges.
        """
        if label != CONFIDENCE_LOW or not features:
            return False

        duration = ConfidenceClassifier._none_to_zero(features.get('duration'))
        hesitation_count = ConfidenceClassifier._none_to_zero(features.get('verbal_hesitation_count'))
        pause_max = ConfidenceClassifier._none_to_zero(features.get('pause_max'))

        return (
            duration > LONG_DURATION_THRESHOLD
            and hesitation_count < LOW_HESITATION_THRESHOLD
            and pause_max <= LOW_PAUSE_MAX_THRESHOLD
        )

    def classify(self, features: dict) -> tuple[np.ndarray, str]:
        """Return model probabilities and final confidence label.

        The returned label normally follows the max-probability class, but a
        low label can be adjusted to high/medium when the low prediction appears
        to be driven by duration alone.
        """
        probs = self.probs(features)
        labels = [CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM]
        label = labels[np.argmax(probs)]

        if self._duration_only_low_confidence(features, label):
            high_prob = probs[0]
            medium_prob = probs[2]
            label = CONFIDENCE_HIGH if high_prob >= medium_prob else CONFIDENCE_MEDIUM

        return probs, label
