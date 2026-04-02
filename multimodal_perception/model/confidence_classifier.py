import math
import numpy as np

CONFIDENCE_LEVELS = ["low", "medium", "high"]

# Features (order must match the one used when training / extracting coefficients)
SELECTED_FEATURES = [
    'duration',
    'pause_max',
    'verbal_hesitation_count_dev',
    'duration_dev',
    'speech_rate',
    'mfcc_2_mean',
    'speech_rate_dev',
    'verbal_hesitation_count',
    'pause_count_dev',
    'hnr_dev',
    'energy_range_dev',
    'hnr',
    'energy_std',
    'energy_std_dev',
    'pause_mid_speech',
]

# Coefficients extracted from clean_model_confidence.ipynb (clf.coef_[0])
# Feature -> coefficient
DEFAULT_WEIGHTS = {
    'duration': -0.724751,
    'pause_max': -0.363689,
    'verbal_hesitation_count_dev': -0.224249,
    'duration_dev': 0.011265,
    'speech_rate': 0.203491,
    'mfcc_2_mean': 0.128271,
    'speech_rate_dev': 0.263720,
    'verbal_hesitation_count': 0.480902,
    'pause_count_dev': -0.016484,
    'hnr_dev': -0.182924,
    'energy_range_dev': 0.154083,
    'hnr': 0.712520,
    'energy_std': 0.138211,
    'energy_std_dev': 0.001496,
    'pause_mid_speech': 0.238308,
}


class ConfidenceClassifier:
    """
    A simple linear scorer that computes a numeric confidence score from a feature
    dictionary using pre-extracted weights (from a trained logistic regression).

    The classifier computes:
        s = w^T x + b
    and returns sigmoid(s) in [0, 1] as the numeric confidence score. By
    default `b` (intercept) is 0.0 because the notebook output did not capture
    the intercept explicitly here; you can pass a custom intercept if available.

    Methods
    -------
    - score(features): returns a float in [0,1]
    - classify(features): returns the numeric score (keeps earlier name)
    - classify_label(features, thresholds=(0.33, 0.66)): optional helper to get
      a discrete label ('low','medium','high') using simple thresholds.
    """

    def __init__(self, weights: dict = None, intercept: float = 0.0):
        # use the default weights if none provided
        self.weights = DEFAULT_WEIGHTS.copy() if weights is None else dict(weights)
        self.intercept = float(intercept)

        # Ensure all selected features have an entry in weights (missing -> 0)
        for f in SELECTED_FEATURES:
            if f not in self.weights:
                self.weights[f] = 0.0

        # Build ordered weight vector for fast dot products
        self._weight_vector = np.array([self.weights[f] for f in SELECTED_FEATURES], dtype=float)

    def _features_to_vector(self, features: dict) -> np.ndarray:
        """Turn a feature dict into a numeric vector aligned with SELECTED_FEATURES."""
        vec = np.zeros(len(SELECTED_FEATURES), dtype=float)
        for i, f in enumerate(SELECTED_FEATURES):
            # allow nested/raw keys like 'raw_duration' or accept exact match
            if f in features:
                try:
                    vec[i] = float(features[f])
                except Exception:
                    vec[i] = 0.0
            else:
                # try common alternatives
                alt = f
                if f.startswith('raw_') and f[4:] in features:
                    alt = f[4:]
                if alt in features:
                    try:
                        vec[i] = float(features[alt])
                    except Exception:
                        vec[i] = 0.0
                else:
                    # Some code paths store dev fields or differently named keys; default 0
                    vec[i] = 0.0
        return vec

    @staticmethod
    def _sigmoid(x: float) -> float:
        # numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    def score(self, features: dict) -> float:
        """Compute a numeric confidence score in [0,1] for the provided features.

        Parameters
        ----------
        features : dict
            Mapping from feature name to numeric value. Missing features are
            treated as 0.

        Returns
        -------
        float
            A score between 0 and 1 computed as sigmoid(w^T x + intercept).
        """
        x = self._features_to_vector(features)
        linear = float(np.dot(self._weight_vector, x) + self.intercept)
        return float(self._sigmoid(linear))

    # keep the old method name but change its behavior to return numeric score
    def classify(self, features: dict) -> float:
        """Return a numeric confidence score (0..1) for compatibility.

        If you need a discrete label, use `classify_label`.
        """
        return self.score(features)

    def classify_label(self, features: dict, thresholds=(0.33, 0.66)) -> str:
        """Map numeric score to a discrete label using thresholds.

        thresholds : tuple(low_thresh, high_thresh)
            score < low_thresh => 'low'
            low_thresh <= score < high_thresh => 'medium'
            score >= high_thresh => 'high'
        """
        s = self.score(features)
        low_t, high_t = thresholds
        if s < low_t:
            return CONFIDENCE_LEVELS[0]
        if s >= high_t:
            return CONFIDENCE_LEVELS[2]
        return CONFIDENCE_LEVELS[1]
