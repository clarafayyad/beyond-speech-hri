import random

CONFIDENCE_LEVELS = ["low", "medium", "high"]


class DummyConfidenceClassifier:
    """
    Placeholder confidence classifier.

    Returns a randomly sampled confidence label regardless of the provided
    features.  This is intended as a stand-in until the real model is
    integrated.
    """

    def classify(self, features):
        """
        Return a mock confidence level for the given feature dictionary.

        Parameters
        ----------
        features : dict
            Extracted audio features (currently unused by the dummy).

        Returns
        -------
        str
            One of ``"low"``, ``"medium"``, or ``"high"``.
        """
        return random.choice(CONFIDENCE_LEVELS)
