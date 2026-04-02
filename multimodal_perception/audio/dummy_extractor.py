class DummyFeatureExtractor:
    """
    Placeholder feature extractor.

    Returns a fixed mock feature set regardless of the provided audio.
    This is intended as a stand-in until the real extractor is integrated.
    """

    def extract(self, audio_path):
        """
        Return a mock feature dictionary for the given audio file.

        Parameters
        ----------
        audio_path : str or None
            Path to the recorded WAV file (currently unused by the dummy).

        Returns
        -------
        dict
            A placeholder feature set with zero/neutral values.
        """
        return {
            "duration": 0.0,
            "pause_count": 0,
            "pause_mean": 0.0,
            "pause_max": 0.0,
            "speech_rate": 0.0,
            "filler_count": 0,
            "repetition_count": 0,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_slope": 0.0,
            "energy_mean": 0.0,
            "jitter": 0.0,
            "shimmer": 0.0,
            "hnr": 0.0,
            "verbal_hesitation_count": 0,
            "mfcc_1_mean": 0.0,
            "mfcc_2_mean": 0.0,
        }
