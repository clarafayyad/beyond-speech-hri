class HeadPoseService:
    def __init__(self, window_size=30):
        self.positions = deque(maxlen=window_size)

    def update(self, landmarks):
        nose = landmarks["nose"]
        self.positions.append(nose)

    def classify(self):
        if len(self.positions) < 2:
            return "unknown"

        variance = compute_variance(self.positions)
        return "focused" if variance < THRESHOLD else "scanning"
