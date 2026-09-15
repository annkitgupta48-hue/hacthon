class MovementTracker:
    def __init__(self):
        self.history = []

    def add(self, sample):
        self.history.append(sample)
        self.history = self.history[-10:]

    def detect(self):
        if not self.history:
            return "stationary"
        return "stable"
