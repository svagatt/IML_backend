class Parameters:
    def __init__(self, subject, filters, autoreject, features, channels=None, train_score=None, test_score=None, other=None):
        self.subject = subject
        self.filters = filters
        self.autoreject = autoreject
        self.features = features
        self.channels = channels if channels is not None else 64
        self.train_score = train_score if train_score is not None else 0
        self.test_score = test_score if test_score is not None else 0
        self.other = other if other is not None else ''
