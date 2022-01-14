class Parameters:
    def __init__(self, subject, filters, autoreject, features, channels=None, train_score=None, test_score=None, classifier=None, others=None):
        self.subject = subject
        self.filters = filters
        self.autoreject = 'autoreject' if autoreject is True else 'no_autoreject'
        self.features = features
        self.channels = channels if channels is not None else 64
        self.train_score = train_score if train_score is not None else 0
        self.test_score = test_score if test_score is not None else 0
        self.classifier = classifier if classifier is not None else ''
        self.others = others if others is not None else ''
