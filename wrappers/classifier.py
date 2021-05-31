import numpy as np

from models.bayes import NaiveBayes

from pgc.base import PGCBase


class PGCClassifier(PGCBase):

    def __init__(self, model=NaiveBayes()):
        self.model = model
        super().__init__(self.model, 'clf')
        self.p = 0
        self.over_under = 0

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        self.p, self.model, self.over_under = super().fit_init(X, y, verbose=verbose)

    def predict(self, X: np.ndarray):
        mean_y_preds = []
        num_times = 3
        for x in range(num_times):
            y_preds = self.model.predict(X)
            if self.over_under > 0:
                y_preds = np.array([round(pred + self.p) for pred in y_preds])
            elif self.over_under < 0:
                y_preds = np.array([round(pred - self.p) for pred in y_preds])
            mean_y_preds.append(y_preds)

        mean_y_preds = np.mean(
            np.array([array for array in mean_y_preds]), axis=0)
        return mean_y_preds
