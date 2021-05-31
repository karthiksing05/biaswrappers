import numpy as np

from models.linear import LinearRegression
# from sklearn.linear_model import LinearRegression

from pgc.base import PGCBase


class PGCRegressor(PGCBase):

    def __init__(self, model=LinearRegression()):
        self.model = model
        super().__init__(self.model, 'reg')
        self.p = 0
        self.over_under = 0

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        self.p, self.model, self.over_under = super().fit_init(X, y, verbose=verbose)

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        if self.over_under > 0:
            y_preds = np.array([pred + self.p for pred in y_preds])
        elif self.over_under < 0:
            y_preds = np.array([pred - self.p for pred in y_preds])
        return y_preds
