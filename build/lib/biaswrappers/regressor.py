import numpy as np

from sklearn.linear_model import LinearRegression

class BiasRegressor(object):

    def __init__(self, model=LinearRegression()):
        self.model = model
        self.m_vals = []
        self.k_vals = []
        self.ftr_means: list = []
        self.over_under_lst = []
        self.p = 0
        self.over_under = 0

    @classmethod
    def get_params(self):
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, split_size: float = 0.5, verbose: bool = False):
        '''
        This function does an initial fit with calibration along the way.
        '''

        p = 0

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")
        split_idx = int(round(float(len(X) * split_size)))
        ftr_lsts = [[] for i in range(X.shape[1])]
        for ftr in X:
            for idx, val in enumerate(ftr):
                ftr_lsts[idx].append(val)
        for lst in ftr_lsts:
            self.ftr_means = np.append(self.ftr_means, (sum(lst)/len(lst)))
        self.ftr_means = np.array(self.ftr_means).reshape(-1, len(X[0]))

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = y[:split_idx]
        y_val = y[split_idx:]

        self.model.fit(X_train, y_train)
        try:
            self.k_vals = np.zeros(y.shape[1]).tolist()
        except IndexError:
            self.k_vals = np.zeros(1).tolist()

        self.m_vals = self.model.predict(self.ftr_means)
        self.m_vals = self.m_vals[0]

        for idx, ftr in enumerate(X_val):
            ftr = ftr.reshape(1, -1)
            y_preds = self.model.predict(ftr)
            if sum(self.over_under_lst) > 0:
                y_preds = np.array([pred + p for pred in y_preds])
            elif sum(self.over_under_lst) < 0:
                y_preds = np.array([pred - p for pred in y_preds])
        p = self._calibrate(y_preds[0], y_val[idx])
        return p, self.model, sum(self.over_under_lst)

    def _calibrate(self, y_pred, y_val):
        if type(y_val) != np.ndarray:
            y_val = np.array([y_val])
        for i in range(len(y_val)):
            error = float(y_val[i] - y_pred[i])
            if error > 0.05:
                self.over_under_lst.append(1)
            elif error < -0.05:
                self.over_under_lst.append(-1)
            else:
                self.over_under_lst.append(0)
            m_val = float(self.m_vals[i])
            k = np.abs(error) / (m_val + 1)
            self.k_vals.append(k)
        return float(sum(self.k_vals) / len(self.k_vals))

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        if self.over_under > 0:
            y_preds = np.array([pred + self.p for pred in y_preds])
        elif self.over_under < 0:
            y_preds = np.array([pred - self.p for pred in y_preds])
        return y_preds
