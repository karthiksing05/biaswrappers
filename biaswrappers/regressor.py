import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

class BiasRegressorC1(object):

    def __init__(self, model=LinearRegression(), deadband=0.05):
        self.model = model
        self.deadband = deadband
        self.m_vals = []
        self.k_vals = []
        self.ftr_means: list = []
        self.over_under_lst = []
        self.p = 0
        self.over_under = 0

    @classmethod
    def get_params(self):
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, split_size: float = 0.45):
        '''
        This function does a fit of the given with penalty calculation along the way.
        '''

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
        if len(list(self.m_vals.shape)) > 1:
            self.m_vals = self.m_vals[0]

        for idx, ftr in enumerate(X_val):
            ftr = ftr.reshape(1, -1)
            y_preds = self.model.predict(ftr)
            if sum(self.over_under_lst) > 0:
                y_preds = np.array([pred + self.p for pred in y_preds])
            elif sum(self.over_under_lst) < 0:
                y_preds = np.array([pred - self.p for pred in y_preds])
            self._calibrate(y_preds, y_val[idx])
        self.over_under = sum(self.over_under_lst) / len(self.over_under_lst)
        return self.model

    def _calibrate(self, y_pred, y_val):
        if type(y_val) != np.ndarray:
            y_val = np.array([y_val])
        if len(list(y_pred.shape)) > 1:
            y_pred = y_pred[0]
        for i in range(len(y_val)):
            error = float(y_val[i] - y_pred[i])
            if error > self.deadband:
                self.over_under_lst.append(1)
            elif error < self.deadband:
                self.over_under_lst.append(-1)
            else:
                self.over_under_lst.append(0)
            m_val = float(self.m_vals[i])
            k = np.abs(error) / (m_val + 1)
            self.k_vals.append(k)
        self.p = float(sum(self.k_vals) / len(self.k_vals))

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        if self.over_under > 0:
            y_preds = np.array([pred + self.p for pred in y_preds])
        elif self.over_under < 0:
            y_preds = np.array([pred - self.p for pred in y_preds])
        return y_preds


class BiasRegressorC2(object):

    def __init__(self, model=LinearRegression(), postModel=LinearRegression()):
        self.model = model
        self.postModel = postModel
        self.rmse = 0

    @classmethod
    def get_params(self):
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, split_size: float = 0.25):
        
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")
        split_idx = int(round(float(len(X) * split_size)))

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = y[:split_idx]
        y_val = y[split_idx:]

        self.model.fit(X_train, y_train)

        y_val_preds = self.model.predict(X_val)

        # add predicted answers to X_val and train the postModel
        newShape = list(X_val.shape)
        try:
            newShape[1] += len(list(y_val_preds[0]))
        except TypeError:
            newShape[1] += 1
        newX_val = np.ndarray(shape=tuple(newShape))
        for i in range(len(X_val)):
            newX_val[i] = np.append(X_val[i], y_val_preds[i])

        self.postModel.fit(newX_val, y_val)

    def predict(self, X: np.ndarray):
        prePreds = self.model.predict(X)

        newShape = list(X.shape)
        try:
            newShape[1] += len(list(prePreds[0]))
        except TypeError:
            newShape[1] += 1
        newX = np.ndarray(shape=tuple(newShape))
        for i in range(len(X)):
            newX[i] = np.append(X[i], prePreds[i])

        return self.postModel.predict(newX)