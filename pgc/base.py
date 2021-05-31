import numpy as np
from numpy import random

from sklearn import metrics


class PGCBase(object):

    def __init__(self, model, modeltype):
        self.model = model
        self.modeltype = modeltype
        self.m_vals = []
        self.k_vals = []
        self.ftr_means = []
        self.over_under_lst = []

    def fit_init(self, X: np.ndarray, y: np.ndarray, split_size: float = 0.5, verbose: bool = False):
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
            self.ftr_means.append(sum(lst)/len(lst))
        self.ftr_means = np.array(self.ftr_means).reshape(1, -1)

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

        for idx, ftr in enumerate(X_val):
            ftr = ftr.reshape(1, -1)
            y_preds = self.model.predict(ftr)
            if self.modeltype == 'reg':
                if sum(self.over_under_lst) > 0:
                    y_preds = np.array([pred + p for pred in y_preds])
                elif sum(self.over_under_lst) < 0:
                    y_preds = np.array([pred - p for pred in y_preds])
            elif self.modeltype == 'clf':
                if sum(self.over_under_lst) > 0:
                    y_preds = np.array([round(pred + p) for pred in y_preds])
                elif sum(self.over_under_lst) < 0:
                    y_preds = np.array([round(pred - p) for pred in y_preds])
            p = self._calibrate(y_preds, y_val[idx])
        return p, self.model, sum(self.over_under_lst)

    def _calibrate(self, y_pred, y_val):
        if type(y_val) != np.ndarray:
            y_val = np.array([y_val])
        for i in range(len(y_val)):
            if self.modeltype == 'reg':
                error = float(y_val[i] - y_pred[i])
                if error > 0.05:
                    self.over_under_lst.append(1)
                elif error < -0.05:
                    self.over_under_lst.append(-1)
                else:
                    self.over_under_lst.append(0)
            elif self.modeltype == 'clf':
                error = len(y_val) - (((y_pred[i] == y_val[i]).sum()) / len(y_val))
                clf_val = 1 / np.unique(y_val)
                if error > clf_val:
                    self.over_under_lst.append(1)
                elif 0 < error < clf_val:
                    self.over_under_lst.append(-1)
                else:
                    self.over_under_lst.append(0)
            m_val = float(self.m_vals[i])
            if self.modeltype == 'reg':
                k = np.abs(error) / (m_val + 1)
            elif self.modeltype == 'clf':
                k = (np.abs(error) + m_val) / 2
            self.k_vals.append(k)
        return float(sum(self.k_vals) / len(self.k_vals))
