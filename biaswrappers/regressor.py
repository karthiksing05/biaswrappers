import numpy as np

from sklearn.linear_model import LinearRegression

class RandomWrapper(object):

    def __init__(self, model=LinearRegression()):
        self.model = model
        self.pLst = []
        self.over_under_lst = []

    def get_params(self, deep=False):
        return {"model": self.model}

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        This function does a fit of the given with penalty calculation along the way.
        '''

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")

        try:
            self.numTargets = y.shape[1]
        except IndexError:
            self.numTargets = 1

        self.pLst = [np.random.rand() for _ in range(self.numTargets)]
        self.over_under_lst = [np.random.choice([-1, 1]) for _ in range(self.numTargets)]

        self.model.fit(X, y)

        return self.model

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        for i in range(len(y_preds)):
            if self.over_under_lst[i] > 0:
                y_preds[i] += self.pLst[i]
            elif self.over_under_lst[i] < 0:
                y_preds[i] -= self.pLst[i]
        return y_preds

class BiasRegressorC1(object):

    def __init__(self, model=LinearRegression(), deadband=0.05, split_size=0.45):
        self.model = model
        self.deadband = deadband
        self.split_size = split_size
        self.over_under_lst = []
        self.pLst = []
        self.numTargets = 0
        self.ftr_means = []
        self.m_vals = []

    def get_params(self, deep=False):
        return {"model": self.model, "deadband": self.deadband, "split_size": self.split_size}

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        This function does a fit of the given with penalty calculation along the way.
        '''

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")
        split_idx = int(round(float(len(X) * self.split_size)))

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = y[:split_idx]
        y_val = y[split_idx:]

        ftr_lsts = [[] for i in range(X.shape[1])]
        for ftr in X:
            for idx, val in enumerate(ftr):
                ftr_lsts[idx].append(val)
        for lst in ftr_lsts:
            self.ftr_means = np.append(self.ftr_means, (sum(lst)/len(lst)))
        self.ftr_means = np.array(self.ftr_means).reshape(-1, len(X[0]))

        try:
            self.numTargets = y.shape[1]
        except IndexError:
            self.numTargets = 1

        self.over_under_lst = [[]] * self.numTargets
        self.pLst = [0] * self.numTargets

        self.model.fit(X_train, y_train)

        self.m_vals = self.model.predict(self.ftr_means)
        while len(list(self.m_vals.shape)) > 1:
            self.m_vals = self.m_vals[0]

        for idx, ftr in enumerate(X_val):
            ftr = ftr.reshape(1, -1)
            y_preds = self.model.predict(ftr)
            for i in range(len(y_preds)):
                if sum(self.over_under_lst[i]) > 0:
                    y_preds[i] += self.pLst[i]
                elif sum(self.over_under_lst[i]) < 0:
                    y_preds[i] -= self.pLst[i]
            self._calibrate(idx, y_preds, y_val[idx])
        return self.model

    def _calibrate(self, iteration, y_pred, y_val):
        if type(y_val) != np.ndarray:
            y_val = np.array([y_val])
        if len(list(y_pred.shape)) > 1:
            y_pred = y_pred[0]

        for i in range(len(y_val)):
            error = float(y_val[i] - y_pred[i])
            if error > self.deadband:
                self.over_under_lst[i].append(-1)
            elif error < self.deadband:
                self.over_under_lst[i].append(1)
            else:
                self.over_under_lst[i].append(0)
            self.pLst[i] = (self.pLst[i] + (np.abs(error) / (self.m_vals[i] + 1))) / (2)

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        for i in range(len(y_preds)):
            y_preds[i] += sum(self.over_under_lst[i]) / len(self.over_under_lst[i]) * self.pLst[i]
        return y_preds


class BiasRegressorC2(object):

    def __init__(self, model=LinearRegression(), postModel=LinearRegression(), split_size=0.45):
        self.model = model
        self.postModel = postModel
        self.split_size = split_size
        self.totalRMSE = 0
        self.numTargets = 0

    def get_params(self, deep=False):
        return {"model": self.model, "postModel": self.postModel}

    def fit(self, X: np.ndarray, y: np.ndarray):
        
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")
        split_idx = int(round(float(len(X) * self.split_size)))

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = y[:split_idx]
        y_val = y[split_idx:]

        self.model.fit(X_train, y_train)

        try:
            self.numTargets = y.shape[1]
        except IndexError:
            self.numTargets = 1

        y_val_preds = self.model.predict(X_val)

        if self.numTargets == 1:
            y_val_preds = np.array([[pred] for pred in y_val_preds])

        self.postModel.fit(y_val_preds, y_val)

    def predict(self, X: np.ndarray):
        prePreds = self.model.predict(X)
        if self.numTargets == 1:
            prePreds = np.array([[pred] for pred in prePreds])

        return self.postModel.predict(prePreds)