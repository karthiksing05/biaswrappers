import numpy as np

from sklearn.linear_model import LinearRegression

class FakeWrapper(object):

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
            self.numFtrs = y.shape[1]
        except IndexError:
            self.numFtrs = 1

        self.pLst = [np.random.rand() for _ in range(self.numFtrs)]
        self.over_under_lst = [np.random.choice([-1, 1]) for _ in range(self.numFtrs)]

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

    def __init__(self, model=LinearRegression(), deadband=0.05):
        self.model = model
        self.deadband = deadband
        self.over_under_lst = []
        self.pLst = []
        self.numFtrs = 0
        self.over_under = 0

    def get_params(self, deep=False):
        return {"model": self.model, "deadband": self.deadband}

    def fit(self, X: np.ndarray, y: np.ndarray, split_size: float = 0.45):
        '''
        This function does a fit of the given with penalty calculation along the way.
        '''

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same shape.")
        split_idx = int(round(float(len(X) * split_size)))

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = y[:split_idx]
        y_val = y[split_idx:]

        try:
            self.numFtrs = y.shape[1]
        except IndexError:
            self.numFtrs = 1

        self.over_under_lst = [0] * self.numFtrs
        self.pLst = [0] * self.numFtrs

        self.model.fit(X_train, y_train)

        for idx, ftr in enumerate(X_val):
            ftr = ftr.reshape(1, -1)
            y_preds = self.model.predict(ftr)
            for i in range(len(y_preds)):
                if self.over_under_lst[i] > 0:
                    y_preds[i] += self.pLst[i]
                elif self.over_under_lst[i] < 0:
                    y_preds[i] -= self.pLst[i]
            self._calibrate(y_preds, y_val[idx])
        return self.model

    def _calibrate(self, y_pred, y_val):
        if type(y_val) != np.ndarray:
            y_val = np.array([y_val])
        if len(list(y_pred.shape)) > 1:
            y_pred = y_pred[0]

        for i in range(len(y_val)):
            error = float(y_val[i] - y_pred[i])
            if error > self.deadband:
                self.over_under_lst[i] += 1
            elif error < self.deadband:
                self.over_under_lst[i] -= 1
            self.pLst[i] = (self.pLst[i] + np.abs(error)) / 2

    def predict(self, X: np.ndarray):
        y_preds = np.array([self.model.predict(X)])
        for i in range(len(y_preds)):
            if self.over_under_lst[i] > 0:
                y_preds[i] += self.pLst[i]
            elif self.over_under_lst[i] < 0:
                y_preds[i] -= self.pLst[i]
        return y_preds


class BiasRegressorC2(object):

    def __init__(self, model=LinearRegression(), postModel=LinearRegression()):
        self.model = model
        self.postModel = postModel
        self.totalRMSE = 0

    def get_params(self, deep=False):
        return {"model": self.model, "postModel": self.postModel}

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