import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((int(self.num_examples/2), self.num_classes))
        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self._density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def _density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs

class LinearRegression:

    def __init__(self, lr=.005, epochs=550):
        self.costs = []
        self.b = 0
        self.w = []
        self.lr = lr
        self.epochs = epochs

    @classmethod
    def forward(self, x, w, b):
        y_pred = np.dot(x, w) + b
        return y_pred

    @classmethod
    def compute_cost(self, y_pred, y, n):
        cost = (1/(2*n)) * np.sum(np.square(y_pred - y))
        return cost

    @classmethod
    def backward(self, y_pred, y, x, n):
        # print(y_pred.shape, y.shape, x.shape)
        dw = (1/n) * np.dot(x.T, (y_pred-y))
        # print(dw.shape)
        db = (1/n) * np.sum((y_pred-y))
        return dw, db

    @classmethod
    def update(self, w, b, dw, db, lr):
        w -= lr*dw
        b -= lr*db
        return w, b

    @classmethod
    def normalize(self, df):
        result = df.copy()
        for feature_name in df.columns:
            mean = np.mean(df[feature_name])
            std = np.std(df[feature_name])
            result[feature_name] = (df[feature_name] - mean) / std
        return result

    @classmethod
    def initialize(self, m):
        w = np.random.normal(size=(m, 1))
        b = 0
        return (w, b)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        X = self.normalize(X)
        y = np.reshape(np.array(y), (len(y), 1))
        n, m = X.shape
        w, b = self.initialize(m)
        w = w
        b = b
        self.w, self.b = self.initialize(m)
        for _ in range(self.epochs):
            y_pred = self.forward(X, self.w, self.b)
            cost = self.compute_cost(y_pred, y, n)
            dw, db = self.backward(y_pred, y, X, n)
            self.w, self.b = self.update(self.w, self.b, dw, db, self.lr)
            self.costs.append(cost)

    def predict(self, X):
        X = pd.DataFrame(X)
        X = self.normalize(X)
        y_pred = self.forward(X, self.w, self.b)
        return y_pred
