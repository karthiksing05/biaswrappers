from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

def test_regression(model=None):

    X, y = make_regression(n_samples=1000, n_features=10, n_targets=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if not model:
        models = [LinearRegression()]
    else:
        models = [model, LinearRegression()]

    for model in models:
        model.fit(X_train, y_train)

        preds = model.predict(X_test).reshape(-1, len(y[0]))
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print("Regression Results:")
        print("y_test Mean: {}".format(np.mean(y_test)))
        print("RMSE for {0}: {1}\n".format(get_model_name(model), rmse))