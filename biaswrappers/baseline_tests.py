from biaswrappers.regressor import BiasRegressorC1, BiasRegressorC2, RandomWrapper
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

def test_regression():

    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = [RandomWrapper(), BiasRegressorC2(), BiasRegressorC1(), LinearRegression()]

    for model in models:
        model.fit(X_train, y_train)

        preds = model.predict(X_test).reshape(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print("Regression Results:")
        try:
            print("Penalty Value: {}".format(model.p))
        except:
            pass
        print("RMSE for {0}: {1}\n".format(get_model_name(model), rmse))