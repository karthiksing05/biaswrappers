from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_boston, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np


def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]


def test_classification(model=None, mode=1):

    if mode == 0:
        data = load_digits()

        X = data['data']
        y = data['target']
    elif mode == 1:
        X, y = make_classification(
            n_samples=1000, n_informative=5, flip_y=0.8, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if not model:
        models = [LogisticRegression(max_iter=5000)]
    else:
        models = [model, LogisticRegression(max_iter=5000)]

    for model in models:
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        num_correct = str(int(cm[0][0]) + int(cm[1][1])) + "/" + str(np.sum(cm[0]) + np.sum(cm[1]))
        num_fp = str(int(cm[1][0]) + int(cm[0][1])) + "/" + str(np.sum(cm[0]) + np.sum(cm[1]))

        print("Classification Results:")
        print("Correct Answers out of total for {0}: {1}\n".format(get_model_name(model), num_correct))
        print("False Positives out of total for {0}: {1}\n".format(get_model_name(model), num_fp))

def test_regression(model=None, mode=0):

    if mode == 0:
        data = load_boston()

        X = data['data']
        y = data['target']

    elif mode == 1:
        X, y = make_regression(
            n_samples=1000, n_informative=5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if not model:
        models = [LinearRegression()]
    else:
        models = [model, LinearRegression()]

    for model in models:
        model.fit(X_train, y_train)

        preds = model.predict(X_test).reshape(-1, 1)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print("Regression Results:")
        print("y_test Mean: {}".format(np.mean(y_test)))
        print("RMSE for {0}: {1}\n".format(get_model_name(model), round(rmse, ndigits=9)))