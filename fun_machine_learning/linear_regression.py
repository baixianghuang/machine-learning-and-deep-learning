import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train and y_train must be equal"
        X_b = np.hstack(X_train, np.ones((len(X_train), 1)))
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None, "Invoke fit_normal first"
        assert X_predict.shape[0] == len(self.coef_), "The number of features must be equal"
        X_b = np.hstack(X_predict, np.ones((len(X_predict), 1)))
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

