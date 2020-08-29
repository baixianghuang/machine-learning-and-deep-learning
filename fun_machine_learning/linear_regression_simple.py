import numpy as np
from .metrics import r2_score


class LinearRegressionSimple1:
    """only support sample with 1 feature"""

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "This approach only works for data with 1 feature"
        assert len(x_train) == len(y_train), "The length of x_train and y_train must be equal"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        numerator = 0.0
        denominator = 0.0
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) ** 2

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """To predict multiple data at once, x_predict is a vector"""
        assert x_predict.ndim == 1, "This approach only works for data with 1 feature"
        assert self.a_ is not None and self.b_ is not None, "fit() must be invoke first"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return x_single * self.a_ + self.b_

    def __repr__(self):
        return "LinearRegressionSimple1"


class LinearRegressionSimple2:
    """
    Better performance than LinearRegressionSimple1
    In fit() function, use vector1.dot(vercor2) to improve performance
    """

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "This approach only works for data with 1 feature"
        assert len(x_train) == len(y_train), "The length of x_train and y_train must be equal"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        numerator = (x_train - x_mean).dot(y_train - y_mean)
        denominator = (x_train - x_mean).dot(x_train - x_mean)
        # for x_i, y_i in zip(x_train, y_train):
        #     numerator += (x_i - x_mean) * (y_i - y_mean)
        #     denominator += (x_i - x_mean) ** 2


        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """To predict multiple data at once, x_predict is a vector"""
        assert x_predict.ndim == 1, "This approach only works for data with 1 feature"
        assert self.a_ is not None and self.b_ is not None, "fit() must be invoke first"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return x_single * self.a_ + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegressionSimple2"
