import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    # The point of this function is to make this simple algorithm to follow the
    # pattern of other machine learning algorithms
    def fit(self, X_train, y_train):
        assert self.k <= X_train.shape[0], "k must be valid"
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train and y_train must be equal"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, "must invoke fit() before this function"
        assert self._X_train.shape[1] == X_predict.shape[1], "the number of features of x must be equal to that of X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        k_nearest = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(k_nearest)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k


