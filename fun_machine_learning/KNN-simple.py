import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], "the size of X_train and y_train must be equal"
    assert X_train.shape[1] == x.shape[0], "the number of features of x must be equal to that of X_train"

    distances = [sqrt(np.sum((point - x) ** 2)) for point in X_train]
    nearest = np.argsort(distances)
    k_nearest = [y_train[i] for i in nearest[:k]]
    votes = Counter(k_nearest)
    return votes.most_common(1)[0][0]