import numpy as np


class StandardScalerImpl:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        # Compute mean and standard deviation of each column
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    # Transform X to
    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.scale_ is not None and self.mean_ is not None, "Must invoke fit() first"
        assert X.shape[1] == len(self.mean_), "The number of features must be equal to the length of mean_ and std_"
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX