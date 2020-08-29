import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """The accuracy score between y_true and y_predict"""
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to that of y_predict"
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """MSE between y_true and y_predict"""
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to hat of y_predict"
    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """RMSE  between y_true and y_predict"""
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """MAE between y_true and y_predict"""
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """R square between y_true and y_predict"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
