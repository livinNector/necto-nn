import numpy as np
from numba import njit

from .utils import one_hot


@njit
def accuracy(y, y_pred):
    return np.mean(y == y_pred)


@njit
def precision(y, y_pred):
    """Precision for binary labels."""
    tp = (y & y_pred).sum()
    fp = (~y & y_pred).sum()
    return tp and tp / (tp + fp)


@njit
def recall(y, y_pred):
    """Recall for binary labels."""
    tp = (y & y_pred).sum()
    fn = (y & ~y_pred).sum()
    return tp and tp / (tp + fn)


@njit
def micro_f1_score(y, y_pred):
    """F1 score for binary labels."""
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return p and r and 2 * p * r / (p + r)


@njit
def f1_score(y, y_pred):
    """Macro F1 score."""
    labels = np.unique(y)
    micro_f1_scores = np.zeros_like(labels, dtype=np.float64)
    for i, label in enumerate(labels):
        y_bin = y == label
        y_pred_bin = y_pred == label
        micro_f1_scores[i] = micro_f1_score(y_bin, y_pred_bin)
    return np.mean(micro_f1_scores)


def confusion_matrix(y, y_pred):
    """Confusion Matrix."""
    n = np.unique(y).shape[0]
    y, y_pred = one_hot(y, n), one_hot(y_pred, n)

    return (y[..., np.newaxis] & y_pred[..., np.newaxis, :]).sum(axis=0)


metrics = {"f1_score": f1_score, "accuracy": accuracy}


def get_metric(name):
    return metrics[name]
