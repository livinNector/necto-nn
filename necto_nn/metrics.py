from .utils import array_args
import numpy as np


@array_args
def accuracy(y, y_pred):
    return np.mean(y == y_pred)


@array_args
def precision(y, y_pred):
    """Precision for binary labels."""
    TP = (y & y_pred).sum()
    FP = (~y & y_pred).sum()
    return TP and TP / (TP + FP)


@array_args
def recall(y, y_pred):
    """Recall for binary labels."""
    TP = (y & y_pred).sum()
    FN = (y & ~y_pred).sum()
    return TP and TP / (TP + FN)


def micro_f1_score(y, y_pred):
    """F1 score for binary labels."""
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return p and r and 2 * p * r / (p + r)


def f1_score(y, y_pred):
    """Macro F1 score."""

    labels = np.unique(y)
    micro_f1_scores = []
    for label in labels:
        y_bin = y == label
        y_pred_bin = y_pred == label
        micro_f1_scores.append(micro_f1_score(y_bin, y_pred_bin))
    return np.mean(micro_f1_scores)


metrics = {"f1_score": f1_score, "accuracy": accuracy}


def get_metric(name):
    return metrics[name]
