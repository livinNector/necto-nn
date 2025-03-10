import numpy as np


class Loss:
    """Base class for all loss functions."""

    def __call__(self, y, y_pred):
        raise NotImplementedError

    def gradient(self, y, y_pred):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error (MSE) Loss."""

    def __call__(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def gradient(self, y, y_pred):
        return 2 * (y - y_pred) / y.shape[0]


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss where y is one-hot encoded."""

    def __init__(self, eps=1e-100):
        self.eps = eps

    def __call__(self, y, y_pred):
        return -np.mean(np.sum(y * np.log(y_pred + self.eps), axis=-1))

    def gradient(self, y, y_pred):
        return -y / (y_pred + self.eps)


class SparseCrossEntropyLoss(Loss):
    """Cross Entropy Loss where y contains class labels."""

    def __init__(self, eps=1e-100):
        self.eps = eps

    def __call__(self, y, y_pred):
        return -np.mean(np.log(y_pred[np.arange(y.shape[0]), y]))

    def gradient(self, y, y_pred):
        return -y / (y_pred + self.eps)


losses = {
    "mean_squared_error": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "sparse_cross_entropy": SparseCrossEntropyLoss,
}


def get_loss(name, *args, **kwargs):
    """Retrieve and compute the specified loss function."""
    loss_cls = losses.get(name)
    if loss_cls is None:
        raise ValueError(f"Invalid Loss Funciton: Loss function '{name}' not found.")

    return loss_cls(*args, **kwargs)
