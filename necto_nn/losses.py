from numba.experimental import jitclass
import numpy as np


class Loss:
    """Base class for all loss functions."""

    def __init__(self):
        pass

    def forward(self, y, y_pred):
        raise NotImplementedError

    def gradient(self, y, y_pred):
        raise NotImplementedError


@jitclass()
class MSELoss(Loss):
    """Mean Squared Error (MSE) Loss."""

    def forward(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def gradient(self, y, y_pred):
        return 2 * (y - y_pred) / y.shape[0]


@jitclass()
class CrossEntropyLoss(Loss):
    """Cross Entropy Loss where y is one-hot encoded."""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    def forward(self, y, y_pred):
        return -np.sum(y * np.log(y_pred + self.eps)) / y.shape[0]

    def gradient(self, y, y_pred):
        return -y / (y_pred + self.eps) / y.shape[0]


@jitclass
class SparseCrossEntropyLoss(Loss):
    """Cross Entropy Loss where y contains class labels."""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    def forward(self, y, y_pred):
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
