import numpy as np
from .utils import einsum


class Activation:
    """Base class for activation functions."""

    def __call__(self, X):
        raise NotImplementedError

    def gradient(self, out):
        raise NotImplementedError

    def backward(self, gradient, out):
        """Backpropogates the inward gradient through the activation function.

        Args:
            gradient: Inward gradient
            out: Output of the activation.
        """
        return gradient * self.gradient(out)


class Identity(Activation):
    """Identity activation function: f(X) = X"""

    def __call__(self, X):
        return X

    def gradient(self, out):
        return np.ones_like(out)


class Sigmoid(Activation):
    """Sigmoid activation function: f(X) = 1 / (1 + exp(-X))"""

    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def gradient(self, out):
        return out * (1 - out)


class ReLU(Activation):
    """ReLU activation function: f(X) = max(0, X)"""

    def __call__(self, X):
        return np.maximum(0, X)

    def gradient(self, out):
        # since the output of ReLU is non negative
        return np.sign(out)


class TanH(Activation):
    """Hyperbolic tangent activation function: f(X) = tanh(X)"""

    def __call__(self, X):
        return np.tanh(X)

    def gradient(self, out):
        return 1 - out**2


class SoftMax(Activation):
    """SoftMax activation function: f(X) = exp(X) / sum(exp(X))"""

    def __init__(self, eps=1e-9):
        self.eps = eps

    def __call__(self, X):
        exp = np.exp(X - np.max(X, axis=-1, keepdims=True))  # For numerical stability
        return exp / (exp.sum(axis=-1, keepdims=True) + self.eps)

    def gradient(self, out):
        """Computes the Jacobian matrix of SoftMax."""

        return np.einsum("...i,ij->...ij", out, np.eye(out.shape[-1])) - np.einsum(
            "...i,...j->...ij", out, out
        )

    def backward(self, gradient, out):
        return einsum("b...,b...o->bo", gradient, self.gradient(out))


# Dictionary for activation functions
activations = {
    "identity": Identity,
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "tanh": TanH,
    "softmax": SoftMax,
}


def get_activation(name):
    return activations[name]


# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2, -1], [-1, 0, 3]])

    act = get_activation("relu")
    print("ReLU Output:\n", act(X))
    print("ReLU Gradient:\n", act.gradient(X))

    act = get_activation("softmax")
    print("SoftMax Output:\n", act(X))
    print("SoftMax Gradient:\n", act.gradient(X))
