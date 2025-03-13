from numba import types, njit
from numba.experimental import jitclass
import numpy as np
from functools import cached_property


f_type = types.float64[:, ::1](types.float64[:, ::1]).as_type()
b_type = types.float64[:, ::1](types.float64[:, ::1], types.float64[:, ::1]).as_type()


@jitclass
class JitActivation:
    forward: f_type
    backward: b_type

    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


class Activation:
    """Base class for activation functions."""

    def __init__(self):
        pass

    @cached_property
    def forward(self):
        raise NotImplementedError

    @cached_property
    def gradient(self):
        raise NotImplementedError

    @cached_property
    def backward(self):
        """Backpropogates the inward gradient through the activation function.

        Args:
            gradient: Inward gradient
            X: output of the activation.
        """

        gradient_func = self.gradient

        @njit
        def f(loss_gradient, X):
            return loss_gradient * gradient_func(X)

        return f

    def compiled(self):
        return JitActivation(self.forward, self.backward)


class Identity(Activation):
    """Identity activation function: f(X) = X"""

    @cached_property
    def forward(self):
        @njit
        def f(X):
            return X

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(X):
            return np.ones_like(X)

        return f


class Sigmoid(Activation):
    """Sigmoid activation function: f(X) = 1 / (1 + exp(-X))"""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    @cached_property
    def forward(self):
        eps = self.eps

        @njit
        def f(X):
            return 1 / (1 + np.exp(-X + eps))

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(X):
            return X * (1 - X)

        return f


class ReLU(Activation):
    """ReLU activation function: f(X) = max(0, X)"""

    @cached_property
    def forward(self):
        @njit
        def f(X):
            return np.maximum(0, X)

        return f

    @cached_property
    def gradient(self):
        # since the output of ReLU is non negative
        @njit
        def f(X):
            return np.sign(X)

        return f


class TanH(Activation):
    """Hyperbolic tangent activation function: f(X) = tanh(X)"""

    @cached_property
    def forward(self):
        @njit
        def f(X):
            return np.tanh(X)

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(X):
            return 1 - X**2

        return f


class SoftMax(Activation):
    """SoftMax activation function: f(X) = exp(X) / sum(exp(X))"""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    @cached_property
    def softmax(self):
        """Softmax for 1-dim"""
        eps = self.eps

        @njit
        def f(x):
            exp = np.exp(x - x.max())
            return exp / (exp.sum() + eps)

        return f

    @cached_property
    def forward(self):
        softmax = self.softmax

        @njit
        def f(X):
            if X.ndim == 1:
                return softmax(X)
            if X.ndim == 2:
                out = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    out[i] = softmax(X[i])
                return out

        return f

    @cached_property
    def gradient(self):
        """Computes the Jacobian matrix of SoftMax."""

        @njit
        def f(X):
            out = np.zeros(X.shape + (X.shape[-1],))
            for i in range(X.shape[-1]):
                for j in range(X.shape[-1]):
                    out[..., i, j] = -X[..., i] * (X[..., j])
                    if i == j:
                        out[..., i, j] += X[..., i]
            return out

        return f

    @cached_property
    def backward(self):
        gradient_func = self.gradient

        @njit
        def f(gradient, X):
            out = np.zeros(X.shape)
            act_gradient = gradient_func(X)
            for n in range(X.shape[0]):
                out[n] = act_gradient[n] @ gradient[n]
            return out

        return f


# class NpSoftMax(Activation):
#     """Numpy implementation of softMax activation function: f(X) = exp(X) / sum(exp(X))"""

#     def __init__(self, eps=1e-100):
#         self.eps = eps

#     def forward(self, X):
#         exp = np.exp(X - np.max(X, axis=-1, keepdims=True))  # For numerical stability
#         return exp / (exp.sum(axis=-1, keepdims=True) + self.eps)

#     def gradient(self, X):
#         """Computes the Jacobian matrix of SoftMax."""

#         return np.einsum("...i,ij->...ij", X, np.eye(X.shape[-1])) - np.einsum(
#             "...i,...j->...ij", X, X
#         )

#     def backward(self, gradient, out):
#         return einsum("b...,b...o->bo", gradient, self.gradient(out))


activations = {
    "identity": Identity,
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "tanh": TanH,
    "softmax": SoftMax,
}


def get_activation(name):
    return activations[name]
