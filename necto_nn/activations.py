from functools import cached_property

import numpy as np
from numba import njit, types
from numba.experimental import jitclass

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
            x: output of the activation.

        """
        gradient_func = self.gradient

        @njit
        def f(loss_gradient, x):
            return loss_gradient * gradient_func(x)

        return f

    def compiled(self):
        return JitActivation(self.forward, self.backward)


class Identity(Activation):
    """Identity activation function: f(x) = x."""

    @cached_property
    def forward(self):
        @njit
        def f(x):
            return x

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(x):
            return np.ones_like(x)

        return f


class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))."""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    @cached_property
    def forward(self):
        eps = self.eps

        @njit
        def f(x):
            return 1 / (1 + np.exp(-x + eps))

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(x):
            return x * (1 - x)

        return f


class ReLU(Activation):
    """ReLU activation function: f(x) = max(0, x)."""

    @cached_property
    def forward(self):
        @njit
        def f(x):
            return np.maximum(0, x)

        return f

    @cached_property
    def gradient(self):
        # since the output of ReLU is non negative
        @njit
        def f(x):
            return np.sign(x)

        return f


class TanH(Activation):
    """Hyperbolic tangent activation function: f(x) = tanh(x)."""

    @cached_property
    def forward(self):
        @njit
        def f(x):
            return np.tanh(x)

        return f

    @cached_property
    def gradient(self):
        @njit
        def f(x):
            return 1 - x**2

        return f


class SoftMax(Activation):
    """SoftMax activation function: f(x) = exp(x) / sum(exp(x))."""

    eps: float

    def __init__(self, eps=1e-100):
        self.eps = eps

    @cached_property
    def softmax(self):
        """Softmax for 1-dim."""
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
        def f(x):
            if x.ndim == 1:
                return softmax(x)
            if x.ndim == 2:
                out = np.zeros(x.shape)
                for i in range(x.shape[0]):
                    out[i] = softmax(x[i])
                return out

        return f

    @cached_property
    def gradient(self):
        """Computes the Jacobian matrix of SoftMax."""

        @njit
        def f(x):
            out = np.zeros(x.shape + (x.shape[-1],))
            for i in range(x.shape[-1]):
                for j in range(x.shape[-1]):
                    out[..., i, j] = -x[..., i] * (x[..., j])
                    if i == j:
                        out[..., i, j] += x[..., i]
            return out

        return f

    @cached_property
    def backward(self):
        gradient_func = self.gradient

        @njit
        def f(gradient, x):
            out = np.zeros(x.shape)
            act_gradient = gradient_func(x)
            for n in range(x.shape[0]):
                out[n] = act_gradient[n] @ gradient[n]
            return out

        return f


# class NpSoftMax(Activation):
#     """Numpy implementation of softMax activation function: f(x) = exp(x) / sum(exp(x))"""

#     def __init__(self, eps=1e-100):
#         self.eps = eps

#     def forward(self, x):
#         exp = np.exp(x - np.max(x, axis=-1, keepdims=True))  # For numerical stability
#         return exp / (exp.sum(axis=-1, keepdims=True) + self.eps)

#     def gradient(self, x):
#         """Computes the Jacobian matrix of SoftMax."""

#         return np.einsum("...i,ij->...ij", x, np.eye(x.shape[-1])) - np.einsum(
#             "...i,...j->...ij", x, x
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
