from typing import List

import numpy as np
from numba import float64, typed
from numba.experimental import jitclass


class Optimizer:
    learning_rate: float
    params: List[float64[:, ::1]]
    gradients: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def init(self, params, gradients):
        """Attaches the model weights and gradients to the optimizer.

        Args:
            params (list[2d ndarray]):
                list of numpy array consisting of the weights and biases.
                Biases can be attached by taking a 2d view using np.newaxis.
            gradients(list[2d ndarray]):
                The corresponding gradient to the parameters.

        """
        self.params, self.gradients = params, gradients

    def update(self):
        """Update the weights using inplace updates in the params.

        Any subclass of this should update the params using inplace operators
        like +=, -=, *= , /= or inplace update using [...] slice update.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def update(self):
        for i in range(len(self.params)):
            self.params[i] -= self.learning_rate * self.gradients[i]


class Momentum(Optimizer):
    """SGD paramsith Momentum."""

    momentum: float
    mparams: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def init(self, params, gradients):
        self.params, self.gradients = params, gradients
        self.mparams = typed.List([np.zeros_like(x) for x in params])

    def update(self):
        for i in range(len(self.params)):
            # update momentum
            self.mparams[i][...] = self.momentum * self.mparams[i] + self.gradients[i]

            # update paramseights
            self.params[i] -= self.learning_rate * self.mparams[i]


class NAG(Momentum):
    """Nesterov Accelerated Gradient."""

    def update(self):
        for i in range(len(self.params)):
            # update momentum
            self.mparams[i][...] = self.momentum * self.mparams[i] + self.gradients[i]

            # Update paramseights paramsith nestrov momentum
            self.params[i] -= self.learning_rate * (
                self.momentum * self.mparams[i] + self.mparams[i]
            )


class RMSProp(Optimizer):
    """RMSProp."""

    beta: float
    eps: float
    sparams: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3, beta=0.9, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def init(self, params, gradients):
        self.params, self.gradients = params, gradients
        # Initialize adaptive learning rate
        self.sparams = typed.List([np.zeros_like(x) for x in params])

    def update(self):
        for i in range(len(self.params)):
            # Update Adaptive learning rate
            self.sparams[i][...] = self.beta * self.sparams[i] + (1 - self.beta) * (
                self.gradients[i] ** 2
            )

            # update paramseights
            self.params[i] -= (
                self.learning_rate
                * self.gradients[i]
                / (np.sqrt(self.sparams[i]) + self.eps)
            )


class Adam(Optimizer):
    """Adam Optimizer."""

    beta1: float
    beta2: float
    eps: float
    mparams: List[float64[:, ::1]]
    vparams: List[float64[:, ::1]]
    t: int

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def init(self, params, gradients):
        self.params, self.gradients = params, gradients
        self.mparams = typed.List([np.zeros_like(x) for x in params])
        self.vparams = typed.List([np.zeros_like(x) for x in params])

    def update(self):
        self.t += 1
        for i in range(len(self.params)):
            # Update Momentum
            self.mparams[i][...] = (
                self.beta1 * self.mparams[i] + (1 - self.beta1) * self.gradients[i]
            )

            # Update Adaptive Learning Rate
            self.vparams[i][...] = self.beta2 * self.vparams[i] + (1 - self.beta2) * (
                self.gradients[i] ** 2
            )

            # Bias correction for momentum
            mparams_hat = self.mparams[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vparams_hat = self.vparams[i] / (1 - self.beta2**self.t)

            # Upddate paramseights
            self.params[i] -= (
                self.learning_rate * mparams_hat / (np.sqrt(vparams_hat) + self.eps)
            )


class NAdam(Adam):
    """NAdam Optimizer (Nesterov-accelerated Adaptive Moment Estimation).

    Reference: https://optimization.cbe.cornell.edu/index.php?title=Nadam
    """

    def update(self):
        self.t += 1
        for i in range(len(self.params)):
            # Update Momentum
            self.mparams[i][...] = (
                self.beta1 * self.mparams[i] + (1 - self.beta1) * self.gradients[i]
            )

            # Update Adaptive Learning Rate
            self.vparams[i][...] = self.beta2 * self.vparams[i] + (1 - self.beta2) * (
                self.gradients[i] ** 2
            )

            # Bias correction for momentum
            mparams_hat = self.mparams[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vparams_hat = self.vparams[i] / (1 - self.beta2**self.t)

            # Nestrov
            mparams_prime = self.beta1 * mparams_hat + (
                1 - self.beta1
            ) * self.gradients[i] / (1 - self.beta1**self.t)

            # Upddate paramseights paramsith nestrov
            self.params[i] -= (
                self.learning_rate * mparams_prime / (np.sqrt(vparams_hat) + self.eps)
            )


def jittify(cls):
    @jitclass
    class cls_new(cls): ...

    cls_new.__name__ = f"Jit{cls.__name__}"
    cls_new.__qualname__ = f"Jit{cls.__qualname__}"
    return cls_new


optimizers = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": NAdam,
}
jit_optimizers = {k: jittify(v) for k, v in optimizers.items()}


def get_optimizer(name, *args, **kparamsargs):
    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' not found.")
    return jit_optimizers[name](*args, **kparamsargs)
