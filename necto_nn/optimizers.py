from numba import typed, float64
from numba.experimental import jitclass
from typing import List
import numpy as np


class Optimizer:
    learning_rate: float
    W: List[float64[:, ::1]]
    dW: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def init(self, W, dW):
        self.W, self.dW = W, dW

    def update(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def update(self):
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * self.dW[i]


class Momentum(Optimizer):
    """SGD with Momentum"""

    momentum: float
    mW: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def init(self, W, dW):
        self.W, self.dW = W, dW
        self.mW = typed.List([np.zeros_like(x) for x in W])

    def update(self):
        for i in range(len(self.W)):
            # update momentum
            self.mW[i][...] = self.momentum * self.mW[i] + self.dW[i]

            # update weights
            self.W[i] -= self.learning_rate * self.mW[i]


class NAG(Momentum):
    """Nesterov Accelerated Gradient"""

    def update(self):
        for i in range(len(self.W)):
            # update momentum
            self.mW[i][...] = self.momentum * self.mW[i] + self.dW[i]

            # Update weights with nestrov momentum
            self.W[i] -= self.learning_rate * (self.momentum * self.mW[i] + self.mW[i])


class RMSProp(Optimizer):
    """RMSProp"""

    beta: float
    eps: float
    sW: List[float64[:, ::1]]

    def __init__(self, learning_rate=1e-3, beta=0.9, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def init(self, W, dW):
        self.W, self.dW = W, dW
        # Initialize adaptive learning rate
        self.sW = typed.List([np.zeros_like(x) for x in W])

    def update(self):
        for i in range(len(self.W)):
            # Update Adaptive learning rate
            self.sW[i][...] = self.beta * self.sW[i] + (1 - self.beta) * (
                self.dW[i] ** 2
            )

            # update weights
            self.W[i] -= (
                self.learning_rate * self.dW[i] / (np.sqrt(self.sW[i]) + self.eps)
            )


class Adam(Optimizer):
    """Adam Optimizer"""

    beta1: float
    beta2: float
    eps: float
    mW: List[float64[:, ::1]]
    vW: List[float64[:, ::1]]
    t: int

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def init(self, W, dW):
        self.W, self.dW = W, dW
        self.mW = typed.List([np.zeros_like(x) for x in W])
        self.vW = typed.List([np.zeros_like(x) for x in W])

    def update(self):
        self.t += 1
        for i in range(len(self.W)):
            # Update Momentum
            self.mW[i][...] = self.beta1 * self.mW[i] + (1 - self.beta1) * self.dW[i]

            # Update Adaptive Learning Rate
            self.vW[i][...] = self.beta2 * self.vW[i] + (1 - self.beta2) * (
                self.dW[i] ** 2
            )

            # Bias correction for momentum
            mW_hat = self.mW[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)

            # Upddate weights
            self.W[i] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.eps)


class NAdam(Adam):
    """NAdam Optimizer (Nesterov-accelerated Adaptive Moment Estimation)

    Reference: https://optimization.cbe.cornell.edu/index.php?title=Nadam
    """

    def update(self):
        self.t += 1
        for i in range(len(self.W)):
            # Update Momentum
            self.mW[i][...] = self.beta1 * self.mW[i] + (1 - self.beta1) * self.dW[i]

            # Update Adaptive Learning Rate
            self.vW[i][...] = self.beta2 * self.vW[i] + (1 - self.beta2) * (
                self.dW[i] ** 2
            )

            # Bias correction for momentum
            mW_hat = self.mW[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)

            # Nestrov
            mW_prime = self.beta1 * mW_hat + (1 - self.beta1) * self.dW[i] / (
                1 - self.beta1**self.t
            )

            # Upddate weights with nestrov
            self.W[i] -= self.learning_rate * mW_prime / (np.sqrt(vW_hat) + self.eps)


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


def get_optimizer(name, *args, **kwargs):
    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' not found.")
    return jit_optimizers[name](*args, **kwargs)
