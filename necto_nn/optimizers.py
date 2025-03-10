import numpy as np
from .nn import FeedForwardNetwork


class Optimizer:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def init(
        self,
        model: FeedForwardNetwork,
    ):
        self.model = model

    def update(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def update(self):
        for i in range(len(self.model.W)):
            self.model.W[i] -= self.learning_rate * self.model.dW[i]
            self.model.b[i] -= self.learning_rate * self.model.db[i]


class Momentum(Optimizer):
    """SGD with Momentum"""

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.mW = None
        self.mb = None

    def init(self, model: FeedForwardNetwork, *args, **kwargs):
        super().init(model)
        self.mW = [np.zeros_like(W) for W in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]

    def update(self):
        for i in range(len(self.model.W)):
            # update momentum
            self.mW[i] = self.momentum * self.mW[i] + self.model.dW[i]
            self.mb[i] = self.momentum * self.mb[i] + self.model.db[i]

            # update weights
            self.model.W[i] -= self.learning_rate * self.mW[i]
            self.model.b[i] -= self.learning_rate * self.mb[i]


class NAG(Momentum):
    """Nesterov Accelerated Gradient"""

    def update(self):
        for i in range(len(self.model.W)):
            # update momentum
            self.mW[i] = self.momentum * self.mW[i] + self.model.dW[i]
            self.mb[i] = self.momentum * self.mb[i] + self.model.db[i]

            # Update weights with nestrov momentum
            self.model.W[i] -= self.learning_rate * (
                self.momentum * self.mW[i] + self.mW[i]
            )
            self.model.b[i] -= self.learning_rate * (
                self.momentum * self.mb[i] + self.mb[i]
            )


class RMSProp(Optimizer):
    """RMSProp"""

    def __init__(self, learning_rate=1e-3, beta=0.9, eps=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = eps
        self.sW = None
        self.sb = None

    def init(self, model: FeedForwardNetwork):
        super().init(model)
        # Initialize adaptive learning rate
        self.sW = [np.zeros_like(W) for W in model.W]
        self.sb = [np.zeros_like(b) for b in model.b]

    def update(self):
        for i in range(len(self.model.W)):
            # Update Adaptive learning rate
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (
                self.model.dW[i] ** 2
            )
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (
                self.model.db[i] ** 2
            )

            # update weights
            self.model.W[i] -= (
                self.learning_rate * self.model.dW[i] / (np.sqrt(self.sW[i]) + self.eps)
            )
            self.model.b[i] -= (
                self.learning_rate * self.model.db[i] / (np.sqrt(self.sb[i]) + self.eps)
            )


class Adam(Optimizer):
    """Adam Optimizer"""

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = None
        self.vW = None
        self.mb = None
        self.vb = None
        self.t = 0

    def init(self, model: FeedForwardNetwork):
        super().init(model)
        self.mW = [np.zeros_like(W) for W in model.W]
        self.vW = [np.zeros_like(W) for W in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]
        self.vb = [np.zeros_like(b) for b in model.b]

    def update(self):
        self.t += 1
        for i in range(len(self.model.W)):
            # Update Momentum
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * self.model.dW[i]
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * self.model.db[i]

            # Update Adaptive Learning Rate
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (
                self.model.dW[i] ** 2
            )
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (
                self.model.db[i] ** 2
            )

            # Bias correction for momentum
            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            # Upddate weights
            self.model.W[i] -= (
                self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.eps)
            )
            self.model.b[i] -= (
                self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.eps)
            )


class NAdam(Adam):
    """NAdam Optimizer (Nesterov-accelerated Adaptive Moment Estimation)

    Reference: https://optimization.cbe.cornell.edu/index.php?title=Nadam
    """

    def update(self):
        self.t += 1
        for i in range(len(self.model.W)):
            # Update Momentum
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * self.model.dW[i]
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * self.model.db[i]

            # Update Adaptive Learning Rate
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (
                self.model.dW[i] ** 2
            )
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (
                self.model.db[i] ** 2
            )

            # Bias correction for momentum
            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)

            # Bias correction for Adaptive learning rate
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            # Nestrov
            mW_prime = self.beta1 * mW_hat + (1 - self.beta1) * self.model.dW[i] / (
                1 - self.beta1**self.t
            )
            mb_prime = self.beta1 * mb_hat + (1 - self.beta1) * self.model.db[i] / (
                1 - self.beta1**self.t
            )

            # Upddate weights with nestrov
            self.model.W[i] -= (
                self.learning_rate * mW_prime / (np.sqrt(vW_hat) + self.eps)
            )
            self.model.b[i] -= (
                self.learning_rate * mb_prime / (np.sqrt(vb_hat) + self.eps)
            )


optimizers = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": NAdam,
}


def get_optimizer(name, *args, **kwargs):
    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' not found.")
    return optimizers[name](*args, **kwargs)
